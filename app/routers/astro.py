"""
astro.py — Astro Medha premium backend
"""

import os
import json
from datetime import date, datetime
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from supabase import create_client
from groq import Groq

from app.auth import verify_token

router = APIRouter()
_groq  = Groq(api_key=os.getenv("GROQ_API_KEY"))


def get_sb():
    return create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))


# ── Zodiac calculator ─────────────────────────────────────────────────────────
def get_zodiac(dob: str) -> str:
    try:
        d = datetime.strptime(dob, "%Y-%m-%d")
        month, day = d.month, d.day
        signs = [
            (1, 20, "Capricorn"), (2, 19, "Aquarius"), (3, 20, "Pisces"),
            (4, 20, "Aries"),     (5, 21, "Taurus"),   (6, 21, "Gemini"),
            (7, 23, "Cancer"),    (8, 23, "Leo"),       (9, 23, "Virgo"),
            (10, 23, "Libra"),    (11, 22, "Scorpio"),  (12, 22, "Sagittarius"),
            (12, 31, "Capricorn"),
        ]
        for m, d_limit, sign in signs:
            if month < m or (month == m and day <= d_limit):
                return sign
        return "Capricorn"
    except Exception:
        return "Unknown"


ZODIAC_TRAITS = {
    "Aries":       "bold, impulsive, energetic, natural leader",
    "Taurus":      "grounded, patient, loyal, lover of beauty and comfort",
    "Gemini":      "curious, adaptable, communicative, dual-natured",
    "Cancer":      "intuitive, nurturing, deeply emotional, home-loving",
    "Leo":         "confident, creative, generous, craves recognition",
    "Virgo":       "analytical, practical, detail-oriented, service-minded",
    "Libra":       "diplomatic, fair-minded, social, seeks harmony",
    "Scorpio":     "intense, perceptive, passionate, transformative",
    "Sagittarius": "adventurous, philosophical, optimistic, freedom-loving",
    "Capricorn":   "ambitious, disciplined, patient, traditional",
    "Aquarius":    "innovative, humanitarian, independent, unconventional",
    "Pisces":      "empathetic, dreamy, artistic, deeply intuitive",
}


# ── Pydantic models ───────────────────────────────────────────────────────────
class OnboardingData(BaseModel):
    dob:         str
    birth_time:  str
    birth_place: str

class AstroChatMessage(BaseModel):
    session_id: str | None = None
    message:    str

class LanguageUpdate(BaseModel):
    language: str  # 'hindi' or 'english'


# ── Helper: build forecast for a user ────────────────────────────────────────
async def _generate_forecast(user_id: str, language: str, profile: dict, sb) -> str:
    zodiac = profile.get("zodiac_sign", "Unknown")
    name   = (profile.get("display_name") or "").split(" ")[0] or "you"
    traits = ZODIAC_TRAITS.get(zodiac, "thoughtful and sensitive")

    entries_res = sb.table("diary_entries")\
        .select("content, entry_date, mood_label")\
        .eq("user_id", user_id)\
        .eq("is_deleted", False)\
        .order("entry_date", desc=True)\
        .limit(5).execute()

    diary_context = ""
    if entries_res.data:
        diary_context = "\n".join([
            f"- {e['entry_date']} ({e.get('mood_label','neutral')}): {e['content'][:150]}"
            for e in entries_res.data
        ])

    today_fmt = datetime.now().strftime("%A, %d %B %Y")

    if language == "hindi":
        structure = """🌅 **आज की ऊर्जा**
इस राशि के लिए आज के दिन की समग्र ऊर्जा के बारे में 2-3 वाक्य।

💼 **कार्य और महत्वाकांक्षा**
आज के करियर/पढ़ाई की ऊर्जा के बारे में 1-2 वाक्य।

❤️ **दिल और रिश्ते**
आज की भावनात्मक और प्रेम ऊर्जा के बारे में 1-2 वाक्य।

🌿 **आज क्या करें**
राशि और डायरी के मूड के अनुसार 2-3 व्यावहारिक सुझाव।

⚠️ **किससे बचें**
1-2 सच्ची सावधानियाँ।

✨ **मेधा का ब्रह्मांडीय संदेश**
एक व्यक्तिगत वाक्य जो उनकी डायरी की भावनाओं को आज की ब्रह्मांडीय ऊर्जा से जोड़े।

अंत में यह पंक्ति लिखें:
*यह राशिफल ज्योतिष के आधार पर है। सुनिश्चित परिणाम नहीं।*"""
        lang_note = "पूरी राशिफल हिंदी में लिखें। राशि, कुंडली, लग्न, ग्रह, गोचर, भाग्य जैसे हिंदी शब्द उपयोग करें।"
    else:
        structure = """🌅 **Today's Energy**
2-3 sentences about overall energy for this sign.

💼 **Work & Ambition**
1-2 sentences about career/study energy today.

❤️ **Heart & Relationships**
1-2 sentences about emotional and relationship energy.

🌿 **What to do today**
2-3 practical suggestions aligned with their sign and diary mood.

⚠️ **What to be careful about**
1-2 honest cautions.

✨ **Medha's Cosmic Note**
One personal sentence connecting their diary feelings to today's cosmic energy.

End with:
*All insights are based on Vedic astrology. Not guaranteed outcomes.*"""
        lang_note = "Write the entire forecast in English."

    prompt = f"""You are Astro Medha, a warm and wise astrology guide.

Today is {today_fmt}.
User's sun sign (राशि): {zodiac}
Their nature: {traits}
Born in: {profile.get('birth_place', 'India')}

Their recent diary entries:
{diary_context if diary_context else "No recent entries."}

Write a warm, personal daily rasiphal for {name}. {lang_note}

Use this structure:
{structure}

Keep total under 250 words. Write like a wise friend, not a generic horoscope."""

    response = _groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are Astro Medha — warm, wise, personal. Follow language instructions strictly."},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=600,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/astro/language")
async def update_language(body: LanguageUpdate, user_id: str = Depends(verify_token)):
    sb = get_sb()
    if body.language not in ("hindi", "english"):
        raise HTTPException(status_code=400, detail="language must be 'hindi' or 'english'")

    sb.table("profiles").update({"astro_language": body.language})\
        .eq("id", user_id).execute()

    # ── Delete today's cached forecast so it regenerates in new language ──
    today = date.today().isoformat()
    sb.table("astro_forecasts")\
        .delete()\
        .eq("user_id", user_id)\
        .eq("date", today)\
        .execute()

    return {"language": body.language}


@router.post("/astro/onboard")
async def astro_onboard(body: OnboardingData, user_id: str = Depends(verify_token)):
    sb     = get_sb()
    zodiac = get_zodiac(body.dob)

    profile_res = sb.table("profiles")\
        .select("astro_onboarded, astro_edit_count")\
        .eq("id", user_id).single().execute()
    profile = profile_res.data or {}

    already_onboarded = profile.get("astro_onboarded", False)
    edit_count        = profile.get("astro_edit_count", 0)

    if already_onboarded:
        if edit_count >= 2:
            raise HTTPException(
                status_code=403,
                detail="You have used both your allowed birth info edits."
            )
        new_edit_count = edit_count + 1
    else:
        new_edit_count = 0

    sb.table("profiles").update({
        "dob":              body.dob,
        "birth_time":       body.birth_time,
        "birth_place":      body.birth_place,
        "zodiac_sign":      zodiac,
        "astro_onboarded":  True,
        "astro_edit_count": new_edit_count,
    }).eq("id", user_id).execute()

    return {
        "zodiac_sign":      zodiac,
        "astro_edit_count": new_edit_count,
        "message":          "Astro profile saved",
    }


@router.get("/astro/profile")
async def get_astro_profile(user_id: str = Depends(verify_token)):
    sb  = get_sb()
    res = sb.table("profiles")\
        .select("is_premium, astro_onboarded, dob, birth_time, birth_place, zodiac_sign, display_name, astro_edit_count, astro_language")\
        .eq("id", user_id).single().execute()
    return res.data or {}


@router.get("/astro/forecast")
async def get_daily_forecast(user_id: str = Depends(verify_token)):
    sb    = get_sb()
    today = date.today().isoformat()

    # Get profile FIRST — need language before checking cache
    profile_res = sb.table("profiles")\
        .select("zodiac_sign, dob, birth_place, display_name, ai_name, astro_language")\
        .eq("id", user_id).single().execute()
    profile  = profile_res.data or {}
    language = profile.get("astro_language") or "english"  # ← never None

    # Check cache — only use if language matches
    cached = sb.table("astro_forecasts")\
        .select("forecast, language")\
        .eq("user_id", user_id)\
        .eq("date", today)\
        .execute()

    if cached.data:
        cached_lang = cached.data[0].get("language", "english")
        if cached_lang == language:
            return {"forecast": cached.data[0]["forecast"], "date": today, "cached": True}
        # Language changed — delete stale cache and regenerate
        sb.table("astro_forecasts").delete().eq("user_id", user_id).eq("date", today).execute()

    try:
        forecast = await _generate_forecast(user_id, language, profile, sb)

        sb.table("astro_forecasts").upsert({
            "user_id":  user_id,
            "date":     today,
            "forecast": forecast,
            "language": language,   # ← store language with cache
        }).execute()

        return {"forecast": forecast, "date": today, "cached": False}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/astro/chat")
async def astro_chat(body: AstroChatMessage, user_id: str = Depends(verify_token)):
    sb = get_sb()

    # ── Fetch profile INCLUDING astro_language ──
    profile_res = sb.table("profiles")\
        .select("is_premium, zodiac_sign, dob, birth_place, birth_time, display_name, astro_language")\
        .eq("id", user_id).single().execute()
    profile = profile_res.data or {}

    if not profile.get("is_premium"):
        raise HTTPException(status_code=403, detail="Premium required")

    zodiac   = profile.get("zodiac_sign", "Unknown")
    name     = (profile.get("display_name") or "").split(" ")[0] or "you"
    traits   = ZODIAC_TRAITS.get(zodiac, "thoughtful and sensitive")
    language = profile.get("astro_language") or "english"  # ← never None

    # Get or create session
    session_id = body.session_id
    if not session_id:
        session_res = sb.table("astro_sessions").insert({
            "user_id": user_id,
            "title":   f"Astro Chat — {date.today().strftime('%d %b')}",
        }).execute()
        session_id = session_res.data[0]["id"]

    # Conversation history
    history_res = sb.table("astro_messages")\
        .select("role, content")\
        .eq("session_id", session_id)\
        .order("created_at")\
        .limit(20).execute()
    history = history_res.data or []

    # Recent diary entries
    entries_res = sb.table("diary_entries")\
        .select("content, entry_date, mood_label")\
        .eq("user_id", user_id)\
        .eq("is_deleted", False)\
        .order("entry_date", desc=True)\
        .limit(7).execute()

    diary_context = ""
    if entries_res.data:
        diary_context = "\n".join([
            f"- {e['entry_date']} ({e.get('mood_label','neutral')}): {e['content'][:200]}"
            for e in entries_res.data
        ])

    # ── Language instructions ──
    if language == "hindi":
        lang_instruction = """IMPORTANT: तुम्हें पूरा जवाब हिंदी में देना है। कोई भी वाक्य अंग्रेज़ी में मत लिखो।

ज्योतिष के लिए इन हिंदी शब्दों का उपयोग करो:
राशि (zodiac sign), कुंडली (horoscope/birth chart), लग्न (ascendant),
चंद्र राशि (moon sign), ग्रह (planet), गोचर (transit),
भाग्य (destiny/luck), विवाह (marriage), करियर, किस्मत,
दशा/अंतर्दशा (time period), नक्षत्र (constellation)"""
        disclaimer = "_ज्योतिष के अनुसार जानकारी दी गई है। यह सुनिश्चित परिणाम नहीं हैं।_"
    else:
        lang_instruction = "Respond in English only. Do not use Hindi."
        disclaimer = "_Astro Medha's insights are based on astrological patterns. Not guaranteed outcomes._"

    system_prompt = f"""You are Astro Medha — a wise, warm astrology guide who deeply knows {name}.

{name}'s astrology profile:
- Sun sign (राशि): {zodiac}
- Nature: {traits}
- Born in: {profile.get('birth_place', 'India')}
- Birth time: {profile.get('birth_time', 'unknown')}

Their recent diary entries (personal context):
{diary_context if diary_context else "No diary entries yet."}

{lang_instruction}

Your role:
- Answer questions about future, relationships, career, life using astrology as a lens
- Weave in what their diary reveals — make it feel personal
- Be warm, honest, grounded — never make hard date-specific predictions
- For sensitive questions (marriage, death, illness) be compassionate, focus on patterns not certainties
- Keep responses under 150 words unless the question needs more depth

End EVERY response with this exact line on its own:
{disclaimer}"""

    messages = [{"role": "system", "content": system_prompt}]
    for h in history:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": body.message})

    try:
        response = _groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=400,
            temperature=0.7,
        )
        reply = response.choices[0].message.content.strip()

        sb.table("astro_messages").insert([
            {"session_id": session_id, "user_id": user_id, "role": "user",      "content": body.message},
            {"session_id": session_id, "user_id": user_id, "role": "assistant",  "content": reply},
        ]).execute()

        return {"reply": reply, "session_id": session_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/astro/send-daily-notifications")
async def send_daily_notifications():
    sb = get_sb()
    premium_users = sb.table("profiles")\
        .select("id, zodiac_sign, display_name, ai_name")\
        .eq("is_premium", True)\
        .eq("astro_onboarded", True)\
        .execute()

    count = 0
    for user in (premium_users.data or []):
        try:
            count += 1
        except Exception as e:
            print(f"Failed for user {user['id']}: {e}")

    return {"sent": count}