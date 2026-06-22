"""
astro.py — Astro Medha premium backend
Handles zodiac calculation, daily forecast, and astro chat.
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


# ── Models ────────────────────────────────────────────────────────────────────
class OnboardingData(BaseModel):
    dob:         str   # YYYY-MM-DD
    birth_time:  str   # HH:MM or "unknown"
    birth_place: str


class AstroChatMessage(BaseModel):
    session_id: str | None
    message:    str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/astro/onboard")
async def astro_onboard(body: OnboardingData, user_id: str = Depends(verify_token)):
    sb      = get_sb()
    zodiac  = get_zodiac(body.dob)

    sb.table("profiles").update({
        "dob":             body.dob,
        "birth_time":      body.birth_time,
        "birth_place":     body.birth_place,
        "zodiac_sign":     zodiac,
        "astro_onboarded": True,
    }).eq("id", user_id).execute()

    return {"zodiac_sign": zodiac, "message": "Astro profile saved"}


@router.get("/astro/profile")
async def get_astro_profile(user_id: str = Depends(verify_token)):
    sb  = get_sb()
    res = sb.table("profiles")\
        .select("is_premium, astro_onboarded, dob, birth_time, birth_place, zodiac_sign, display_name")\
        .eq("id", user_id).single().execute()
    return res.data or {}


@router.get("/astro/forecast")
async def get_daily_forecast(user_id: str = Depends(verify_token)):
    sb    = get_sb()
    today = date.today().isoformat()

    # Return cached forecast if already generated today
    cached = sb.table("astro_forecasts")\
        .select("forecast")\
        .eq("user_id", user_id)\
        .eq("date", today)\
        .execute()

    if cached.data:
        return {"forecast": cached.data[0]["forecast"], "date": today, "cached": True}

    # Get user astro profile
    profile_res = sb.table("profiles")\
        .select("zodiac_sign, dob, birth_place, display_name, ai_name")\
        .eq("id", user_id).single().execute()
    profile = profile_res.data or {}

    zodiac = profile.get("zodiac_sign", "Unknown")
    name   = (profile.get("display_name") or "").split(" ")[0] or "you"
    traits = ZODIAC_TRAITS.get(zodiac, "thoughtful and sensitive")

    # Get recent diary entries for emotional context
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

    prompt = f"""You are Astro Medha, a warm and wise astrology guide who blends cosmic wisdom with personal insight.

Today is {today_fmt}.
User's sun sign: {zodiac}
Their nature: {traits}
Born in: {profile.get('birth_place', 'India')}

Their recent diary entries:
{diary_context if diary_context else "No recent entries."}

Write a warm, personal daily rasiphal (horoscope) for {name} today. Structure it exactly like this:

🌅 **Today's Energy**
2-3 sentences about the overall energy of the day for this sign. Make it feel cosmic but grounded.

💼 **Work & Ambition**
1-2 sentences specific to career/study energy today.

❤️ **Heart & Relationships**
1-2 sentences about emotional and relationship energy.

🌿 **What to do today**
2-3 practical suggestions aligned with their sign and diary mood.

⚠️ **What to be careful about**
1-2 honest cautions. Don't sugarcoat.

✨ **Medha's Cosmic Note**
One personal sentence connecting their recent diary feelings to today's cosmic energy. Make this feel like Medha actually read their diary.

Keep the total under 250 words. Write warmly, like a wise friend — not like a generic horoscope website.
At the end add exactly this line on its own:
_All insights are based on Vedic astrology and your personal profile. Medha's answers reflect astrological patterns, not guaranteed outcomes._"""

    try:
        response = _groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are Astro Medha — warm, wise, personal. Never make hard predictions about specific dates or guaranteed events."},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=600,
            temperature=0.7,
        )
        forecast = response.choices[0].message.content.strip()

        # Cache it
        sb.table("astro_forecasts").upsert({
            "user_id":  user_id,
            "date":     today,
            "forecast": forecast,
        }).execute()

        return {"forecast": forecast, "date": today, "cached": False}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/astro/send-daily-notifications")
async def send_daily_notifications():
    """
    Called by Render cron at 6:30 AM IST daily.
    Generates and caches forecast for all premium users.
    """
    sb = get_sb()

    premium_users = sb.table("profiles")\
        .select("id, zodiac_sign, display_name, ai_name")\
        .eq("is_premium", True)\
        .eq("astro_onboarded", True)\
        .execute()

    count = 0
    for user in (premium_users.data or []):
        try:
            # Trigger forecast generation (it caches automatically)
            # In production: call get_daily_forecast logic here directly
            # and then send push notification via Expo push API
            uid  = user["id"]
            name = (user.get("display_name") or "").split(" ")[0] or "friend"

            # Get expo push token from your profiles/devices table
            # push_token = get_push_token(uid)
            # send_push(push_token, f"🌟 {name}, your cosmic forecast is ready!")

            count += 1
        except Exception as e:
            print(f"Failed for user {user['id']}: {e}")

    return {"sent": count}


@router.post("/astro/chat")
async def astro_chat(body: AstroChatMessage, user_id: str = Depends(verify_token)):
    sb = get_sb()

    # Verify premium
    profile_res = sb.table("profiles")\
        .select("is_premium, zodiac_sign, dob, birth_place, birth_time, display_name")\
        .eq("id", user_id).single().execute()
    profile = profile_res.data or {}

    if not profile.get("is_premium"):
        raise HTTPException(status_code=403, detail="Premium required")

    zodiac  = profile.get("zodiac_sign", "Unknown")
    name    = (profile.get("display_name") or "").split(" ")[0] or "you"
    traits  = ZODIAC_TRAITS.get(zodiac, "thoughtful and sensitive")

    # Get or create session
    session_id = body.session_id
    if not session_id:
        session_res = sb.table("astro_sessions").insert({
            "user_id": user_id,
            "title":   f"Astro Chat — {date.today().strftime('%d %b')}",
        }).execute()
        session_id = session_res.data[0]["id"]

    # Get conversation history
    history_res = sb.table("astro_messages")\
        .select("role, content")\
        .eq("session_id", session_id)\
        .order("created_at")\
        .limit(20).execute()
    history = history_res.data or []

    # Get recent diary for context
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

    system_prompt = f"""You are Astro Medha — a wise, warm astrology guide who deeply knows {name}.

{name}'s astrology profile:
- Sun sign: {zodiac}
- Nature: {traits}  
- Born in: {profile.get('birth_place', 'India')}
- Birth time: {profile.get('birth_time', 'unknown')}

Their recent diary entries (for personal context):
{diary_context if diary_context else "No diary entries yet."}

Your role:
- Answer questions about their future, relationships, career, and life using astrology as a lens
- Always weave in what their diary reveals about them — make it feel personal
- Be warm, honest, and grounded — never make hard date-specific predictions
- For sensitive questions (marriage, death, illness) be compassionate and redirect to patterns and possibilities, not certainties
- Keep responses under 150 words unless the question needs more depth

IMPORTANT: End every response with this exact disclaimer on its own line:
_Astro Medha's insights are based on astrological patterns. Not guaranteed outcomes._"""

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

        # Save both messages
        sb.table("astro_messages").insert([
            {"session_id": session_id, "user_id": user_id, "role": "user",      "content": body.message},
            {"session_id": session_id, "user_id": user_id, "role": "assistant",  "content": reply},
        ]).execute()

        return {"reply": reply, "session_id": session_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))