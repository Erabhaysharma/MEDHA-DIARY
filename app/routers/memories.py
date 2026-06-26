"""
memories.py — Generate AI memory cards and life trend graphs from diary entries.
"""

import json
import os
from datetime import date

from fastapi        import APIRouter, Depends, HTTPException
from supabase       import create_client
from groq           import Groq

from app.auth import verify_token

router = APIRouter()
_groq  = Groq(api_key=os.getenv("GROQ_API_KEY"))


def get_supabase():
    return create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_KEY"),
    )


# ── Shared JSON cleaner ───────────────────────────────────────────────────────
def extract_json(raw: str) -> str:
    """Strip markdown fences and return clean JSON string."""
    raw = raw.strip()
    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                return part
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY CARDS
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/memory-cards")
async def get_memory_cards(user_id: str = Depends(verify_token)):
    """
    Generate colorful emotion-based memory cards from diary entries.
    Cached once per day on the profiles table — instant on repeat loads.
    """
    supabase = get_supabase()

    # ── Fetch profile (includes cache fields) ──
    profile_resp = supabase.table("profiles") \
        .select("display_name, ai_name, total_entries, memory_cards_cache, memory_cards_date") \
        .eq("id", user_id) \
        .execute()

    profile       = profile_resp.data[0] if profile_resp.data else {}
    total_entries = profile.get("total_entries", 0)

    if total_entries < 5:
        return {"cards": [], "unlocked": False}

    # ── Return cache if generated today ──
    cached_date  = profile.get("memory_cards_date")
    cached_cards = profile.get("memory_cards_cache")

    if cached_date == date.today().isoformat() and cached_cards:
        print(f"Memory cards: serving from cache for user {user_id}")
        return cached_cards

    # ── Fetch diary entries ──
    entries_resp = supabase.table("diary_entries") \
        .select("id, content, entry_date, mood_label, mood_score, title") \
        .eq("user_id", user_id) \
        .eq("is_deleted", False) \
        .order("entry_date", desc=True) \
        .limit(30) \
        .execute()

    entries = entries_resp.data or []
    if not entries:
        return {"cards": [], "unlocked": False}

    entries_text = "\n\n".join([
        f"[{e['entry_date']} — {e.get('mood_label', 'neutral')}]\n{e['content'][:300]}"
        for e in entries
    ])

    user_name = (profile.get("display_name") or "").split(" ")[0] or "the user"

    prompt = f"""Analyze these diary entries from {user_name} and create 5-7 meaningful memory cards.

Each card should capture a distinct emotional theme, pattern, or memorable moment from the entries.

ENTRIES:
{entries_text}

Return ONLY valid JSON in this exact format — no extra text, no markdown:
{{
  "cards": [
    {{
      "id": "card_1",
      "emotion": "joy",
      "emoji": "😊",
      "color": "#6A9E72",
      "gradient_end": "#4A7A52",
      "title": "Short punchy title (max 4 words)",
      "summary": "2-3 sentences capturing this emotional pattern. Make it personal and specific.",
      "date_range": "June 2024",
      "insight": "One honest insight about this pattern"
    }}
  ]
}}

EMOTION OPTIONS and their exact colors — use ONLY these hex values:
- joy:        color #6A9E72,  gradient_end #4A7A52  (green)
- love:       color #C8A96E,  gradient_end #A88B52  (gold)
- growth:     color #5A8AAE,  gradient_end #3A6A8E  (blue)
- peace:      color #8A7A9E,  gradient_end #6A5A7E  (purple)
- energy:     color #C87A52,  gradient_end #A85A32  (orange)
- reflection: color #7A9E9E,  gradient_end #5A7E7E  (teal)
- strength:   color #9E7A7A,  gradient_end #7E5A5A  (rose)
- hope:       color #C8B86E,  gradient_end #A8984E  (amber)

Rules:
- Titles max 4 words — punchy and emotional
- Summaries specific to what was actually written — never generic
- Each card must be a genuinely different emotional theme
- Return between 5 and 7 cards"""

    try:
        response = _groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role":    "system",
                    "content": "You are an expert at analyzing personal diaries and finding emotional patterns. Return ONLY valid JSON — no markdown, no explanation.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,
            temperature=0.5,
        )

        raw    = extract_json(response.choices[0].message.content)
        parsed = json.loads(raw)

        # Filter out any cards missing required fields
        valid_cards = [
            c for c in parsed.get("cards", [])
            if c.get("title", "").strip()
            and c.get("summary", "").strip()
            and c.get("emoji")
            and c.get("color")
        ]

        result = {"cards": valid_cards, "unlocked": True}

        # ── Cache on profile ──
        supabase.table("profiles").update({
            "memory_cards_cache": result,
            "memory_cards_date":  date.today().isoformat(),
        }).eq("id", user_id).execute()

        print(f"Memory cards: generated {len(valid_cards)} cards for user {user_id}")
        return result

    except json.JSONDecodeError as e:
        print(f"Memory cards JSON error: {e}")
        raise HTTPException(status_code=500, detail="Could not parse memory cards response")
    except Exception as e:
        print(f"Memory cards error: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# LIFE TREND GRAPHS
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/trends")
async def get_trends(user_id: str = Depends(verify_token)):
    """
    Analyze diary entries and return scored life trend data for 5 dimensions.
    Cached once per day in trend_cache table — instant on repeat loads.
    """
    supabase = get_supabase()
    today    = date.today()

    # ── Check trend cache ──
    cached = supabase.table("trend_cache") \
        .select("data") \
        .eq("user_id", user_id) \
        .eq("date", today.isoformat()) \
        .execute()

    if cached.data:
        print(f"Trends: serving from cache for user {user_id}")
        return cached.data[0]["data"]

    # ── Fetch diary entries ──
    entries_resp = supabase.table("diary_entries") \
        .select("id, content, entry_date, mood_label, mood_score") \
        .eq("user_id", user_id) \
        .eq("is_deleted", False) \
        .order("entry_date", desc=False) \
        .limit(30) \
        .execute()

    entries = entries_resp.data or []

    if len(entries) < 2:
        return {"trends": [], "has_data": False}

    entries_text = "\n\n".join([
        f"[{e['entry_date']} | mood: {e.get('mood_label', 'neutral')}]\n{e['content'][:300]}"
        for e in entries
    ])

    prompt = f"""Analyze these diary entries and score each date on 5 life dimensions.

DIARY ENTRIES:
{entries_text}

For EVERY date listed above, give integer scores 1-10 for each dimension:
- mood_happiness : overall emotional wellbeing, happiness, positivity
- productivity   : tasks completed, goals achieved, focused work done
- health         : sleep quality, exercise, physical energy, body care
- learning       : new skills, books read, courses, intellectual growth
- career         : work progress, career moves, professional achievements

Scoring guide:
  1-3  = very negative mentions OR completely absent from this entry
  4-6  = neutral, routine, or moderately mentioned
  7-10 = strong positive mentions, clearly a good day for this dimension

IMPORTANT: Return ONLY this exact JSON. No markdown, no explanation, no extra text.
{{
  "scores": [
    {{
      "date": "YYYY-MM-DD",
      "mood_happiness": 7,
      "productivity":   6,
      "health":         5,
      "learning":       4,
      "career":         6
    }}
  ]
}}

Include one object per diary entry date. Match dates exactly as given above."""

    try:
        response = _groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role":    "system",
                    "content": "You are a diary analyst. Return ONLY valid JSON — no markdown, no explanation, no extra text.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,
            temperature=0.1,
        )

        raw    = extract_json(response.choices[0].message.content)
        parsed = json.loads(raw)
        scores = parsed.get("scores", [])

        if not scores:
            print(f"Trends: LLM returned empty scores for user {user_id}")
            return {"trends": [], "has_data": False}

        def build_trend(key: str, label: str, emoji: str, color: str):
            points = []
            for s in scores:
                raw_val = s.get(key)
                if raw_val is None:
                    continue
                try:
                    val = max(1, min(10, int(float(raw_val))))
                    points.append({"date": s["date"], "value": val})
                except (TypeError, ValueError):
                    continue

            if not points:
                return None

            values    = [p["value"] for p in points]
            avg       = round(sum(values) / len(values), 1)

            # Compare first 3 vs last 3 for trend direction
            n          = min(3, len(values))
            first_avg  = sum(values[:n]) / n
            last_avg   = sum(values[-n:]) / n
            diff       = last_avg - first_avg
            trend      = "up" if diff > 0.5 else "down" if diff < -0.5 else "flat"

            return {
                "key":    key,
                "label":  label,
                "emoji":  emoji,
                "color":  color,
                "points": points,
                "avg":    avg,
                "trend":  trend,
                "latest": values[-1],
            }

        trends_raw = [
            build_trend("mood_happiness", "Mood & Happiness", "😊", "#C8A96E"),
            build_trend("productivity",   "Productivity",     "⚡", "#5A8AAE"),
            build_trend("health",         "Health Score",     "❤️", "#9E7A9E"),
            build_trend("learning",       "Learning Growth",  "📚", "#6A9E72"),
            build_trend("career",         "Career Growth",    "💼", "#C87A52"),
        ]
        trends = [t for t in trends_raw if t is not None]

        result = {"trends": trends, "has_data": len(trends) > 0}

        # ── Cache today's result ──
        supabase.table("trend_cache").upsert({
            "user_id": user_id,
            "date":    today.isoformat(),
            "data":    result,
        }).execute()

        print(f"Trends: generated {len(trends)} trends for user {user_id}")
        return result

    except json.JSONDecodeError as e:
        print(f"Trends JSON error: {e}")
        print(f"Raw response was: {raw[:400]}")
        return {"trends": [], "has_data": False}
    except Exception as e:
        print(f"Trends error: {type(e).__name__}: {e}")
        return {"trends": [], "has_data": False}