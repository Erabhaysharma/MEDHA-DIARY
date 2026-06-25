"""
memories.py — Generate AI memory cards from diary entries

Called when user has 5+ entries.
Returns colorful emotion-tagged memory cards based on diary patterns.
"""

import json
from fastapi import APIRouter, Depends, HTTPException
from supabase import create_client
from groq import Groq
import os

from app.auth import verify_token
from app.service.embeddings import embed_query
from app.service.vector_store import search_similar

router = APIRouter()
_groq  = Groq(api_key=os.getenv("GROQ_API_KEY"))


def get_supabase():
    return create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_KEY"),
    )


@router.get("/memory-cards")
async def get_memory_cards(user_id: str = Depends(verify_token)):
    """
    Generate colorful emotion-based memory cards from diary entries.
    Returns 4-8 cards, each with an emotion, color, summary, and date range.
    """
    supabase = get_supabase()

    # Get profile
    profile_resp = supabase.table("profiles")\
        .select("display_name, ai_name, total_entries")\
        .eq("id", user_id)\
        .execute()

    profile       = profile_resp.data[0] if profile_resp.data else {}
    total_entries = profile.get("total_entries", 0)

    if total_entries < 5:
        return {"cards": [], "unlocked": False}

    # Get recent diary entries for context
    entries_resp = supabase.table("diary_entries")\
        .select("id, content, entry_date, mood_label, mood_score, title")\
        .eq("user_id", user_id)\
        .eq("is_deleted", False)\
        .order("entry_date", desc=True)\
        .limit(30)\
        .execute()

    entries = entries_resp.data or []
    if not entries:
        return {"cards": [], "unlocked": False}

    # Format entries for the prompt
    entries_text = "\n\n".join([
        f"[{e['entry_date']} — {e.get('mood_label', 'neutral')}]\n{e['content'][:300]}"
        for e in entries
    ])

    user_name = profile.get("display_name", "").split(" ")[0] or "the user"

    prompt = f"""Analyze these diary entries from {user_name} and create 5-7 meaningful memory cards.

Each card should capture a distinct emotional theme, pattern, or memorable moment from the entries.

ENTRIES:
{entries_text}

Return ONLY valid JSON in this exact format:
{{
  "cards": [
    {{
      "id": "card_1",
      "emotion": "joy",
      "emoji": "😊",
      "color": "#6A9E72",
      "gradient_end": "#4A7A52",
      "title": "Short catchy title (max 4 words)",
      "summary": "2-3 sentences capturing this emotional pattern or memory. Make it personal and specific to what was written.",
      "date_range": "June 2024",
      "insight": "One honest insight about this pattern"
    }}
  ]
}}

EMOTION OPTIONS and their colors:
- joy: #6A9E72 → #4A7A52 (green)
- love: #C8A96E → #A88B52 (gold)
- growth: #5A8AAE → #3A6A8E (blue)
- peace: #8A7A9E → #6A5A7E (purple)
- energy: #C87A52 → #A85A32 (orange)
- reflection: #7A9E9E → #5A7E7E (teal)
- strength: #9E7A7A → #7E5A5A (rose)
- hope: #C8B86E → #A8984E (amber)

Rules:
- Make titles SHORT and punchy (max 4 words)
- Make summaries SPECIFIC to what was actually written — not generic
- Each card must capture a genuinely different theme
- Use the EXACT hex colors provided
- Return 5 to 7 cards"""

    try:
        response = _groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing personal diaries and finding emotional patterns. Return only valid JSON."},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=2048,
            temperature=0.5,
        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)
        return {
            "cards":    parsed.get("cards", []),
            "unlocked": True,
        }

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Could not generate memory cards")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/trends")
async def get_trends(user_id: str = Depends(verify_token)):
    """
    Analyze diary entries and return trend scores over time.
    Each trend scored 1-10 per entry using LLM.
    """
    supabase = get_supabase()

    entries_resp = supabase.table("diary_entries")\
        .select("id, content, entry_date, mood_label, mood_score")\
        .eq("user_id", user_id)\
        .eq("is_deleted", False)\
        .order("entry_date", desc=False)\
        .limit(30)\
        .execute()

    entries = entries_resp.data or []

    if len(entries) < 2:
        return {"trends": [], "has_data": False}

    # Format entries for LLM
    entries_text = "\n\n".join([
        f"[{e['entry_date']}]: {e['content'][:400]}"
        for e in entries
    ])

    prompt = f"""Analyze these diary entries and score each date on 5 dimensions.

ENTRIES:
{entries_text}

For each date that has an entry, give scores 1-10 for:
1. mood_happiness: Overall mood, happiness, emotional wellbeing
2. productivity: Tasks completed, work done, goals achieved  
3. health: Sleep quality, exercise, physical wellbeing, energy
4. learning: New skills learned, books read, growth mindset
5. career: Work progress, career moves, professional achievements

Rules:
- Score 1-3: Very low/negative mentions or absence
- Score 4-6: Neutral or moderate
- Score 7-10: Strong positive mentions
- If a dimension is not mentioned at all, score 5 (neutral)
- Be consistent across dates

Return ONLY valid JSON:
{{
  "scores": [
    {{
      "date": "2024-01-15",
      "mood_happiness": 7,
      "productivity": 6,
      "health": 5,
      "learning": 4,
      "career": 6
    }}
  ]
}}"""

    try:
        response = _groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {{"role": "system", "content": "You are a diary analyst. Return only valid JSON."}},
                {{"role": "user",   "content": prompt}},
            ],
            max_tokens=2000,
            temperature=0.2,
        )

        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    raw = part
                    break

        parsed  = json.loads(raw)
        scores  = parsed.get("scores", [])

        # Build 5 trend objects
        def build_trend(key, label, emoji, color, gradient):
            points = [
                {"date": s["date"], "value": s.get(key, 5)}
                for s in scores
            ]
            values = [p["value"] for p in points]
            avg    = round(sum(values) / len(values), 1) if values else 5.0
            trend  = "up" if len(values) > 1 and values[-1] > values[0] else \
                     "down" if len(values) > 1 and values[-1] < values[0] else "flat"
            return {
                "key":      key,
                "label":    label,
                "emoji":    emoji,
                "color":    color,
                "gradient": gradient,
                "points":   points,
                "avg":      avg,
                "trend":    trend,
                "latest":   values[-1] if values else 5,
            }

        trends = [
            build_trend("mood_happiness", "Mood & Happiness", "😊",
                        "#C8A96E", "#A88B52"),
            build_trend("productivity",   "Productivity",     "⚡",
                        "#5A8AAE", "#3A6A8E"),
            build_trend("health",         "Health Score",     "❤️",
                        "#9E7A9E", "#7E5A7E"),
            build_trend("learning",       "Learning Growth",  "📚",
                        "#6A9E72", "#4A7A52"),
            build_trend("career",         "Career Growth",    "💼",
                        "#C87A52", "#A85A32"),
        ]

        return {"trends": trends, "has_data": True}

    except Exception as e:
        print(f"Trend analysis error: {e}")
        return {"trends": [], "has_data": False}