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