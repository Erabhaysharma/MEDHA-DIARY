"""
people.py — Extract people from diary entries and analyze relationships

Two main jobs:
1. POST /api/extract-people — called after diary entry is saved
   Uses Groq to find people mentioned and their sentiment
   
2. GET /api/people — list all people for the current user
3. GET /api/people/{name} — full relationship analysis for one person
"""

import json
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
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


# ─── Request model ────────────────────────────────────────────────────────────
class ExtractRequest(BaseModel):
    entry_id:   str
    content:    str
    entry_date: str


# ─── POST /api/extract-people ─────────────────────────────────────────────────
@router.post("/extract-people")
async def extract_people(
    body:    ExtractRequest,
    user_id: str = Depends(verify_token),
):
    """
    Extract people mentioned in a diary entry using Groq.
    Called automatically after every diary entry is saved.
    
    For each person found:
    - Creates/updates a row in the people table
    - Creates a row in entry_people with sentiment score
    """
    supabase = get_supabase()

    prompt = f"""Analyze this diary entry and extract all people mentioned.

DIARY ENTRY ({body.entry_date}):
{body.content}

Return ONLY valid JSON — no explanation, no markdown:
{{
  "people": [
    {{
      "name": "exact name as written",
      "relationship": "friend|family|colleague|romantic|acquaintance|other",
      "sentiment": 0.8,
      "context": "one sentence about what was written about this person"
    }}
  ]
}}

Rules:
- Only include real people (not places, brands, or abstract concepts)
- sentiment is a float from -1.0 (very negative) to 1.0 (very positive), 0.0 is neutral
- relationship must be one of: friend, family, colleague, romantic, acquaintance, other
- If no people are mentioned return {{"people": []}}
- Common family terms: Mom, Dad, Bhai, Didi, Mama, Chacha = family
- First names, nicknames, and titles are all valid (Rahul, R, my boss, Sir)"""

    try:
        response = _groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role":    "system",
                    "content": "You extract people from diary entries. Return only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=0.1,
        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)
        people = parsed.get("people", [])

        if not people:
            return {"extracted": 0}

        # Save each person to the database
        for person in people:
            name         = person.get("name", "").strip()
            relationship = person.get("relationship", "other")
            sentiment    = float(person.get("sentiment", 0.0))
            context      = person.get("context", "")

            if not name:
                continue

            # Clamp sentiment to -1 to 1
            sentiment = max(-1.0, min(1.0, sentiment))

            # Check if person already exists for this user
            existing = supabase.table("people")\
                .select("id, mention_count, sentiment_avg")\
                .eq("user_id", user_id)\
                .ilike("name", name)\
                .execute()

            if existing.data:
                # Update existing person
                person_id     = existing.data[0]["id"]
                old_count     = existing.data[0]["mention_count"] or 1
                old_sentiment = existing.data[0]["sentiment_avg"] or 0.0

                # Rolling average sentiment
                new_sentiment = round(
                    (old_sentiment * old_count + sentiment) / (old_count + 1),
                    3,
                )

                supabase.table("people").update({
                    "mention_count":  old_count + 1,
                    "sentiment_avg":  new_sentiment,
                    "last_mentioned": body.entry_date,
                    "relationship":   relationship,
                }).eq("id", person_id).execute()

            else:
                # Create new person
                result = supabase.table("people").insert({
                    "user_id":        user_id,
                    "name":           name,
                    "relationship":   relationship,
                    "sentiment_avg":  sentiment,
                    "mention_count":  1,
                    "last_mentioned": body.entry_date,
                }).execute()

                person_id = result.data[0]["id"] if result.data else None

            if not person_id:
                continue

            # Save entry_people link
            # Check if already linked
            already_linked = supabase.table("entry_people")\
                .select("id")\
                .eq("entry_id",  body.entry_id)\
                .eq("person_id", person_id)\
                .execute()

            if not already_linked.data:
                supabase.table("entry_people").insert({
                    "entry_id":        body.entry_id,
                    "person_id":       person_id,
                    "user_id":         user_id,
                    "sentiment_score": sentiment,
                    "context_snippet": context,
                }).execute()

        # Update people_mentioned array on diary_entries
        supabase.table("diary_entries").update({
            "people_mentioned": [p["name"] for p in people if p.get("name")],
        }).eq("id", body.entry_id).execute()

        return {"extracted": len(people), "people": [p["name"] for p in people]}

    except json.JSONDecodeError:
        return {"extracted": 0, "error": "Could not parse response"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── GET /api/people ─────────────────────────────────────────────────────────
@router.get("/people")
async def get_people(user_id: str = Depends(verify_token)):
    """
    Get all people mentioned in the user's diary.
    Sorted by mention count (most mentioned first).
    """
    supabase = get_supabase()

    result = supabase.table("people")\
        .select("*")\
        .eq("user_id", user_id)\
        .order("mention_count", desc=True)\
        .execute()

    people = result.data or []

    # Add vibe label to each person
    for person in people:
        sentiment = person.get("sentiment_avg") or 0.0
        person["vibe"]        = get_vibe_label(sentiment)
        person["vibe_color"]  = get_vibe_color(sentiment)
        person["vibe_emoji"]  = get_vibe_emoji(sentiment)

    return {"people": people}


# ─── GET /api/people/{person_id} ─────────────────────────────────────────────
@router.get("/people/{person_id}")
async def get_person_analysis(
    person_id: str,
    user_id:   str = Depends(verify_token),
):
    """
    Full relationship analysis for one person.
    Returns:
    - Basic info (name, relationship, mention count)
    - Month-by-month sentiment trend
    - Percentage breakdown (positive/neutral/negative)
    - AI-generated relationship summary
    - Key moments from diary
    """
    supabase = get_supabase()

    # Get person
    person_resp = supabase.table("people")\
        .select("*")\
        .eq("id", person_id)\
        .eq("user_id", user_id)\
        .execute()

    if not person_resp.data:
        raise HTTPException(status_code=404, detail="Person not found")

    person = person_resp.data[0]

    # Get all entry_people records for this person
    entries_resp = supabase.table("entry_people")\
        .select("sentiment_score, context_snippet, entry_id")\
        .eq("person_id", person_id)\
        .eq("user_id",   user_id)\
        .execute()

    entry_links = entries_resp.data or []

    # Get the actual diary entries for dates and content
    entry_ids = [e["entry_id"] for e in entry_links]
    
    diary_entries = []
    if entry_ids:
        diary_resp = supabase.table("diary_entries")\
            .select("id, entry_date, content, mood_label")\
            .in_("id", entry_ids)\
            .order("entry_date", desc=False)\
            .execute()
        diary_entries = diary_resp.data or []

    # Build month-by-month sentiment trend
    monthly = {}
    for link in entry_links:
        # Find the matching diary entry for the date
        matching_entry = next(
            (e for e in diary_entries if e["id"] == link["entry_id"]),
            None,
        )
        if not matching_entry:
            continue

        date      = matching_entry["entry_date"]
        month_key = date[:7]  # "2024-03"
        sentiment = link.get("sentiment_score") or 0.0

        if month_key not in monthly:
            monthly[month_key] = {"total": 0, "count": 0}
        monthly[month_key]["total"] += sentiment
        monthly[month_key]["count"] += 1

    trend = [
        {
            "month":     k,
            "sentiment": round(v["total"] / v["count"], 3),
            "count":     v["count"],
        }
        for k, v in sorted(monthly.items())
    ]

    # Sentiment breakdown percentages
    sentiments     = [e.get("sentiment_score") or 0.0 for e in entry_links]
    total          = len(sentiments)
    positive_count = sum(1 for s in sentiments if s > 0.2)
    negative_count = sum(1 for s in sentiments if s < -0.2)
    neutral_count  = total - positive_count - negative_count

    breakdown = {
        "positive": round((positive_count / total * 100) if total > 0 else 0),
        "neutral":  round((neutral_count  / total * 100) if total > 0 else 0),
        "negative": round((negative_count / total * 100) if total > 0 else 0),
    }

    # Key moments — context snippets
    key_moments = [
        {
            "text":  link.get("context_snippet", ""),
            "sentiment": link.get("sentiment_score", 0),
        }
        for link in entry_links
        if link.get("context_snippet")
    ][:5]

    # Generate AI relationship summary
    person_name = person.get("name", "")
    avg_sentiment = person.get("sentiment_avg", 0)

    # Use diary content to generate summary
    diary_text = "\n".join([
        f"[{e['entry_date']}] {e['content'][:200]}"
        for e in diary_entries[:10]
    ])

    summary_prompt = f"""Based on these diary entries mentioning {person_name},
write a 2-3 sentence relationship summary. Be honest, warm, and specific.
Do not start with the person's name.

{diary_text}

Also provide:
- A "role" label (max 4 words, e.g. "emotional support pillar", "creative spark", "occasional source of stress")
- An overall "vibe" (one of: great energy, good energy, mixed energy, needs attention, difficult)"""

    try:
        summary_resp = _groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role":    "system",
                    "content": "You analyze diary relationships. Be concise and honest. Return only JSON.",
                },
                {
                    "role":    "user",
                    "content": f"{summary_prompt}\n\nReturn JSON: {{\"summary\": \"...\", \"role\": \"...\", \"vibe\": \"...\"}}"
                },
            ],
            max_tokens=256,
            temperature=0.4,
        )

        raw = summary_resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        ai_data = json.loads(raw.strip())

    except Exception:
        ai_data = {
            "summary": f"{person_name} has been a part of your story.",
            "role":    person.get("relationship", "person in your life"),
            "vibe":    get_vibe_label(avg_sentiment),
        }

    return {
        "person":      person,
        "trend":       trend,
        "breakdown":   breakdown,
        "key_moments": key_moments,
        "ai_summary":  ai_data.get("summary", ""),
        "role":        ai_data.get("role", ""),
        "vibe":        ai_data.get("vibe", get_vibe_label(avg_sentiment)),
        "vibe_emoji":  get_vibe_emoji(avg_sentiment),
        "vibe_color":  get_vibe_color(avg_sentiment),
    }


# ─── Helpers ──────────────────────────────────────────────────────────────────
def get_vibe_label(sentiment: float) -> str:
    if sentiment >= 0.5:  return "great energy"
    if sentiment >= 0.2:  return "good energy"
    if sentiment >= -0.1: return "mixed energy"
    if sentiment >= -0.4: return "needs attention"
    return "difficult"

def get_vibe_color(sentiment: float) -> str:
    if sentiment >= 0.5:  return "#6A9E72"
    if sentiment >= 0.2:  return "#C8A96E"
    if sentiment >= -0.1: return "#7A7A76"
    if sentiment >= -0.4: return "#8A7A9E"
    return "#A05252"

def get_vibe_emoji(sentiment: float) -> str:
    if sentiment >= 0.5:  return "🌟"
    if sentiment >= 0.2:  return "😊"
    if sentiment >= -0.1: return "😐"
    if sentiment >= -0.4: return "😔"
    return "💔"