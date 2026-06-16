"""
plan.py — AI-powered daily planning endpoint

Called when user taps "Ask Medha to plan my day".
Reads recent diary entries and generates a structured
daily schedule based on the user's goals and patterns.

Endpoint: POST /api/plan
Auth:      Required
"""

import json
from fastapi import APIRouter, Depends, HTTPException, status
from supabase import create_client, Client
from groq import Groq
import os

from app.auth import verify_token
from app.service.embeddings import embed_query
from app.service.vector_store import search_similar

router  = APIRouter()
_groq   = Groq(api_key=os.getenv("GROQ_API_KEY"))


def get_supabase() -> Client:
    return create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_KEY"),
    )


@router.post("/plan", summary="Generate a daily plan from diary context")
async def generate_plan(user_id: str = Depends(verify_token)):
    """
    Reads the user's diary via RAG and generates a realistic
    daily schedule as structured JSON.

    Returns list of tasks:
    [{ title, time_block, scheduled_time, priority, notes }]
    """
    supabase = get_supabase()

    # ── Get user profile ─────────────────────────────────────────────────────
    profile_resp = supabase.table("profiles")\
        .select("display_name, total_entries")\
        .eq("id", user_id)\
        .execute()

    profile       = profile_resp.data[0] if profile_resp.data else {}
    user_name     = profile.get("display_name") or "friend"
    total_entries = profile.get("total_entries") or 0

    if total_entries < 1:
        return {
            "tasks":   [],
            "message": "Write a few diary entries first so Medha can understand your goals and create a personalized plan.",
        }

    # ── Search diary for goals, habits, and priorities ───────────────────────
    # We embed a planning-focused query to find relevant diary chunks
    planning_queries = [
        "my goals and what I want to achieve",
        "my daily routine and schedule",
        "work study plans and priorities",
        "what I need to do and tasks",
    ]

    all_chunks = []
    for query in planning_queries:
        try:
            embedding = embed_query(query)
            chunks    = search_similar(
                user_id=user_id,
                query_embedding=embedding,
                top_k=4,
            )
            all_chunks.extend(chunks)
        except Exception:
            pass

    # Deduplicate chunks by chunk_text
    seen  = set()
    unique_chunks = []
    for chunk in all_chunks:
        text = chunk.get("chunk_text", "")
        if text and text not in seen:
            seen.add(text)
            unique_chunks.append(chunk)

    # Format diary context
    if unique_chunks:
        context_parts = []
        for i, chunk in enumerate(unique_chunks[:10], 1):
            date = chunk.get("entry_date", "")
            text = chunk.get("chunk_text", "")
            context_parts.append(f"[Entry {i} — {date}]\n{text}")
        diary_context = "\n\n".join(context_parts)
    else:
        diary_context = "No diary entries found yet."

    # ── Build planning prompt ─────────────────────────────────────────────────
    from datetime import datetime
    today = datetime.now().strftime("%A, %B %d, %Y")

    system_prompt = f"""You are a personal productivity planner for {user_name}.
Based on their diary entries, generate a realistic and personalized daily schedule for today ({today}).

RULES:
- Return ONLY valid JSON — no explanation, no markdown, no extra text
- Generate 4 to 8 tasks based on what you know about their goals
- Use specific, actionable task titles (not generic like "be productive")
- Assign each task a time_block: morning, afternoon, evening, night, or anytime
- Assign priority: high, normal, or low
- Add a short note explaining WHY this task matters based on their diary
- Distribute tasks realistically across the day
- If they are a student/GATE aspirant: include study blocks
- If they are a freelancer: include client work and skill building
- If no clear profession: focus on general productivity and wellbeing

JSON format — return EXACTLY this structure:
{{
  "tasks": [
    {{
      "title": "task name here",
      "time_block": "morning",
      "scheduled_time": "09:00",
      "priority": "high",
      "notes": "short reason based on their diary"
    }}
  ],
  "insight": "one sentence about what Medha noticed about their goals"
}}"""

    user_message = f"""Here are {user_name}'s recent diary entries:

{diary_context}

Generate a personalized daily schedule for {user_name} based on their goals, habits, and priorities shown in these entries."""

    # ── Call Groq ─────────────────────────────────────────────────────────────
    try:
        response = _groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=1024,
            temperature=0.4,  # lower temp = more structured, consistent output
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown code blocks if model added them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)

        return {
            "tasks":   parsed.get("tasks", []),
            "insight": parsed.get("insight", ""),
        }

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Medha could not generate a structured plan. Try again.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Planning failed: {str(e)}",
        )