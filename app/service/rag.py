import os
from datetime import datetime
from typing import AsyncGenerator
from groq import Groq


# ─── Initialise Groq client once ─────────────────────────────────────────────
def _init_groq() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in environment variables")
    return Groq(api_key=api_key)

_groq = _init_groq()

# The model we use — LLaMA 3.3 70B is the best free option for personality-driven chat
MODEL = "llama-3.3-70b-versatile"


# ─── System prompt ────────────────────────────────────────────────────────────
def build_system_prompt(
    user_name:     str,
    ai_name:       str,
    total_entries: int,
    today:         str,
) -> str:
    """
    This prompt defines Medha's entire personality and behaviour.
    It's injected at the start of every conversation.

    Why it matters:
    The LLM is just a text predictor — the system prompt is what
    transforms it from a generic chatbot into a personal life companion.
    Every word here shapes how Medha responds.

    Design principles used:
    1. Clear identity — who Medha is, what she knows
    2. Tone instructions — warm, honest, direct, not preachy
    3. What she CAN do — reference past, notice patterns, ask questions
    4. What she NEVER does — no therapy, no making things up, no lecturing
    5. Grounding in reality — today's date, how many entries exist
    """
    return f"""You are {ai_name}, {user_name}'s personal AI life companion.

You have been given access to {user_name}'s private diary entries — their most honest, unfiltered thoughts. You know them more deeply than almost anyone.

Today is {today}. {user_name} has written {total_entries} diary entries so far.

═══ WHO YOU ARE ═══
You are not a therapist. You are not a search engine.
You are Medha, a thoughtful and emotionally intelligent companion.
You know the user through their diary and conversations.
You are a wise, caring friend who happens to have read every word {user_name} has ever written in their diary. 
You remember everything — the people they mention, the patterns in their moods, the mistakes they keep making, the things that bring them joy.
Your goal is to help them reflect, understand themselves, and feel heard.
═══ HOW YOU TALK ═══
- Speak like you've always known — never "you wrote" or "you mentioned" or "back in March"
- Never reveal that you're reading diary entries. You just know.
- Instead of: "In March you wrote you felt used by Rahul"
  Say: "That whole situation with Rahul and the money really got to you"
- Instead of: "Your entry from April shows you forgave him"
  Say: "You forgave him but something shifted — you haven't fully trusted him since"
- Use {user_name}'s name occasionally, naturally
- Be direct and warm — say what you see without tiptoeing
- Ask one real question when it adds something — not just to fill space
- Match their energy — heavy moment gets presence, light moment gets lightness
- Keep it conversational — 3 to 5 sentences usually. Friends don't lecture.
- Sometimes gently challenge: "You've been here before with him — what's different now?"


═══ WHAT YOU CAN DO ═══
- Talk about people in their life as if you know them too ("Rahul has always been complicated for you")
- Notice patterns without explaining where you saw them ("This keeps coming up for you")
- Celebrate growth naturally ("You're handling this so differently than you used to")
- Give perspective from shared history ("Last time you felt this way, you pushed through it")
- Reflect their own wisdom back to them


═══ WHAT YOU NEVER DO ═══
- Never say "you wrote", "your entry", "you mentioned", "back in [date]", "according to"
- Never reference the diary, the dates, or the fact that you're reading anything
- Never diagnose or prescribe anything
- Never make up things you don't know — just say "tell me more about that"
- Never start with "{ai_name}:" or "As {ai_name}..."
- Never be generic — everything you say should feel specific to {user_name}
- Never be a yes-person — be real, even when it's a little uncomfortable


═══ IMPORTANT ═══
The diary entries below are your ONLY source of truth about {user_name}'s life.
If something isn't in those entries, say you don't have that memory yet.
The more {user_name} writes, the better you know them.

═══ THE GOLDEN RULE ═══
If someone reading your response could tell you were summarising diary entries — rewrite it.
You don't summarise. You just know. Speak from that place."""


# ─── Context builder ──────────────────────────────────────────────────────────
def build_diary_context(chunks: list[dict]) -> str:
    """
    Format the retrieved diary chunks into readable context for the LLM.

    Each chunk gets a clear header showing when it was written and how
    the user was feeling. This helps the LLM reference entries accurately.

    Why not just dump the raw text?
    Structure helps the LLM understand the temporal context.
    "Back in [date] you wrote..." is only possible if the model
    can see which text belongs to which date.
    """
    if not chunks:
        return "No relevant diary entries found for this conversation."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        date  = chunk.get("entry_date", "Unknown date")
        mood  = chunk.get("mood_label", "")
        title = chunk.get("title", "")
        text  = chunk.get("chunk_text", "").strip()
        score = chunk.get("score", 0)

        # Build a descriptive header for each chunk
        header = f"── Entry {i} [{date}"
        if mood:  header += f" · feeling {mood}"
        if title: header += f' · "{title}"'
        header += f" · relevance {score:.0%}]"

        parts.append(f"{header}\n{text}")

    return "\n\n".join(parts)


# ─── Streaming response ───────────────────────────────────────────────────────
async def stream_response(
    user_message:         str,
    retrieved_chunks:     list[dict],
    conversation_history: list[dict],
    user_name:            str,
    ai_name:              str,
    total_entries:        int,
) -> AsyncGenerator[str, None]:
    """
    Build the full prompt and stream Groq's response token by token.

    This is an async generator — it yields one token at a time.
    The chat router pipes each token to the mobile app as a
    Server-Sent Event so text appears in real time.

    Args:
        user_message:         What the user just typed
        retrieved_chunks:     Top-k similar diary chunks from Pinecone
        conversation_history: Last N messages for continuity
        user_name:            User's display name from their profile
        ai_name:              Name they gave their AI companion
        total_entries:        Total diary entries written

    Yields:
        str — one token at a time (word fragments from the LLM)
    """
    today   = datetime.now().strftime("%B %d, %Y")
    context = build_diary_context(retrieved_chunks)

    # Assemble the full message array sent to Groq
    # Order matters — system prompt first, then diary context,
    # then conversation history, then the current message
    messages = [
        {
            "role":    "system",
            "content": build_system_prompt(
                user_name=user_name,
                ai_name=ai_name,
                total_entries=total_entries,
                today=today,
            ),
        },
        {
            # Diary context as a second system message
            # Separating it from the personality prompt keeps things clean
            "role":    "system",
            "content": f"RELEVANT DIARY ENTRIES — use these to answer:\n\n{context}",
        },
        # Last 10 messages for conversation continuity
        # We limit to 10 to avoid hitting context window limits
        *conversation_history[-10:],
        {
            "role":    "user",
            "content": user_message,
        },
    ]

    # Call Groq with streaming enabled
    stream = _groq.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=512,   # ~350 words — enough for a thoughtful reply
        temperature=0.75, # slightly creative — feels more human, less robotic
        stream=True,      # stream tokens as they're generated
    )

    # Yield each token as it arrives
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ─── Non-streaming response (for testing) ────────────────────────────────────
def get_response(
    user_message:         str,
    retrieved_chunks:     list[dict],
    conversation_history: list[dict],
    user_name:            str,
    ai_name:              str,
    total_entries:        int,
) -> str:
    """
    Non-streaming version — returns the complete response at once.
    Used for testing and any future batch processing.
    Not used in production chat (we always stream there).
    """
    today   = datetime.now().strftime("%B %d, %Y")
    context = build_diary_context(retrieved_chunks)

    messages = [
        {
            "role":    "system",
            "content": build_system_prompt(
                user_name=user_name,
                ai_name=ai_name,
                total_entries=total_entries,
                today=today,
            ),
        },
        {
            "role":    "system",
            "content": f"RELEVANT DIARY ENTRIES:\n\n{context}",
        },
        *conversation_history[-10:],
        {
            "role":    "user",
            "content": user_message,
        },
    ]

    response = _groq.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=512,
        temperature=0.75,
        stream=False,
    )

    return response.choices[0].message.content