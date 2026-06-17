"""
chat.py — RAG chat endpoint

This is the endpoint the mobile chat screen calls.
It runs the full RAG pipeline and streams Medha's response
back to the mobile app token by token.

Endpoint: POST /api/chat
Auth:      Required — JWT token in Authorization header

Flow:
1. Verify JWT → get user_id
2. Fetch user profile (name, ai_name, total_entries)
3. Embed the user's message
4. Search Pinecone for relevant diary chunks
5. Save user message to Supabase
6. Stream Groq response back as Server-Sent Events
7. Save complete assistant response to Supabase
"""

import json
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from supabase import create_client, Client
import os

from app.auth import verify_token
from app.service.embeddings import embed_query
from app.service.vector_store import search_similar
from app.service.rag import stream_response

router = APIRouter()


# ─── Supabase client ──────────────────────────────────────────────────────────
def get_supabase() -> Client:
    return create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_KEY"),
    )


# ─── Request / Response models ────────────────────────────────────────────────
class Message(BaseModel):
    """A single message in the conversation history."""
    role:    str = Field(..., description="user or assistant")
    content: str = Field(..., description="The message text")


class ChatRequest(BaseModel):
    """
    What the mobile app sends on every chat message.

    session_id: Links messages to a conversation thread in Supabase
    message:    What the user just typed
    history:    Last N messages so Medha remembers the conversation context
                Mobile app manages this — sends it on every request
    """
    session_id: str           = Field(..., description="chat_sessions.id from Supabase")
    message:    str           = Field(..., description="The user's current message")
    history:    list[Message] = Field(default=[], description="Previous messages in this session")


# ─── Endpoint ─────────────────────────────────────────────────────────────────
@router.post(
    "/chat",
    summary="Send a message to Medha and stream her response",
    response_description="Server-Sent Events stream of tokens",
)
async def chat(
    body:    ChatRequest,
    user_id: str = Depends(verify_token),
):
    """
    RAG chat pipeline with streaming response.

    Returns a text/event-stream where each event is one of:
      data: {"token": "Hello"}        ← a word fragment from the LLM
      data: {"token": " Abhay"}       ← another fragment
      data: {"done": true, "sources": ["2024-03-15", "2024-04-02"]}  ← end signal
    """
    supabase = get_supabase()

    # ── Step 1: Validate message ─────────────────────────────────────────────
    if not body.message or not body.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty",
        )

    # ── Step 2: Get user profile ─────────────────────────────────────────────
    # We need display_name and ai_name for the system prompt
    # so Medha knows who she's talking to and what she's called
    try:
       profile_resp = supabase.table("profiles")\
        .select("display_name, ai_name, total_entries")\
        .eq("id", user_id)\
        .execute()

       profile = profile_resp.data[0] if profile_resp.data else {}
       user_name     = profile.get("display_name") or "friend"
       ai_name       = profile.get("ai_name")      or "Medha"
       total_entries = profile.get("total_entries") or 0

    except Exception:
        # If profile fetch fails use safe defaults — don't break the chat
        user_name     = "friend"
        ai_name       = "Medha"
        total_entries = 0

    # ── Step 3: Validate session belongs to this user ────────────────────────
    session_check = supabase.table("chat_sessions")\
    .select("id")\
    .eq("id", body.session_id)\
    .eq("user_id", user_id)\
    .execute()

    if not session_check.data or len(session_check.data) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found or does not belong to this user",
        )

    # ── Step 4: Embed the user's message ────────────────────────────────────
    # Convert the question into a vector so we can search Pinecone
    try:
        query_embedding = embed_query(body.message.strip())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to embed message: {str(e)}",
        )

    # ── Step 5: Search Pinecone for relevant diary chunks ────────────────────
    # This is the retrieval part of RAG
    # We search only within this user's namespace
    try:
        relevant_chunks = search_similar(
            user_id=user_id,
            query_embedding=query_embedding,
            top_k=8,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vector search failed: {str(e)}",
        )

    # Extract source info for citation chips in the mobile UI
    # De-duplicate using sets — same entry might appear in multiple chunks
    source_entry_ids = list({
        c["entry_id"]
        for c in relevant_chunks
        if c.get("entry_id")
    })
    source_dates = list({
        c["entry_date"]
        for c in relevant_chunks
        if c.get("entry_date")
    })

    # ── Step 6: Save user message to Supabase ───────────────────────────────
    try:
        supabase.table("chat_messages").insert({
            "session_id":       body.session_id,
            "user_id":          user_id,
            "role":             "user",
            "content":          body.message.strip(),
            "source_entry_ids": [],
            "source_dates":     [],
        }).execute()
    except Exception:
        # Don't fail the chat if message saving fails
        # The user's experience matters more than perfect logging
        pass

    # ── Step 7: Build conversation history for the LLM ──────────────────────
    conversation_history = [
        {"role": m.role, "content": m.content}
        for m in body.history
        if m.role in ("user", "assistant")  # only valid roles
    ]

    # ── Step 8: Stream the response ──────────────────────────────────────────
    # This is an async generator that yields SSE events
    # Each event is a JSON string the mobile app reads and appends
    async def generate():
        full_response_parts = []

        try:
            async for token in stream_response(
            user_message=body.message.strip(),
            retrieved_chunks=relevant_chunks,
            conversation_history=conversation_history,
            user_name=user_name,
            ai_name=ai_name,
            total_entries=total_entries,
        ):
                full_response_parts.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

    # ── Save complete assistant response to Supabase ──────────────────
        complete_response = "".join(full_response_parts)

        if complete_response:
            try:
                supabase.table("chat_messages").insert({
                "session_id":       body.session_id,
                "user_id":          user_id,
                "role":             "assistant",
                "content":          complete_response,
                "source_entry_ids": source_entry_ids,
                "source_dates":     source_dates,
            }).execute()
            except Exception:
                pass

    # ── Auto-update session title from first user message ─────────────
    try:
        session_data = supabase.table("chat_sessions")\
            .select("title")\
            .eq("id", body.session_id)\
            .execute()

        if session_data.data and session_data.data[0]["title"] == "New conversation":
            auto_title = body.message.strip()[:50]
            supabase.table("chat_sessions")\
                .update({"title": auto_title})\
                .eq("id", body.session_id)\
                .execute()
    except Exception:
        pass  # Don't break chat if title update fails

    # ── Send done signal with source dates ────────────────────────────
    yield f"data: {json.dumps({'done': True, 'sources': source_dates})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            # Prevent any proxy or CDN from buffering the stream
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


# ─── Create a new chat session ────────────────────────────────────────────────
@router.post(
    "/chat/session",
    summary="Create a new chat session",
)
async def create_session(
    user_id: str = Depends(verify_token),
):
    """
    Create a new chat session before sending the first message.

    Mobile app calls this when user opens a new chat.
    Returns the session_id which is then passed to /api/chat.

    Why a separate endpoint?
    Each conversation thread needs an ID so messages can be
    grouped together in the chat history screen.
    """
    supabase = get_supabase()

    result = supabase.table("chat_sessions").insert({
    "user_id": user_id,
    "title":   "New conversation",
}).execute()
    if not result.data or len(result.data) == 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create chat session",
        )

    return {
        "session_id": result.data[0]["id"],
        "created_at": result.data[0]["created_at"],
    }


# ─── Get chat history for a session ─────────────────────────────────────────
@router.get(
    "/chat/session/{session_id}",
    summary="Get all messages in a chat session",
)
async def get_session_messages(
    session_id: str,
    user_id:    str = Depends(verify_token),
):
    """
    Fetch all messages in a chat session.

    Mobile app calls this when user reopens a past conversation.
    Returns messages in chronological order.
    """
    supabase = get_supabase()

    # Verify session belongs to this user
    messages = supabase.table("chat_messages")\
        .select("*")\
        .eq("session_id", session_id)\
        .eq("user_id", user_id)\
        .order("created_at", desc=False)\
        .execute()

    return {
        "session_id": session_id,
        "messages":   messages.data or [],
    }

# ─── Get all chat sessions for sidebar ───────────────────────────────────────
@router.get(
    "/chat/sessions",
    summary="Get all chat sessions for the current user",
)
async def get_all_sessions(
    user_id: str = Depends(verify_token),
):
    """
    Returns all chat sessions grouped for the sidebar.
    Mobile app calls this when user opens the chat sidebar.
    """
    supabase = get_supabase()

    # Get all sessions for this user, newest first
    sessions = supabase.table("chat_sessions")\
        .select("id, title, created_at")\
        .eq("user_id", user_id)\
        .order("created_at", desc=True)\
        .execute()

    if not sessions.data:
        return []

    result = []
    for session in sessions.data:
        # Get the last message for preview
        last_msg = supabase.table("chat_messages")\
            .select("content, role")\
            .eq("session_id", session["id"])\
            .eq("user_id", user_id)\
            .order("created_at", desc=True)\
            .limit(1)\
            .execute()

        last_message = ""
        if last_msg.data:
            last_message = last_msg.data[0]["content"][:60] + "..." \
                if len(last_msg.data[0]["content"]) > 60 \
                else last_msg.data[0]["content"]

        result.append({
            "session_id":   session["id"],
            "title":        session.get("title") or "New conversation",
            "created_at":   session["created_at"],
            "last_message": last_message,
        })

    return result


# ─── Get messages for a session (sidebar load) ────────────────────────────────
@router.get(
    "/chat/session/{session_id}/messages",
    summary="Get messages in a session for sidebar reload",
)
async def get_session_messages_list(
    session_id: str,
    user_id:    str = Depends(verify_token),
):
    """
    Returns messages as a simple list for the chat sidebar.
    When user taps a past session, mobile loads it with this.
    """
    supabase = get_supabase()

    # Verify session belongs to this user first
    session_check = supabase.table("chat_sessions")\
        .select("id")\
        .eq("id", session_id)\
        .eq("user_id", user_id)\
        .execute()

    if not session_check.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    messages = supabase.table("chat_messages")\
        .select("role, content, created_at, source_dates")\
        .eq("session_id", session_id)\
        .eq("user_id", user_id)\
        .order("created_at", desc=False)\
        .execute()

    # Return in format chatService expects
    return [
        {
            "role":    msg["role"],
            "content": msg["content"],
        }
        for msg in (messages.data or [])
    ]