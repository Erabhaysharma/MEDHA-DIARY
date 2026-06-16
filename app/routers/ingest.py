from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from supabase import create_client, Client
import os

from app.auth import verify_token
from app.service.embeddings import embed_texts
from app.service.vector_store import upsert_chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter


router = APIRouter()


# ─── Supabase client (service role) ──────────────────────────────────────────
# We use the service role key here — not the anon key.
# Service role bypasses RLS so the backend can update any row.
# This is safe because the backend already verified the JWT —
# we know exactly which user owns this entry.
def get_supabase() -> Client:
    return create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_KEY"),
    )


# ─── Text splitter ────────────────────────────────────────────────────────────
# Splits long diary entries into smaller chunks for better search accuracy.
#
# chunk_size=600: ~100-150 words per chunk
#   - Small enough to be specific (find the exact Rahul paragraph)
#   - Large enough to have context (not just one sentence)
#
# chunk_overlap=80: last 80 chars of chunk N = first 80 chars of chunk N+1
#   - Prevents losing context at chunk boundaries
#   - "I felt angry because Rahul..." doesn't get split mid-thought
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=80,
    length_function=len,
)


# ─── Request model ────────────────────────────────────────────────────────────
class IngestRequest(BaseModel):
    """
    What the mobile app sends to this endpoint.
    All fields except entry_id and content are optional.
    """
    entry_id:   str            = Field(..., description="Supabase diary_entries.id")
    content:    str            = Field(..., description="Full diary entry text")
    entry_date: str            = Field(..., description="ISO date: 2024-06-01")
    mood_label: str | None     = Field(None, description="amazing/good/neutral/bad/awful")
    title:      str | None     = Field(None, description="Entry title if user added one")


# ─── Response model ───────────────────────────────────────────────────────────
class IngestResponse(BaseModel):
    success:     bool
    entry_id:    str
    chunks_made: int
    message:     str


# ─── Endpoint ─────────────────────────────────────────────────────────────────
@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest a diary entry into the vector store",
)
async def ingest_entry(
    body:    IngestRequest,
    user_id: str = Depends(verify_token),  # verify JWT first
):
    """
    Process a diary entry for AI memory.

    Steps:
    1. Validate the entry belongs to the authenticated user
    2. Split text into chunks
    3. Embed chunks with Gemini
    4. Store vectors in Pinecone (namespaced by user_id)
    5. Save chunk records to Supabase entry_chunks table
    6. Mark the entry as is_indexed = true in Supabase
    """
    supabase = get_supabase()

    # ── Step 1: Verify ownership ─────────────────────────────────────────────
    # Even though JWT is verified, we double-check the entry belongs
    # to this user. Prevents one user from ingesting another's entry
    # by guessing entry IDs.
    entry_check = supabase.table("diary_entries")\
    .select("id, user_id, is_indexed")\
    .eq("id", body.entry_id)\
    .eq("user_id", user_id)\
    .execute()

    if not entry_check.data or len(entry_check.data) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Entry not found or does not belong to this user",
        )

    # ── Step 2: Split into chunks ────────────────────────────────────────────
    if not body.content or not body.content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Entry content is empty — nothing to ingest",
        )

    raw_chunks = splitter.split_text(body.content.strip())

    if not raw_chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not split entry into chunks",
        )

    # ── Step 3: Embed all chunks ─────────────────────────────────────────────
    try:
        embeddings = embed_texts(raw_chunks)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embedding failed: {str(e)}",
        )

    # ── Step 4: Prepare chunk objects ────────────────────────────────────────
    pinecone_chunks  = []
    supabase_chunks  = []

    for i, (text, embedding) in enumerate(zip(raw_chunks, embeddings)):
        vector_id = f"{body.entry_id}_chunk_{i}"

        # For Pinecone
        pinecone_chunks.append({
            "id":        vector_id,
            "text":      text,
            "embedding": embedding,
            "metadata": {
                "entry_date": body.entry_date,
                "mood_label": body.mood_label or "",
                "title":      body.title or "",
            },
        })

        # For Supabase entry_chunks table
        supabase_chunks.append({
            "entry_id":           body.entry_id,
            "user_id":            user_id,
            "chunk_index":        i,
            "chunk_text":         text,
            "pinecone_vector_id": vector_id,
            "token_count":        len(text.split()),
        })

    # ── Step 5: Store in Pinecone ────────────────────────────────────────────
    try:
        upsert_chunks(
            user_id=user_id,
            entry_id=body.entry_id,
            chunks=pinecone_chunks,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vector store failed: {str(e)}",
        )

    # ── Step 6: Save chunks to Supabase ─────────────────────────────────────
    # Delete old chunks first in case this is a re-ingest after editing
    supabase.table("entry_chunks")\
        .delete()\
        .eq("entry_id", body.entry_id)\
        .execute()

    supabase.table("entry_chunks")\
        .insert(supabase_chunks)\
        .execute()

    # ── Step 7: Mark entry as indexed ───────────────────────────────────────
    supabase.table("diary_entries")\
        .update({"is_indexed": True})\
        .eq("id", body.entry_id)\
        .execute()

    return IngestResponse(
        success=True,
        entry_id=body.entry_id,
        chunks_made=len(raw_chunks),
        message=f"Entry indexed successfully into {len(raw_chunks)} chunk(s)",
    )