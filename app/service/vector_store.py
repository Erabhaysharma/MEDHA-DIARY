
import os
from pinecone import Pinecone


# ─── Initialise Pinecone client once at module level ─────────────────────────
# Creating the client is cheap but we still do it once to avoid
# unnecessary object creation on every request.
def _init_pinecone() -> Pinecone:
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY not set in environment variables")
    return Pinecone(api_key=api_key)


def _get_index():
    pc         = _init_pinecone()
    index_name = os.getenv("PINECONE_INDEX_NAME", "medha-diary")
    return pc.Index(index_name)


# Get index once — reused across all function calls
_index = _get_index()


# ─── Store vectors ────────────────────────────────────────────────────────────
def upsert_chunks(
    user_id:  str,
    entry_id: str,
    chunks:   list[dict],
) -> int:
    """
    Store embedded diary chunks in Pinecone.

    Args:
        user_id:  The user who owns this entry — used as namespace
        entry_id: The Supabase entry ID — stored in metadata
        chunks:   List of dicts with keys:
                    id        — unique vector ID (entry_id_chunk_N)
                    text      — the diary text chunk
                    embedding — 768-dim vector from Gemini
                    metadata  — entry_date, mood_label, title

    Returns:
        Number of vectors stored

    Why upsert not insert?
    Upsert = insert if new, update if exists.
    If the user edits a diary entry and re-ingests it,
    upsert replaces the old vectors without creating duplicates.
    """
    if not chunks:
        return 0

    vectors = []
    for chunk in chunks:
        vectors.append({
            "id":     chunk["id"],
            "values": chunk["embedding"],
            "metadata": {
                # Store text in metadata so we can retrieve it during search
                # without a separate Supabase query
                "chunk_text": chunk["text"],
                "entry_id":   entry_id,
                "user_id":    user_id,
                "entry_date": chunk["metadata"].get("entry_date", ""),
                "mood_label": chunk["metadata"].get("mood_label", ""),
                "title":      chunk["metadata"].get("title", ""),
            },
        })

    # namespace=user_id isolates each user's vectors completely
    _index.upsert(vectors=vectors, namespace=user_id)
    return len(vectors)


# ─── Search vectors ───────────────────────────────────────────────────────────
def search_similar(
    user_id:         str,
    query_embedding: list[float],
    top_k:           int = 8,
) -> list[dict]:
    """
    Find the most semantically similar diary chunks for a user's question.

    This is the core of the RAG pipeline.
    When a user asks "when was I last happy?" we embed that question
    and search for diary chunks with similar vectors.

    Args:
        user_id:         Only search this user's namespace
        query_embedding: The embedded user question (768 floats)
        top_k:           How many chunks to retrieve (8 is a good default —
                         enough context without overwhelming the LLM)

    Returns:
        List of relevant chunks with text and metadata, sorted by relevance.
        Most relevant chunk is first.

    Why top_k=8?
    Each chunk is ~150 words. 8 chunks = ~1200 words of context.
    That's enough for the LLM to find patterns without hitting
    context window limits or making the prompt too expensive.
    """
    results = _index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=user_id,      # NEVER search outside this user's namespace
        include_metadata=True,  # we need the chunk_text back
    )

    if not results.matches:
        return []

    return [
        {
            "score":      round(match.score, 4),  # similarity score 0-1
            "chunk_text": match.metadata.get("chunk_text", ""),
            "entry_id":   match.metadata.get("entry_id", ""),
            "entry_date": match.metadata.get("entry_date", ""),
            "mood_label": match.metadata.get("mood_label", ""),
            "title":      match.metadata.get("title", ""),
        }
        for match in results.matches
        if match.metadata.get("chunk_text")  # skip empty chunks
    ]


# ─── Delete vectors ───────────────────────────────────────────────────────────
def delete_entry_vectors(user_id: str, entry_id: str) -> None:
    """
    Remove all vectors for a specific diary entry from Pinecone.

    When is this called?
    Only on HARD delete — not soft delete.
    Soft delete (is_deleted=True) hides the entry from the UI
    but keeps the vectors so Medha still remembers it.
    Hard delete removes everything permanently.

    Args:
        user_id:  Used to target the correct namespace
        entry_id: Used to filter which vectors to delete
    """
    # Delete by metadata filter
    # This removes all chunks belonging to this entry
    _index.delete(
        filter={"entry_id": {"$eq": entry_id}},
        namespace=user_id,
    )


# ─── Get index stats (useful for debugging) ──────────────────────────────────
def get_index_stats() -> dict:
    """
    Returns stats about the Pinecone index.
    Useful for debugging — shows total vector count per namespace.

    Call this from a test or admin endpoint to verify vectors are being stored.
    """
    stats = _index.describe_index_stats()
    return {
        "total_vectors":     stats.total_vector_count,
        "namespaces":        {
            ns: data.vector_count
            for ns, data in (stats.namespaces or {}).items()
        },
        "dimension":         stats.dimension,
    }