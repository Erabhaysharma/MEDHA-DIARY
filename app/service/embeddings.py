"""
embeddings.py — Text to vector conversion using Google Gemini
"""

import os
import time

from google import genai
from google.genai import types

# ─── Initialise client once at module level ───────────────────────────────────
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not set in environment variables")

client = genai.Client(api_key=api_key)

EMBEDDING_MODEL = "gemini-embedding-001"
DIMENSIONS      = 768  # smaller = faster search, still high quality


def embed_text(text: str) -> list[float]:
    """Embed diary text for STORAGE in Pinecone."""
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text")

    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text.strip(),
        config=types.EmbedContentConfig(
            task_type="retrieval_document",
            output_dimensionality=DIMENSIONS,
        ),
    )
    return result.embeddings[0].values


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed multiple diary chunks for storage."""
    if not texts:
        return []

    embeddings = []
    for i, text in enumerate(texts):
        if not text or not text.strip():
            raise ValueError(f"Chunk {i} is empty")

        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text.strip(),
            config=types.EmbedContentConfig(
                task_type="retrieval_document",
                output_dimensionality=DIMENSIONS,
            ),
        )
        embeddings.append(result.embeddings[0].values)

        if i < len(texts) - 1:
            time.sleep(0.1)

    return embeddings


def embed_query(text: str) -> list[float]:
    """Embed a user question for SEARCHING Pinecone."""
    if not text or not text.strip():
        raise ValueError("Cannot embed empty query")

    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text.strip(),
        config=types.EmbedContentConfig(
            task_type="retrieval_query",
            output_dimensionality=DIMENSIONS,
        ),
    )
    return result.embeddings[0].values