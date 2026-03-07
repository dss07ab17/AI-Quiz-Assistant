"""
RAG pipeline utilities:
  - Text chunking (800 chars, 150 overlap)
  - Gemini embedding generation
  - Cosine similarity retrieval
"""

from dotenv import load_dotenv
load_dotenv()

import os
import math
import asyncio
from typing import List, Dict, Any

from google import genai
from google.genai import types


CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBED_MODEL = "gemini-embedding-001"


def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set.")
    return genai.Client(api_key=api_key)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += size - overlap
    return chunks


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _embed_single(client: genai.Client, text: str) -> List[float]:
    """Embed a single text string using Gemini embedding model (async wrapper)."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
    )
    return result.embeddings[0].values


async def build_vector_store(text: str) -> List[Dict[str, Any]]:
    """
    Chunk text → embed each chunk → return list of {content, embedding}.
    All embeddings generated concurrently.
    """
    chunks = chunk_text(text)
    client = _get_client()

    # Embed all chunks concurrently
    tasks = [_embed_single(client, chunk) for chunk in chunks]
    embeddings = await asyncio.gather(*tasks)

    return [
        {"content": chunk, "embedding": embedding}
        for chunk, embedding in zip(chunks, embeddings)
    ]


async def retrieve_chunks(
    vector_store: List[Dict[str, Any]],
    query: str,
    top_k: int = 5,
) -> str:
    """
    Embed query → cosine similarity against store → return top-k chunks joined.
    """
    client = _get_client()
    loop = asyncio.get_event_loop()

    query_result = await loop.run_in_executor(
        None,
        lambda: client.models.embed_content(
            model=EMBED_MODEL,
            contents=query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
    )
    query_embedding = query_result.embeddings[0].values

    scored = [
        (cosine_similarity(query_embedding, item["embedding"]), item["content"])
        for item in vector_store
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    top_chunks = [content for _, content in scored[:top_k]]
    return "\n\n---\n\n".join(top_chunks)