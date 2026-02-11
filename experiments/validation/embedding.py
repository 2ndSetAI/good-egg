from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Cache directory for embeddings
_EMBEDDING_CACHE_DIR = Path("experiments/validation/data/embeddings")


def _cache_key(text: str, model: str) -> str:
    """Generate a cache key for an embedding."""
    h = hashlib.sha256(f"{model}:{text}".encode()).hexdigest()[:16]
    return h


def _load_cached_embedding(
    text: str, model: str,
) -> np.ndarray | None:
    """Load a cached embedding if available."""
    cache_dir = _EMBEDDING_CACHE_DIR / model
    key = _cache_key(text, model)
    cache_file = cache_dir / f"{key}.json"
    if cache_file.exists():
        data = json.loads(cache_file.read_text())
        return np.array(data["embedding"], dtype=np.float32)
    return None


def _save_cached_embedding(
    text: str, model: str, embedding: np.ndarray,
) -> None:
    """Cache an embedding to disk."""
    cache_dir = _EMBEDDING_CACHE_DIR / model
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(text, model)
    cache_file = cache_dir / f"{key}.json"
    cache_file.write_text(json.dumps({
        "text": text,
        "model": model,
        "embedding": embedding.tolist(),
    }))


async def embed_texts(
    texts: list[str],
    model: str = "text-embedding-004",
    batch_size: int = 50,
) -> list[np.ndarray]:
    """Embed a list of texts using Gemini's embedding API.

    Uses disk caching to avoid re-embedding identical texts.
    Falls back gracefully if google-generativeai is not installed.
    """
    results: list[np.ndarray | None] = [None] * len(texts)
    to_embed: list[tuple[int, str]] = []

    # Check cache first
    for i, text in enumerate(texts):
        cached = _load_cached_embedding(text, model)
        if cached is not None:
            results[i] = cached
        else:
            to_embed.append((i, text))

    if not to_embed:
        return [r for r in results if r is not None]

    try:
        import google.generativeai as genai
    except ImportError:
        logger.warning(
            "google-generativeai not installed; "
            "returning zero vectors for %d texts", len(to_embed),
        )
        dim = 768  # default dimension for text-embedding-004
        for idx, _ in to_embed:
            results[idx] = np.zeros(dim, dtype=np.float32)
        return [r for r in results if r is not None]

    # Batch embed
    for batch_start in range(0, len(to_embed), batch_size):
        batch = to_embed[batch_start:batch_start + batch_size]
        batch_texts = [text for _, text in batch]

        try:
            response = genai.embed_content(
                model=f"models/{model}",
                content=batch_texts,
            )
            embeddings = response["embedding"]

            for (idx, text), emb in zip(batch, embeddings, strict=True):
                emb_array = np.array(emb, dtype=np.float32)
                results[idx] = emb_array
                _save_cached_embedding(text, model, emb_array)

        except Exception:
            logger.exception(
                "Gemini embedding failed for batch starting at %d",
                batch_start,
            )
            dim = 768
            for idx, _ in batch:
                results[idx] = np.zeros(dim, dtype=np.float32)

    return [r for r in results if r is not None]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_centroid(embeddings: list[np.ndarray]) -> np.ndarray:
    """Compute the centroid (mean) of a list of embeddings."""
    if not embeddings:
        return np.zeros(768, dtype=np.float32)
    return np.mean(np.stack(embeddings), axis=0)


def author_repo_similarity(
    author_repo_embeddings: list[np.ndarray],
    target_repo_embedding: np.ndarray,
) -> float:
    """Compute similarity between an author's repo centroid and target repo.

    The author's "profile" is the centroid of embeddings of repos they've
    contributed to. Similarity is cosine between this centroid and the
    target repo embedding.
    """
    if not author_repo_embeddings:
        return 0.0
    centroid = compute_centroid(author_repo_embeddings)
    return cosine_similarity(centroid, target_repo_embedding)
