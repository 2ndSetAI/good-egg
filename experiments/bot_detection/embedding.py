from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_EMBEDDING_CACHE_DIR = Path("experiments/bot_detection/data/embeddings")


def _cache_key(text: str, model: str) -> str:
    return hashlib.sha256(f"{model}:{text}".encode()).hexdigest()


def _load_cached_embedding(
    text: str,
    model: str,
    cache_dir: Path | None = None,
) -> np.ndarray | None:
    base = cache_dir or _EMBEDDING_CACHE_DIR
    key = _cache_key(text, model)
    cache_file = base / model / f"{key}.json"
    if cache_file.exists():
        data = json.loads(cache_file.read_text())
        return np.array(data["embedding"], dtype=np.float32)
    return None


def _save_cached_embedding(
    text: str,
    model: str,
    embedding: np.ndarray,
    cache_dir: Path | None = None,
) -> None:
    base = cache_dir or _EMBEDDING_CACHE_DIR
    d = base / model
    d.mkdir(parents=True, exist_ok=True)
    key = _cache_key(text, model)
    cache_file = d / f"{key}.json"
    cache_file.write_text(json.dumps({
        "text": text,
        "model": model,
        "embedding": embedding.tolist(),
    }))


async def embed_texts(
    texts: list[str],
    model: str = "gemini-embedding-001",
    batch_size: int = 50,
    cache_dir: Path | None = None,
) -> list[np.ndarray]:
    """Embed texts using Gemini, with local file cache.

    Falls back to zero vectors when google-generativeai is not installed.
    """
    results: list[np.ndarray | None] = [None] * len(texts)
    uncached_indices: list[int] = []

    # Check cache first
    for i, text in enumerate(texts):
        cached = _load_cached_embedding(text, model, cache_dir)
        if cached is not None:
            results[i] = cached
        else:
            uncached_indices.append(i)

    if not uncached_indices:
        return [r for r in results if r is not None]

    # Try to import and use Gemini
    try:
        import google.generativeai as genai
    except ImportError:
        logger.warning(
            "google-generativeai not installed; returning zero vectors"
        )
        dim = 3072  # gemini-embedding-001 default dimension
        # If we got any cached embeddings, use their dimension
        for r in results:
            if r is not None:
                dim = len(r)
                break
        for i in uncached_indices:
            results[i] = np.zeros(dim, dtype=np.float32)
        return [r for r in results if r is not None]

    # Batch embed uncached texts
    for start in range(0, len(uncached_indices), batch_size):
        batch_idx = uncached_indices[start:start + batch_size]
        batch_texts = [texts[i] for i in batch_idx]
        try:
            response = genai.embed_content(
                model=f"models/{model}",
                content=batch_texts,
            )
            embeddings = response["embedding"]
            for j, idx in enumerate(batch_idx):
                emb = np.array(embeddings[j], dtype=np.float32)
                results[idx] = emb
                _save_cached_embedding(
                    texts[idx], model, emb, cache_dir,
                )
        except Exception:
            logger.exception(
                "Embedding batch failed (start=%d)", start,
            )
            dim = 3072
            for r in results:
                if r is not None:
                    dim = len(r)
                    break
            for idx in batch_idx:
                if results[idx] is None:
                    results[idx] = np.zeros(dim, dtype=np.float32)

    return [r for r in results if r is not None]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
