"""Async LLM client with concurrency control and file-based caching.

Adapted from experiments/bot_detection/llm_client.py for use with
the proximity experiment pipeline. Uses litellm for model-agnostic
API calls (with its built-in retry and timeout) and asyncio for
concurrent scoring.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path("experiments/bot_detection/data/llm_cache")
_MAX_CONCURRENT = 15


def _cache_key(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()


def _load_cached(model: str, prompt: str, cache_dir: Path) -> str | None:
    key = _cache_key(model, prompt)
    cache_file = cache_dir / model.replace("/", "_") / f"{key}.json"
    if cache_file.exists():
        data = json.loads(cache_file.read_text())
        return data.get("response")
    return None


def _save_cached(
    model: str, prompt: str, response: str, cache_dir: Path,
) -> None:
    d = cache_dir / model.replace("/", "_")
    d.mkdir(parents=True, exist_ok=True)
    key = _cache_key(model, prompt)
    cache_file = d / f"{key}.json"
    cache_file.write_text(json.dumps({
        "model": model,
        "prompt": prompt,
        "response": response,
    }))


def _gemini_call(model: str, prompt: str) -> str:
    """Call Gemini via the google.generativeai SDK.

    The model parameter should include the bare model name (e.g.
    "gemini/gemini-3.1-pro-preview"); the "gemini/" prefix is stripped.
    Retries are handled by the caller (score_authors_batch re-enqueues
    on failure); the SDK handles transport-level retries internally.
    """
    import os

    import google.generativeai as genai  # type: ignore[import-untyped]

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    # Strip litellm-style prefix if present
    model_name = model.removeprefix("gemini/")

    gen_model = genai.GenerativeModel(model_name)
    response = gen_model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(temperature=1.0),
        request_options={"timeout": 120},
    )
    return response.text


def parse_llm_score(response: str) -> float | None:
    """Extract a 0.0-1.0 score from LLM JSON response, with fallbacks.

    Returns None if no score can be parsed (caller should drop the author).
    """
    # Try JSON parse first
    try:
        data = json.loads(response)
        if isinstance(data, dict) and "score" in data:
            return max(0.0, min(1.0, float(data["score"])))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Try extracting JSON from markdown code blocks
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            if isinstance(data, dict) and "score" in data:
                return max(0.0, min(1.0, float(data["score"])))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Fallback: regex for score pattern
    score_match = re.search(r'"score"\s*:\s*([\d.]+)', response)
    if score_match:
        try:
            return max(0.0, min(1.0, float(score_match.group(1))))
        except ValueError:
            pass

    # Last resort: look for any float between 0 and 1
    float_match = re.search(r'\b(0\.\d+|1\.0|0\.0)\b', response)
    if float_match:
        try:
            return float(float_match.group(1))
        except ValueError:
            pass

    logger.warning("Could not parse score from response, dropping")
    return None


async def score_authors_batch(
    model: str,
    authors_data: dict[str, dict],
    prompt_builder: Callable[[dict], str],
    cache_dir: Path | None = None,
    max_concurrent: int = _MAX_CONCURRENT,
) -> dict[str, float]:
    """Score authors asynchronously with semaphore-based concurrency.

    Args:
        model: litellm model string (e.g. "gemini/gemini-3.1-pro-preview")
        authors_data: {author: data_dict} for prompt building
        prompt_builder: callable(data_dict) -> prompt string
        cache_dir: file cache directory
        max_concurrent: max parallel API calls

    Returns:
        {author: score} dict (authors with failed calls or unparseable
        responses are omitted)
    """
    base = cache_dir or _DEFAULT_CACHE_DIR
    sem = asyncio.Semaphore(max_concurrent)
    completed = 0
    total = len(authors_data)
    errors = 0

    async def _score_one(author: str, data: dict) -> tuple[str, float] | None:
        nonlocal completed, errors
        async with sem:
            prompt = prompt_builder(data)
            # Check cache (sync file I/O is fast)
            cached = _load_cached(model, prompt, base)
            if cached is not None:
                completed += 1
                if completed % 500 == 0:
                    logger.info(
                        "Progress: %d/%d (%.1f%%)",
                        completed, total, 100 * completed / total,
                    )
                score = parse_llm_score(cached)
                if score is None:
                    errors += 1
                    return None
                return author, score
            # Call LLM via thread (litellm is sync)
            try:
                result = await asyncio.to_thread(_gemini_call, model, prompt)
            except Exception:
                errors += 1
                logger.warning(
                    "LLM call failed for %s after retries, dropping", author,
                )
                completed += 1
                return None
            _save_cached(model, prompt, result, base)
            completed += 1
            if completed % 500 == 0:
                logger.info(
                    "Progress: %d/%d (%.1f%%), errors: %d",
                    completed, total, 100 * completed / total, errors,
                )
            score = parse_llm_score(result)
            if score is None:
                errors += 1
                return None
            return author, score

    tasks = [_score_one(a, d) for a, d in authors_data.items()]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    scores = {}
    for r in results:
        if r is None or isinstance(r, Exception):
            if isinstance(r, Exception):
                logger.warning("Task exception: %s", r)
            continue
        author, score = r
        scores[author] = score

    logger.info(
        "Scoring complete: %d/%d scored, %d dropped",
        len(scores), total, total - len(scores),
    )
    return scores
