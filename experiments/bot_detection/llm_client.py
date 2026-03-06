from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

_LLM_CACHE_DIR = Path("experiments/bot_detection/data/llm_cache")


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


def _is_retryable(exc: BaseException) -> bool:
    if hasattr(exc, "status_code"):
        code = exc.status_code  # type: ignore[union-attr]
        if code == 429 or (isinstance(code, int) and code >= 500):
            return True
    return False


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _litellm_call(model: str, prompt: str) -> str:
    """Make a single litellm completion call with retry."""
    import litellm

    resp = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return resp.choices[0].message.content  # type: ignore[union-attr]


def llm_classify(
    model: str,
    prompt: str,
    cache_dir: Path | None = None,
) -> str:
    """Call an LLM via litellm with file-based caching and retry.

    Returns the response text content.
    """
    base = cache_dir or _LLM_CACHE_DIR

    cached = _load_cached(model, prompt, base)
    if cached is not None:
        return cached

    result = _litellm_call(model, prompt)
    _save_cached(model, prompt, result, base)
    return result
