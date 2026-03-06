from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from experiments.bot_detection.cache import BotDetectionDB
from experiments.bot_detection.checkpoint import write_stage_checkpoint
from experiments.bot_detection.llm_client import llm_classify
from experiments.bot_detection.models import StudyConfig

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = (
    "You are analyzing GitHub PR titles from a single author to detect"
    " spam/bot accounts. Here are the most recent PR titles:\n\n"
    "{titles}\n\n"
    "Rate the likelihood this author is a spam/bot account on a scale of"
    " 0.0 (definitely legitimate) to 1.0 (definitely spam)."
    ' Respond with JSON: {{"score": 0.X, "reasoning": "..."}}'
)


def _parse_llm_score(response: str) -> float:
    """Extract the score from an LLM JSON response.

    Returns 0.5 on parse failure.
    """
    try:
        # Strip markdown code fences if present
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (fences)
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines)
        data = json.loads(text)
        score = float(data["score"])
        return max(0.0, min(1.0, score))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Failed to parse LLM response: %s — %s", exc, response[:200])
        return 0.5


def run_stage6_llm_content(base_dir: Path, config: StudyConfig) -> None:
    """H11: LLM-based spam/bot content analysis on PR titles."""
    features_dir = base_dir / config.paths.get("features", "data/features")
    features_path = features_dir / "author_features.parquet"
    if not features_path.exists():
        logger.error("author_features.parquet not found at %s", features_path)
        return

    df = pd.read_parquet(features_path)
    logger.info("Loaded %d authors from %s", len(df), features_path)

    author_cfg = config.author_analysis
    llm_model = author_cfg.get("llm_model", "gemini/gemini-2.0-flash")
    max_titles = author_cfg.get("llm_max_titles", 50)
    pre_filter_merge_rate = author_cfg.get("llm_pre_filter_merge_rate", 0.5)
    pre_filter_min_repos = author_cfg.get("llm_pre_filter_min_repos", 2)

    # Pre-filter: low merge rate AND sufficient repo diversity
    mask = (df["merge_rate"] < pre_filter_merge_rate) & (
        df["total_repos"] >= pre_filter_min_repos
    )
    filtered_logins = set(df.loc[mask, "login"].tolist())
    logger.info(
        "Pre-filtered %d / %d authors for LLM analysis "
        "(merge_rate < %.2f, total_repos >= %d)",
        len(filtered_logins), len(df), pre_filter_merge_rate, pre_filter_min_repos,
    )

    # Open DB to get titles
    db_path = base_dir / config.paths.get("local_db", "data/bot_detection.duckdb")
    db = BotDetectionDB(db_path)

    scores: dict[str, float] = {}
    classified = 0

    try:
        for login in filtered_logins:
            titles = db.get_author_pr_titles(login, limit=max_titles)
            if not titles:
                scores[login] = 0.0
                continue

            titles_text = "\n".join(f"- {t}" for t in titles)
            prompt = _PROMPT_TEMPLATE.format(titles=titles_text)

            try:
                response = llm_classify(model=llm_model, prompt=prompt)
                scores[login] = _parse_llm_score(response)
                classified += 1
            except Exception:
                logger.exception("LLM call failed for %s", login)
                scores[login] = 0.5
    finally:
        db.close()

    # Non-filtered authors get 0.0
    df["llm_spam_score"] = df["login"].map(scores).fillna(0.0)

    df.to_parquet(features_path, index=False)
    logger.info(
        "Wrote llm_spam_score for %d authors (%d classified via LLM)",
        len(df), classified,
    )

    write_stage_checkpoint(
        features_dir,
        "stage6_llm_content",
        {"authors": len(df), "llm_classified": classified},
        {"model": llm_model, "pre_filter_count": len(filtered_logins)},
    )
