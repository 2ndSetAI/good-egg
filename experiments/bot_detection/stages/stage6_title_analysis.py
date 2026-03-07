from __future__ import annotations

import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from experiments.bot_detection.cache import BotDetectionDB
from experiments.bot_detection.checkpoint import write_stage_checkpoint
from experiments.bot_detection.models import StudyConfig

logger = logging.getLogger(__name__)

# Regex patterns for generic/template PR titles
_TEMPLATE_PATTERNS = [
    re.compile(r"^fix\s+typo", re.IGNORECASE),
    re.compile(r"^update\s+readme", re.IGNORECASE),
    re.compile(r"^fix\s+grammar", re.IGNORECASE),
    re.compile(r"^remove\s+unused", re.IGNORECASE),
    re.compile(r"^add\s+\w+$", re.IGNORECASE),
    re.compile(r"^fix\s+\w+$", re.IGNORECASE),
    re.compile(r"^update\s+\w+$", re.IGNORECASE),
    re.compile(r"^\w+\s+\w+$"),  # exactly 2-word titles
]


def compute_title_shortness(titles: list[str]) -> float:
    """Median of 1/(1+len(title)). Short generic titles score high."""
    if not titles:
        return 0.0
    scores = [1.0 / (1.0 + len(t)) for t in titles]
    return float(np.median(scores))


def compute_lexical_poverty(titles: list[str]) -> float:
    """1 - (unique_words / total_words) across concatenated titles."""
    words = []
    for t in titles:
        words.extend(t.lower().split())
    if not words:
        return 0.0
    unique = len(set(words))
    total = len(words)
    return 1.0 - (unique / total)


def compute_within_author_homogeneity(titles: list[str]) -> float:
    """Mean pairwise cosine similarity of TF-IDF vectors of an author's titles."""
    if len(titles) < 2:
        return 0.0
    try:
        tfidf = TfidfVectorizer(max_features=500, stop_words="english")
        matrix = tfidf.fit_transform(titles)
    except ValueError:
        return 0.0
    sim_matrix = cosine_similarity(matrix)
    # Mean of upper triangle (excluding diagonal)
    n = sim_matrix.shape[0]
    if n < 2:
        return 0.0
    upper_indices = np.triu_indices(n, k=1)
    return float(np.mean(sim_matrix[upper_indices]))


def compute_template_match_rate(titles: list[str]) -> float:
    """Fraction of titles matching generic template patterns."""
    if not titles:
        return 0.0
    matches = 0
    for title in titles:
        for pattern in _TEMPLATE_PATTERNS:
            if pattern.search(title):
                matches += 1
                break
    return matches / len(titles)


def compute_cross_author_commonality(
    titles: list[str],
    global_title_counts: Counter[str],
    min_count: int = 2,
) -> float:
    """Fraction of an author's titles that appear verbatim from other authors."""
    if not titles:
        return 0.0
    common = sum(1 for t in titles if global_title_counts[t.lower().strip()] >= min_count)
    return common / len(titles)


def compute_title_features(
    all_author_titles: dict[str, list[str]],
) -> pd.DataFrame:
    """Compute per-author title features and aggregate into title_spam_score.

    Returns DataFrame with columns: login, title_shortness, lexical_poverty,
    within_author_homogeneity, template_match_rate, cross_author_commonality,
    title_spam_score.
    """
    # Build global title counter for cross-author commonality
    global_counts: Counter[str] = Counter()
    for titles in all_author_titles.values():
        for t in titles:
            global_counts[t.lower().strip()] += 1

    records: list[dict[str, Any]] = []
    for login, titles in all_author_titles.items():
        records.append({
            "login": login,
            "title_shortness": compute_title_shortness(titles),
            "lexical_poverty": compute_lexical_poverty(titles),
            "within_author_homogeneity": compute_within_author_homogeneity(titles),
            "template_match_rate": compute_template_match_rate(titles),
            "cross_author_commonality": compute_cross_author_commonality(
                titles, global_counts,
            ),
        })

    df = pd.DataFrame(records)
    if df.empty:
        df["title_spam_score"] = pd.Series(dtype=float)
        return df

    # MinMaxScale each feature to [0, 1], then average
    feature_cols = [
        "title_shortness",
        "lexical_poverty",
        "within_author_homogeneity",
        "template_match_rate",
        "cross_author_commonality",
    ]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[feature_cols].values)
    df["title_spam_score"] = scaled.mean(axis=1)

    return df


def run_stage6_title_analysis(
    base_dir: Path,
    config: StudyConfig,
    cutoff: datetime | None = None,
    parquet_path: Path | None = None,
) -> None:
    """H11 alternative: TF-IDF-based title spam scoring for all authors."""
    features_dir = base_dir / config.paths.get("features", "data/features")
    features_path = parquet_path or (features_dir / "author_features.parquet")
    if not features_path.exists():
        logger.error("author_features.parquet not found at %s", features_path)
        return

    df = pd.read_parquet(features_path)
    logger.info("Loaded %d authors from %s", len(df), features_path)

    # Get titles from DB
    db_path = base_dir / config.paths.get("local_db", "data/bot_detection.duckdb")
    db = BotDetectionDB(db_path)
    try:
        if cutoff is not None:
            all_titles = db.get_all_author_titles_before(cutoff, limit_per_author=50)
        else:
            all_titles = db.get_all_author_titles(limit_per_author=50)
    finally:
        db.close()

    logger.info("Got titles for %d authors", len(all_titles))

    # Compute features
    title_df = compute_title_features(all_titles)
    logger.info("Computed title features for %d authors", len(title_df))

    # Merge into author features
    if "title_spam_score" in df.columns:
        df = df.drop(columns=["title_spam_score"])
    # Also drop individual title feature columns if they exist from a previous run
    for col in [
        "title_shortness", "lexical_poverty", "within_author_homogeneity",
        "template_match_rate", "cross_author_commonality",
    ]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = df.merge(
        title_df[["login", "title_spam_score"]],
        on="login",
        how="left",
    )
    df["title_spam_score"] = df["title_spam_score"].fillna(0.0)

    df.to_parquet(features_path, index=False)
    logger.info(
        "Wrote title_spam_score for %d authors (%d with titles) to %s",
        len(df), len(title_df), features_path,
    )

    write_stage_checkpoint(
        base_dir / "data",
        "stage6_title_analysis",
        row_counts={"authors": len(df), "authors_with_titles": len(title_df)},
    )
