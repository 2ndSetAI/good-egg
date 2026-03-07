from __future__ import annotations

import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.bot_detection.checkpoint import write_stage_checkpoint
from experiments.bot_detection.models import StudyConfig

logger = logging.getLogger(__name__)


def compute_time_series_features(
    timestamps: list[datetime],
    burst_gap_days: int = 7,
) -> dict[str, Any]:
    """Compute H9 temporal features from sorted PR timestamps.

    Returns a dict with keys matching AuthorFeatureRow H9 fields.
    """
    n = len(timestamps)

    # weekend_ratio and hour_entropy work even for single-PR authors
    weekend_ratio: float | None = None
    hour_entropy: float | None = None

    if n >= 1:
        weekdays = [t.weekday() for t in timestamps]
        weekend_ratio = sum(1 for d in weekdays if d >= 5) / n

        hours = [t.hour for t in timestamps]
        counts = np.zeros(24)
        for h in hours:
            counts[h] += 1
        probs = counts / n
        # Shannon entropy: -sum(p * log2(p)) for p > 0
        hour_entropy = float(-np.sum(probs[probs > 0] * np.log2(probs[probs > 0])))

    # Interval-based features require >= 2 PRs
    if n < 2:
        return {
            "inter_pr_cv": None,
            "inter_pr_median_hours": None,
            "max_dormancy_days": None,
            "burst_episode_count": None,
            "dormancy_burst_ratio": None,
            "weekend_ratio": weekend_ratio,
            "hour_entropy": hour_entropy,
            "regularity_score": None,
        }

    # Compute inter-PR intervals in hours
    intervals_hours = []
    for i in range(1, n):
        delta = (timestamps[i] - timestamps[i - 1]).total_seconds() / 3600.0
        intervals_hours.append(delta)

    intervals = np.array(intervals_hours)
    mean_interval = float(np.mean(intervals))
    std_interval = float(np.std(intervals))

    inter_pr_cv = std_interval / mean_interval if mean_interval > 0 else None
    inter_pr_median_hours = float(np.median(intervals))
    max_dormancy_days = float(np.max(intervals)) / 24.0

    # Burst episode count: clusters of PRs within 24h, separated by burst_gap_days+ gaps
    gap_threshold_hours = burst_gap_days * 24.0
    burst_episode_count = 0
    in_burst = False
    for i in range(1, n):
        gap = intervals_hours[i - 1]
        if gap <= 24.0:
            if not in_burst:
                burst_episode_count += 1
                in_burst = True
        elif gap >= gap_threshold_hours:
            in_burst = False
        # gaps between 24h and burst_gap_days: keep current state

    # dormancy_burst_ratio
    dormancy_burst_ratio: float | None = None
    if inter_pr_median_hours > 0:
        dormancy_burst_ratio = max_dormancy_days / (inter_pr_median_hours / 24.0)

    # regularity_score: autocorrelation at lag 1
    regularity_score: float | None = None
    if len(intervals) >= 3:
        try:
            corr = np.corrcoef(intervals[:-1], intervals[1:])
            val = corr[0, 1]
            if not math.isnan(val):
                regularity_score = float(val)
        except Exception:
            pass

    return {
        "inter_pr_cv": inter_pr_cv,
        "inter_pr_median_hours": inter_pr_median_hours,
        "max_dormancy_days": max_dormancy_days,
        "burst_episode_count": burst_episode_count,
        "dormancy_burst_ratio": dormancy_burst_ratio,
        "weekend_ratio": weekend_ratio,
        "hour_entropy": hour_entropy,
        "regularity_score": regularity_score,
    }


def run_stage6_time_series(
    base_dir: Path,
    config: StudyConfig,
    cutoff: datetime | None = None,
    parquet_path: Path | None = None,
) -> None:
    """Compute H9 temporal features for all authors and merge into parquet."""
    from experiments.bot_detection.cache import BotDetectionDB

    features_dir = base_dir / config.paths.get("features", "data/features")
    if parquet_path is not None:
        author_parquet = parquet_path
    else:
        author_parquet = features_dir / "author_features.parquet"

    burst_gap_days = config.author_analysis.get("burst_gap_days", 7)

    # Load timestamps from DB
    db_path = base_dir / config.paths.get("local_db", "data/bot_detection.duckdb")
    logger.info("Loading author PR timestamps from %s", db_path)
    with BotDetectionDB(db_path) as db:
        if cutoff is not None:
            all_timestamps = db.get_all_author_pr_timestamps_before(cutoff)
        else:
            all_timestamps = db.get_all_author_pr_timestamps()

    logger.info("Computing H9 features for %d authors", len(all_timestamps))

    rows: list[dict[str, Any]] = []
    for login, ts_list in all_timestamps.items():
        features = compute_time_series_features(ts_list, burst_gap_days=burst_gap_days)
        features["login"] = login
        rows.append(features)

    ts_df = pd.DataFrame(rows)
    logger.info("Computed time-series features: %d rows", len(ts_df))

    # Merge into existing author_features.parquet
    logger.info("Reading existing features from %s", author_parquet)
    author_df = pd.read_parquet(author_parquet)

    # Drop any existing H9 columns to avoid conflicts
    h9_cols = [
        "inter_pr_cv", "inter_pr_median_hours", "max_dormancy_days",
        "burst_episode_count", "dormancy_burst_ratio", "weekend_ratio",
        "hour_entropy", "regularity_score",
    ]
    author_df = author_df.drop(columns=[c for c in h9_cols if c in author_df.columns])

    merged = author_df.merge(ts_df, on="login", how="left")
    merged.to_parquet(author_parquet, index=False)
    logger.info("Wrote merged features to %s (%d rows)", author_parquet, len(merged))

    # Checkpoint
    write_stage_checkpoint(
        base_dir / "data",
        "stage6_time_series",
        row_counts={"authors": len(merged)},
        details={
            "burst_gap_days": burst_gap_days,
            "h9_columns": h9_cols,
        },
    )
