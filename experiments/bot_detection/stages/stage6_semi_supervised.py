from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from experiments.bot_detection.checkpoint import write_stage_checkpoint
from experiments.bot_detection.models import StudyConfig

logger = logging.getLogger(__name__)

# Numeric feature columns from H8 + H9 + H10 used as input to semi-supervised models.
FEATURE_COLS = [
    # H8: Author-level aggregates
    "total_prs",
    "total_repos",
    "merge_rate",
    "rejection_rate",
    "pocket_veto_rate",
    "median_additions",
    "median_deletions",
    "median_files_changed",
    "mean_title_length",
    "mean_body_length",
    "empty_body_rate",
    "active_days",
    "career_span_days",
    "prs_per_active_day",
    "repos_per_active_day",
    "account_age_days",
    "followers",
    "public_repos",
    # H9: Time-series
    "inter_pr_cv",
    "inter_pr_median_hours",
    "max_dormancy_days",
    "burst_episode_count",
    "dormancy_burst_ratio",
    "weekend_ratio",
    "hour_entropy",
    "regularity_score",
    # H10: Network
    "hub_score",
    "bipartite_clustering",
    "author_projection_degree",
    "repo_diversity_entropy",
    "connected_component_size",
    "mean_repo_popularity",
    "isolation_score",
]


def _build_feature_matrix(
    df: pd.DataFrame,
    exclude: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build a scaled feature matrix from the dataframe.

    Missing values are filled with column medians before scaling.
    Returns (scaled_matrix, list of columns actually used).
    """
    cols = FEATURE_COLS
    if exclude:
        cols = [c for c in cols if c not in exclude]
    available = [c for c in cols if c in df.columns]
    raw = df[available].copy()

    # Fill NaN with column medians
    for col in available:
        median = raw[col].median()
        if pd.isna(median):
            median = 0.0
        raw[col] = raw[col].fillna(median)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(raw.values)
    return scaled, available


def compute_knn_distances(
    features: np.ndarray,
    seed_mask: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """Compute distance from each row to its nearest seed neighbor.

    Returns array of distances (float). Seed rows get distance 0.
    """
    seed_indices = np.where(seed_mask)[0]
    n_seeds = len(seed_indices)

    if n_seeds == 0:
        return np.full(len(features), np.nan)

    effective_k = min(k, n_seeds)
    seed_features = features[seed_indices]

    nn = NearestNeighbors(n_neighbors=effective_k, metric="euclidean")
    nn.fit(seed_features)

    distances, _ = nn.kneighbors(features)
    # Mean distance to k nearest seeds
    mean_distances = distances.mean(axis=1)

    # Seeds themselves get distance 0
    mean_distances[seed_mask] = 0.0

    return mean_distances


def compute_isolation_forest_scores(
    features: np.ndarray,
    contamination: float = 0.05,
    random_state: int = 42,
) -> np.ndarray:
    """Run Isolation Forest and return anomaly scores.

    Returns decision_function scores (lower = more anomalous).
    """
    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(features)
    return iso.decision_function(features)


def run_stage6_semi_supervised(
    base_dir: Path,
    config: StudyConfig,
    cutoff: datetime | None = None,
    parquet_path: Path | None = None,
    exclude_features: list[str] | None = None,
) -> None:
    """Compute H13 semi-supervised features and merge into parquet."""
    features_dir = base_dir / config.paths.get("features", "data/features")
    author_parquet = parquet_path or (features_dir / "author_features.parquet")

    knn_k = config.author_analysis.get("knn_k", 5)
    contamination = config.author_analysis.get("isolation_forest_contamination", 0.05)
    seed = config.author_analysis.get("random_seed", 42)

    logger.info("Reading author features from %s", author_parquet)
    df = pd.read_parquet(author_parquet)
    logger.info("Loaded %d authors", len(df))

    # Build feature matrix
    feat_matrix, used_cols = _build_feature_matrix(df, exclude=exclude_features)
    logger.info(
        "Feature matrix: %d x %d (columns: %s)",
        feat_matrix.shape[0], feat_matrix.shape[1], len(used_cols),
    )

    # Identify suspended accounts as positive seeds
    seed_mask = (df["account_status"] == "suspended").values
    n_seeds = int(seed_mask.sum())
    logger.info("Found %d suspended-account seeds for k-NN", n_seeds)

    # k-NN distance to seeds
    if n_seeds == 0:
        logger.warning("No suspended accounts found -- skipping k-NN, using NaN")
        df["knn_distance_to_seed"] = np.nan
    else:
        if n_seeds < 5:
            logger.warning(
                "Only %d seeds (fewer than 5) -- k-NN will use k=%d",
                n_seeds,
                min(knn_k, n_seeds),
            )
        knn_distances = compute_knn_distances(feat_matrix, seed_mask, k=knn_k)
        df["knn_distance_to_seed"] = knn_distances

    # Isolation Forest
    logger.info("Running Isolation Forest (contamination=%.3f)", contamination)
    iso_scores = compute_isolation_forest_scores(
        feat_matrix, contamination=contamination, random_state=seed,
    )
    df["isolation_forest_score"] = iso_scores

    # Write back
    df.to_parquet(author_parquet, index=False)
    logger.info("Wrote updated features to %s", author_parquet)

    write_stage_checkpoint(
        base_dir / "data",
        "stage6_semi_supervised",
        row_counts={"authors": len(df)},
        details={
            "n_seeds": n_seeds,
            "knn_k": knn_k,
            "contamination": contamination,
            "feature_cols_used": len(used_cols),
        },
    )
