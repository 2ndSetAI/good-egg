from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.bot_detection.checkpoint import write_json, write_stage_checkpoint
from experiments.bot_detection.models import StudyConfig
from experiments.bot_detection.stats import (
    auc_roc_with_ci,
    compute_binary_metrics,
    delong_auc_test,
)

logger = logging.getLogger(__name__)


def _make_binary_outcome(df: pd.DataFrame) -> np.ndarray:
    """Binary label: merged=0, not_merged=1 (positive = non-merge)."""
    return (df["outcome"] != "merged").astype(int).values


# ---------------------------------------------------------------------------
# Baseline score generators
# ---------------------------------------------------------------------------

def _ge_score_baseline(df: pd.DataFrame) -> np.ndarray:
    """GE trust score: invert so higher = more suspicious (positive class).

    Rows with missing ge_score get the median value.
    """
    scores = df["ge_score"].values.astype(float).copy()
    median = np.nanmedian(scores)
    scores[np.isnan(scores)] = median
    # Invert: low GE score -> high suspicion
    return 1.0 - scores


def _account_age_baseline(
    df: pd.DataFrame,
    threshold_days: int = 30,
) -> np.ndarray:
    """Binary: 1 if account age < threshold_days, else 0."""
    ages = df["account_age_days"].values.astype(float)
    result = np.zeros(len(df))
    valid = ~np.isnan(ages)
    result[valid] = (ages[valid] < threshold_days).astype(float)
    return result


def _zero_followers_baseline(df: pd.DataFrame) -> np.ndarray:
    """Binary: 1 if followers == 0, else 0."""
    followers = df["followers"].values.astype(float)
    result = np.zeros(len(df))
    valid = ~np.isnan(followers)
    result[valid] = (followers[valid] == 0).astype(float)
    return result


def _zero_repos_baseline(df: pd.DataFrame) -> np.ndarray:
    """Binary: 1 if public_repos == 0, else 0."""
    repos = df["public_repos"].values.astype(float)
    result = np.zeros(len(df))
    valid = ~np.isnan(repos)
    result[valid] = (repos[valid] == 0).astype(float)
    return result


def _random_baseline(n: int, seed: int = 42) -> np.ndarray:
    """Uniform random scores in [0, 1]."""
    rng = np.random.default_rng(seed)
    return rng.random(n)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _compare_baselines(
    y: np.ndarray,
    baselines: dict[str, np.ndarray],
    reference: str = "ge_score",
) -> dict[str, Any]:
    """Compute metrics for each baseline and DeLong-test against reference."""
    results: dict[str, Any] = {}

    ref_scores = baselines.get(reference)
    if ref_scores is None:
        logger.warning("Reference baseline '%s' not found", reference)
        return results

    for name, scores in baselines.items():
        valid = ~(np.isnan(scores) | np.isnan(ref_scores))
        y_v = y[valid]
        s_v = scores[valid]

        auc_ci = auc_roc_with_ci(y_v, s_v)
        metrics = compute_binary_metrics(y_v, s_v)

        entry: dict[str, Any] = {
            "auc_roc": auc_ci["auc"],
            "auc_roc_ci": [auc_ci["ci_lower"], auc_ci["ci_upper"]],
            "auc_roc_se": auc_ci["se"],
            "auc_pr": metrics["auc_pr"],
            "brier_score": metrics["brier_score"],
            "log_loss": metrics["log_loss"],
        }

        # DeLong against reference (skip self-comparison)
        if name != reference:
            ref_v = ref_scores[valid]
            delong = delong_auc_test(y_v, s_v, ref_v)
            entry["delong_vs_reference"] = delong

        results[name] = entry

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_stage4(base_dir: Path, config: StudyConfig) -> dict[str, Any]:
    """Run baseline comparisons.

    Reads features from Parquet, computes baseline scores, compares via
    DeLong test, and writes results.
    """
    seed = config.analysis.get("random_seed", 42)
    features_dir = base_dir / config.paths.get("features", "data/features")
    results_dir = base_dir / config.paths.get("results", "data/results")

    features_path = features_dir / "features.parquet"
    logger.info("Loading features from %s", features_path)
    df = pd.read_parquet(features_path)
    logger.info("Loaded %d rows", len(df))

    y = _make_binary_outcome(df)

    # Build baselines, skipping those with no data
    baselines: dict[str, np.ndarray] = {
        "random": _random_baseline(len(df), seed=seed),
    }

    if df["ge_score"].notna().any():
        baselines["ge_score"] = _ge_score_baseline(df)
    else:
        logger.warning("Skipping ge_score baseline (all NaN)")

    if df["account_age_days"].notna().any():
        baselines["account_age_lt_30d"] = _account_age_baseline(df, threshold_days=30)
    else:
        logger.warning("Skipping account_age baseline (all NaN)")

    if df["followers"].notna().any():
        baselines["zero_followers"] = _zero_followers_baseline(df)
    else:
        logger.warning("Skipping zero_followers baseline (all NaN)")

    if df["public_repos"].notna().any():
        baselines["zero_repos"] = _zero_repos_baseline(df)
    else:
        logger.warning("Skipping zero_repos baseline (all NaN)")

    reference = "ge_score" if "ge_score" in baselines else "random"
    logger.info("Comparing %d baselines (reference: %s)", len(baselines), reference)
    comparison = _compare_baselines(y, baselines, reference=reference)

    results: dict[str, Any] = {
        "n_samples": len(df),
        "reference_baseline": reference,
        "baselines": comparison,
    }

    # Write output
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "baseline_comparison.json"
    write_json(json_path, results)
    logger.info("Baseline comparison written to %s", json_path)

    # Checkpoint
    write_stage_checkpoint(
        base_dir / "data",
        "stage4",
        row_counts={"features": len(df)},
        details={"baselines": list(baselines.keys())},
    )

    return results
