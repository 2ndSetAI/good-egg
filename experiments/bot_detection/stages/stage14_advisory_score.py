from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from experiments.bot_detection.checkpoint import write_json
from experiments.bot_detection.models import StudyConfig
from experiments.bot_detection.stats import delong_auc_test, holm_bonferroni

logger = logging.getLogger(__name__)

DEFAULT_CUTOFFS = [
    "2020-01-01", "2021-01-01", "2022-01-01",
    "2022-07-01", "2023-01-01", "2024-01-01",
]

FULL_FEATURES = [
    "mean_title_length", "rejection_rate", "merge_rate",
    "career_span_days", "hour_entropy", "total_prs",
    "bipartite_clustering", "median_files_changed", "median_additions",
    "hub_score", "total_repos", "empty_body_rate", "title_spam_score",
    "weekend_ratio", "isolation_score", "prs_per_active_day",
]

AVAILABLE_FEATURES = [
    "merge_rate", "total_prs", "career_span_days", "mean_title_length",
    "hub_score", "bipartite_clustering", "isolation_score", "total_repos",
    "median_additions", "median_files_changed",
]

CHEAP_FEATURES = [
    "merge_rate", "total_prs", "career_span_days", "mean_title_length",
    "total_repos", "hub_score",
]

LOG_TRANSFORM = {"median_additions", "median_files_changed", "career_span_days", "total_prs"}

MODEL_VARIANTS = ["susp_lr_full", "susp_lr_available", "susp_lr_cheap"]

VARIANT_FEATURES = {
    "susp_lr_full": FULL_FEATURES,
    "susp_lr_available": AVAILABLE_FEATURES,
    "susp_lr_cheap": CHEAP_FEATURES,
}

FPR_THRESHOLDS = [0.01, 0.05, 0.10]
TIER_THRESHOLDS = {"high": 0.01, "elevated": 0.05}


def run_stage14(
    base_dir: Path,
    config: StudyConfig,
    cutoffs: list[str] | None = None,
) -> dict[str, Any]:
    """Run advisory score experiment across temporal cutoffs."""
    if cutoffs is None:
        holdout_config = config.author_analysis.get("temporal_holdout", {})
        cutoffs = holdout_config.get("cutoffs", DEFAULT_CUTOFFS)

    per_cutoff_results: list[dict[str, Any]] = []

    for cutoff_str in cutoffs:
        parquet_path = (
            base_dir / "data" / "temporal_holdout"
            / f"T_{cutoff_str}" / "author_features.parquet"
        )
        if not parquet_path.exists():
            logger.warning("No parquet at %s, skipping cutoff %s", parquet_path, cutoff_str)
            continue

        logger.info("=== Advisory score: cutoff %s ===", cutoff_str)
        result = evaluate_advisory_cutoff(parquet_path)
        result["cutoff"] = cutoff_str
        per_cutoff_results.append(result)

    aggregated = aggregate_across_cutoffs(per_cutoff_results)

    output = {
        "per_cutoff": per_cutoff_results,
        "aggregated": aggregated,
    }

    output_path = base_dir / "data" / "temporal_holdout" / "advisory_score_experiment.json"
    write_json(output_path, output)
    logger.info("Results written to %s", output_path)

    _print_summary(aggregated)
    return output


def evaluate_advisory_cutoff(
    parquet_path: Path,
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Full pipeline for one cutoff: all 3 model variants."""
    df = pd.read_parquet(parquet_path)
    df = df[df["account_status"].isin(["active", "suspended"])].copy()
    y = (df["account_status"] == "suspended").astype(int).values
    n_suspended = int(y.sum())
    n_labeled = len(y)

    logger.info(
        "  %d labeled (%d suspended, %d active)",
        n_labeled, n_suspended, n_labeled - n_suspended,
    )

    use_loo = n_suspended < 30

    if use_loo:
        logger.info("  Using LOO-CV (n_suspended=%d < 30)", n_suspended)
    else:
        logger.info("  Using %d-fold stratified CV", n_folds)

    models_results: dict[str, Any] = {}
    oof_probs: dict[str, np.ndarray] = {}

    for variant in MODEL_VARIANTS:
        features = _prepare_features(df, VARIANT_FEATURES[variant])
        cv_result = _run_cv_advisory(features, y, n_folds, seed, use_loo)
        probs = cv_result.pop("_oof_probs")
        oof_probs[variant] = probs

        metrics = _compute_metrics(y, probs)
        metrics.update(_compute_precision_at_fpr(y, probs, FPR_THRESHOLDS))
        metrics.update(_compute_calibration(y, probs))
        metrics.update(_compute_advisory_tiers(y, probs, TIER_THRESHOLDS))
        models_results[variant] = metrics

    # DeLong tests: available vs full, cheap vs full
    delong_results: dict[str, Any] = {}
    p_values_for_correction: dict[str, float] = {}

    for alt_name in ["susp_lr_available", "susp_lr_cheap"]:
        full_probs = oof_probs["susp_lr_full"]
        alt_probs = oof_probs[alt_name]
        if np.all(np.isfinite(full_probs)) and np.all(np.isfinite(alt_probs)):
            dl = delong_auc_test(y, alt_probs, full_probs)
            test_key = f"{alt_name}_vs_full"
            delong_results[test_key] = dl
            p_values_for_correction[test_key] = dl["p_value"]

    if p_values_for_correction:
        corrected = holm_bonferroni(p_values_for_correction)
        for test_key, correction in corrected.items():
            delong_results[test_key]["adjusted_p"] = correction["adjusted_p"]
            delong_results[test_key]["reject_h0"] = correction["reject"]

    return {
        "n_labeled": n_labeled,
        "n_suspended": n_suspended,
        "cv_strategy": "loo" if use_loo else f"{n_folds}-fold",
        "models": models_results,
        "delong_tests": delong_results,
    }


def _prepare_features(df: pd.DataFrame, feature_list: list[str]) -> np.ndarray:
    """Extract columns, log-transform skewed ones, fill NaN with 0."""
    arrays = []
    for col in feature_list:
        vals = df[col].fillna(0).values.astype(float)
        if col in LOG_TRANSFORM:
            vals = np.log1p(np.abs(vals)) * np.sign(vals)
        arrays.append(vals)
    return np.column_stack(arrays)


def _compute_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """Compute AUC-ROC, AUC-PR, Precision@25, Precision@50 from scores."""
    metrics: dict[str, Any] = {}
    if y.sum() > 0 and (1 - y).sum() > 0 and np.all(np.isfinite(scores)):
        metrics["auc_roc"] = float(roc_auc_score(y, scores))
        metrics["auc_pr"] = float(average_precision_score(y, scores))
        for k in [25, 50]:
            if k <= len(y):
                top_k_idx = np.argsort(scores)[-k:]
                metrics[f"precision_at_{k}"] = float(y[top_k_idx].sum() / k)
    else:
        metrics["auc_roc"] = float("nan")
        metrics["auc_pr"] = float("nan")
    return metrics


def _run_cv_advisory(
    features: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    seed: int,
    use_loo: bool,
) -> dict[str, Any]:
    """CV returning OOF probabilities. Returns dict with _oof_probs key."""
    n = len(y)
    oof_probs = np.full(n, np.nan)

    if use_loo:
        for i in range(n):
            train_mask = np.ones(n, dtype=bool)
            train_mask[i] = False

            scaler = StandardScaler()
            x_train = scaler.fit_transform(features[train_mask])
            x_test = scaler.transform(features[~train_mask])

            model = LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=seed,
            )
            model.fit(x_train, y[train_mask])
            oof_probs[i] = model.predict_proba(x_test)[0, 1]
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(np.zeros(n), y):
            scaler = StandardScaler()
            x_train = scaler.fit_transform(features[train_idx])
            x_test = scaler.transform(features[test_idx])

            model = LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=seed,
            )
            model.fit(x_train, y[train_idx])
            oof_probs[test_idx] = model.predict_proba(x_test)[:, 1]

    return {"_oof_probs": oof_probs}


def _compute_precision_at_fpr(
    y: np.ndarray,
    scores: np.ndarray,
    fpr_thresholds: list[float],
) -> dict[str, Any]:
    """For each FPR threshold, find score cutoff and compute precision."""
    result: dict[str, Any] = {"precision_at_fpr": {}}
    neg_scores = scores[y == 0]

    if len(neg_scores) == 0:
        return result

    for fpr in fpr_thresholds:
        # Score cutoff: (1-fpr) percentile of negative scores
        cutoff = float(np.percentile(neg_scores, 100 * (1 - fpr)))
        predicted_pos = scores >= cutoff
        tp = int((predicted_pos & (y == 1)).sum())
        fp = int((predicted_pos & (y == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        result["precision_at_fpr"][f"fpr_{fpr:.2f}"] = {
            "cutoff": cutoff,
            "precision": float(precision),
            "tp": tp,
            "fp": fp,
            "n_flagged": int(predicted_pos.sum()),
        }

    return result


def _compute_calibration(
    y: np.ndarray,
    scores: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Calibration curve: binned predicted vs actual, Brier score."""
    result: dict[str, Any] = {}

    valid = np.isfinite(scores)
    if not valid.all():
        scores = scores[valid]
        y = y[valid]

    if len(y) == 0:
        return {"brier_score": float("nan"), "calibration_bins": []}

    with contextlib.suppress(ValueError):
        result["brier_score"] = float(brier_score_loss(y, np.clip(scores, 0, 1)))

    bin_edges = np.linspace(0, 1, n_bins + 1)
    calibration_bins: list[dict[str, Any]] = []
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (scores >= low) & (scores < high) if i < n_bins - 1 else (scores >= low)
        n_in_bin = int(mask.sum())
        if n_in_bin > 0:
            calibration_bins.append({
                "bin_low": float(low),
                "bin_high": float(high),
                "n": n_in_bin,
                "mean_predicted": float(scores[mask].mean()),
                "actual_positive_rate": float(y[mask].mean()),
            })

    result["calibration_bins"] = calibration_bins
    return result


def _compute_advisory_tiers(
    y: np.ndarray,
    scores: np.ndarray,
    tier_thresholds: dict[str, float],
) -> dict[str, Any]:
    """Map scores to HIGH/ELEVATED/NORMAL, report precision per tier."""
    n = len(y)
    high_pct = tier_thresholds["high"]
    elevated_pct = tier_thresholds["elevated"]

    high_cutoff = float(np.percentile(scores, 100 * (1 - high_pct)))
    elevated_cutoff = float(np.percentile(scores, 100 * (1 - elevated_pct)))

    high_mask = scores >= high_cutoff
    elevated_mask = (scores >= elevated_cutoff) & ~high_mask
    normal_mask = ~(high_mask | elevated_mask)

    tiers: dict[str, Any] = {}
    for tier_name, mask in [("high", high_mask), ("elevated", elevated_mask),
                            ("normal", normal_mask)]:
        n_tier = int(mask.sum())
        n_suspended = int(y[mask].sum())
        precision = n_suspended / n_tier if n_tier > 0 else 0.0
        tiers[tier_name] = {
            "n_authors": n_tier,
            "n_suspended": n_suspended,
            "precision": float(precision),
            "pct_of_total": float(n_tier / n) if n > 0 else 0.0,
        }

    return {"advisory_tiers": tiers}


def aggregate_across_cutoffs(
    per_cutoff: list[dict[str, Any]],
) -> dict[str, Any]:
    """Mean/std of metrics across cutoffs, tier stability."""
    if not per_cutoff:
        return {}

    per_model: dict[str, dict[str, Any]] = {}

    for variant in MODEL_VARIANTS:
        aucs_roc: list[float] = []
        aucs_pr: list[float] = []
        brier_scores: list[float] = []
        fpr_precisions: dict[str, list[float]] = {}
        tier_precisions: dict[str, list[float]] = {}

        for r in per_cutoff:
            model_result = r.get("models", {}).get(variant, {})

            for val, dest in [
                (model_result.get("auc_roc"), aucs_roc),
                (model_result.get("auc_pr"), aucs_pr),
                (model_result.get("brier_score"), brier_scores),
            ]:
                if val is not None and not np.isnan(val):
                    dest.append(val)

            # Precision at FPR
            for fpr_key, fpr_data in model_result.get("precision_at_fpr", {}).items():
                fpr_precisions.setdefault(fpr_key, []).append(fpr_data["precision"])

            # Tier precisions
            for tier_name, tier_data in model_result.get("advisory_tiers", {}).items():
                tier_precisions.setdefault(tier_name, []).append(tier_data["precision"])

        agg: dict[str, Any] = {}
        for metric_name, values in [
            ("auc_roc", aucs_roc), ("auc_pr", aucs_pr), ("brier_score", brier_scores),
        ]:
            if values:
                arr = np.array(values)
                agg[f"mean_{metric_name}"] = float(np.mean(arr))
                agg[f"std_{metric_name}"] = float(np.std(arr))
                agg[f"n_cutoffs_{metric_name}"] = len(values)

        # Aggregate precision at FPR
        agg_fpr: dict[str, dict[str, float]] = {}
        for fpr_key, vals in fpr_precisions.items():
            arr = np.array(vals)
            agg_fpr[fpr_key] = {
                "mean_precision": float(np.mean(arr)),
                "std_precision": float(np.std(arr)),
            }
        if agg_fpr:
            agg["precision_at_fpr"] = agg_fpr

        # Aggregate tier precisions
        agg_tiers: dict[str, dict[str, float]] = {}
        for tier_name, vals in tier_precisions.items():
            arr = np.array(vals)
            agg_tiers[tier_name] = {
                "mean_precision": float(np.mean(arr)),
                "std_precision": float(np.std(arr)),
            }
        if agg_tiers:
            agg["advisory_tiers"] = agg_tiers

        per_model[variant] = agg

    # Ranking consistency by AUC-ROC
    rankings: list[dict[str, int]] = []
    for r in per_cutoff:
        models = r.get("models", {})
        cutoff_aucs = {
            m: models.get(m, {}).get("auc_roc", float("nan"))
            for m in MODEL_VARIANTS
        }
        sorted_models = sorted(cutoff_aucs.items(), key=lambda x: -x[1])
        ranking = {m: rank + 1 for rank, (m, _) in enumerate(sorted_models)}
        rankings.append(ranking)

    wins: dict[str, int] = {m: 0 for m in MODEL_VARIANTS}
    mean_rank: dict[str, float] = {m: 0.0 for m in MODEL_VARIANTS}
    for ranking in rankings:
        for m, rank in ranking.items():
            if rank == 1:
                wins[m] += 1
            mean_rank[m] += rank
    for m in MODEL_VARIANTS:
        mean_rank[m] /= max(len(rankings), 1)

    return {
        "per_model": per_model,
        "ranking_consistency": {
            "wins": wins,
            "mean_rank": {m: float(v) for m, v in mean_rank.items()},
        },
    }


def _print_summary(aggregated: dict[str, Any]) -> None:
    """Print a summary table to the log."""
    per_model = aggregated.get("per_model", {})
    logger.info("\n=== Advisory Score Experiment Summary ===")
    logger.info(
        "%-20s  %8s  %8s  %8s  %8s  %8s",
        "Model", "Mean ROC", "Std ROC", "Mean PR", "Std PR", "Brier",
    )
    logger.info("-" * 72)
    for m in MODEL_VARIANTS:
        agg = per_model.get(m, {})
        logger.info(
            "%-20s  %8.3f  %8.3f  %8.3f  %8.3f  %8.4f",
            m,
            agg.get("mean_auc_roc", float("nan")),
            agg.get("std_auc_roc", float("nan")),
            agg.get("mean_auc_pr", float("nan")),
            agg.get("std_auc_pr", float("nan")),
            agg.get("mean_brier_score", float("nan")),
        )

    # Precision at FPR
    logger.info("\nPrecision at fixed FPR thresholds:")
    for m in MODEL_VARIANTS:
        agg = per_model.get(m, {})
        fpr_data = agg.get("precision_at_fpr", {})
        parts = []
        for fpr_key in sorted(fpr_data):
            parts.append(f"{fpr_key}={fpr_data[fpr_key]['mean_precision']:.3f}")
        if parts:
            logger.info("  %-20s  %s", m, "  ".join(parts))

    # Advisory tiers
    logger.info("\nAdvisory tier precision (mean across cutoffs):")
    for m in MODEL_VARIANTS:
        agg = per_model.get(m, {})
        tiers = agg.get("advisory_tiers", {})
        parts = []
        for tier_name in ["high", "elevated", "normal"]:
            td = tiers.get(tier_name, {})
            parts.append(f"{tier_name}={td.get('mean_precision', float('nan')):.3f}")
        logger.info("  %-20s  %s", m, "  ".join(parts))

    # Ranking
    ranking = aggregated.get("ranking_consistency", {})
    wins = ranking.get("wins", {})
    mean_r = ranking.get("mean_rank", {})
    if wins:
        logger.info("\nRanking consistency (wins / mean rank):")
        for m in MODEL_VARIANTS:
            logger.info(
                "  %-20s  wins=%d  mean_rank=%.1f",
                m, wins.get(m, 0), mean_r.get(m, 0),
            )
