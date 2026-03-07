from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import partial_dependence
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
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

MODEL_TYPES = ["linear", "quadratic", "two_feature", "gbt"]

LOW_MERGE_THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


def run_stage10(
    base_dir: Path,
    config: StudyConfig,
    cutoffs: list[str] | None = None,
) -> dict[str, Any]:
    """Run the merge rate non-monotonicity experiment across temporal cutoffs."""
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

        logger.info("=== Merge rate experiment: cutoff %s ===", cutoff_str)
        result = evaluate_single_cutoff(parquet_path)
        result["cutoff"] = cutoff_str
        per_cutoff_results.append(result)

    aggregated = aggregate_across_cutoffs(per_cutoff_results)

    output = {
        "diagnostics": {
            "per_cutoff": [
                {"cutoff": r["cutoff"], **r["diagnostics"]}
                for r in per_cutoff_results
            ],
        },
        "model_comparison": {
            "per_cutoff": [
                {
                    "cutoff": r["cutoff"],
                    "n_labeled": r["n_labeled"],
                    "n_suspended": r["n_suspended"],
                    "models": r["models"],
                    "delong_vs_baseline": r.get("delong_vs_baseline", {}),
                }
                for r in per_cutoff_results
            ],
            "aggregated": aggregated,
        },
    }

    output_path = base_dir / "data" / "temporal_holdout" / "merge_rate_experiment.json"
    write_json(output_path, output)
    logger.info("Results written to %s", output_path)

    _print_summary(aggregated)
    return output


def compute_merge_rate_diagnostics(
    df: pd.DataFrame,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Bin merge_rate into deciles, compute suspension rate per bin, Spearman test."""
    mr = df["merge_rate"].values
    y = (df["account_status"] == "suspended").astype(int).values

    # Use quantile-based bins, but merge_rate is often clumped at 0 and 1
    # so use fixed bins instead for more interpretable output
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(mr, bin_edges[1:-1])  # 0-based bins

    bins: list[dict[str, Any]] = []
    bin_rates = []
    bin_centers = []

    for i in range(n_bins):
        mask = bin_indices == i
        n_total = int(mask.sum())
        n_susp = int(y[mask].sum()) if n_total > 0 else 0
        rate = n_susp / n_total if n_total > 0 else float("nan")
        center = (bin_edges[i] + bin_edges[i + 1]) / 2

        bins.append({
            "bin_lower": float(bin_edges[i]),
            "bin_upper": float(bin_edges[i + 1]),
            "n_total": n_total,
            "n_suspended": n_susp,
            "suspension_rate": float(rate) if not np.isnan(rate) else None,
        })

        if n_total > 0:
            bin_rates.append(rate)
            bin_centers.append(center)

    # Spearman correlation between bin center and suspension rate
    spearman_result: dict[str, Any] = {"rho": None, "p_value": None}
    if len(bin_rates) >= 3:
        rho, p = spearmanr(bin_centers, bin_rates)
        spearman_result = {"rho": float(rho), "p_value": float(p)}

    return {
        "binned_suspension_rates": bins,
        "spearman_monotonicity": spearman_result,
    }


def evaluate_single_cutoff(
    parquet_path: Path,
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Run all 4 models on one cutoff. Returns per-model metrics + diagnostics."""
    df = pd.read_parquet(parquet_path)

    # Filter to labeled authors
    df = df[df["account_status"].isin(["active", "suspended"])].copy()
    y = (df["account_status"] == "suspended").astype(int).values
    n_suspended = int(y.sum())
    n_labeled = len(y)

    logger.info("  %d labeled authors (%d suspended, %d active)", n_labeled, n_suspended,
                n_labeled - n_suspended)

    # Diagnostics
    diagnostics = compute_merge_rate_diagnostics(df)

    # Build feature arrays.
    # account_age_days excluded: 100% NaN for suspended accounts (leaked indicator).
    hub = df["hub_score"].fillna(0).values.astype(float)
    mr = df["merge_rate"].fillna(0).values.astype(float)

    # Determine CV strategy based on sample size
    use_loo = n_suspended < 30
    use_fixed_threshold = n_suspended < 15

    if use_loo:
        logger.info("  Using LOO-CV (n_suspended=%d < 30)", n_suspended)
    elif n_folds > 1:
        logger.info("  Using %d-fold stratified CV", n_folds)

    # Run each model
    models_results: dict[str, Any] = {}
    oof_probs: dict[str, np.ndarray] = {}

    for model_type in MODEL_TYPES:
        result = _run_cv_for_model(
            model_type, hub, mr, y,
            n_folds=n_folds, seed=seed,
            use_loo=use_loo,
            use_fixed_threshold=use_fixed_threshold,
        )
        models_results[model_type] = result
        oof_probs[model_type] = result.pop("_oof_probs")

    # Full-data fit for coefficient inspection
    for model_type in MODEL_TYPES:
        inspection = _fit_and_inspect(model_type, hub, mr, y)
        models_results[model_type].update(inspection)

    # DeLong tests: each non-linear model vs linear baseline
    baseline_probs = oof_probs["linear"]
    delong_results: dict[str, Any] = {}
    p_values_for_correction: dict[str, float] = {}

    for model_type in ["quadratic", "two_feature", "gbt"]:
        alt_probs = oof_probs[model_type]
        # Only run DeLong if we have valid predictions
        if np.all(np.isfinite(baseline_probs)) and np.all(np.isfinite(alt_probs)):
            dl = delong_auc_test(y, alt_probs, baseline_probs)
            delong_results[model_type] = dl
            p_values_for_correction[model_type] = dl["p_value"]

    # Holm-Bonferroni correction
    if p_values_for_correction:
        corrected = holm_bonferroni(p_values_for_correction)
        for model_type, correction in corrected.items():
            delong_results[model_type]["adjusted_p"] = correction["adjusted_p"]
            delong_results[model_type]["reject_h0"] = correction["reject"]

    return {
        "n_labeled": n_labeled,
        "n_suspended": n_suspended,
        "cv_strategy": "loo" if use_loo else f"{n_folds}-fold",
        "diagnostics": diagnostics,
        "models": models_results,
        "delong_vs_baseline": delong_results,
    }


def _build_features(
    model_type: str,
    hub: np.ndarray,
    mr: np.ndarray,
    low_merge_threshold: float = 0.15,
) -> np.ndarray:
    """Build the feature matrix for a given model type."""
    if model_type == "linear":
        return np.column_stack([hub, mr])
    elif model_type == "quadratic":
        return np.column_stack([hub, mr, mr ** 2])
    elif model_type == "two_feature":
        low_flag = (mr < low_merge_threshold).astype(float)
        return np.column_stack([hub, mr, low_flag])
    elif model_type == "gbt":
        return np.column_stack([hub, mr])
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _feature_names(model_type: str) -> list[str]:
    """Return feature names for a given model type."""
    if model_type == "linear":
        return ["hub_score", "merge_rate"]
    elif model_type == "quadratic":
        return ["hub_score", "merge_rate", "merge_rate_sq"]
    elif model_type == "two_feature":
        return ["hub_score", "merge_rate", "low_merge_flag"]
    elif model_type == "gbt":
        return ["hub_score", "merge_rate"]
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _make_model(model_type: str, seed: int = 42) -> Any:
    """Create the sklearn model for a given model type."""
    if model_type == "gbt":
        return GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=seed,
        )
    else:
        return LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=seed,
        )


def _run_cv_for_model(
    model_type: str,
    hub: np.ndarray,
    mr: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
    use_loo: bool = False,
    use_fixed_threshold: bool = False,
) -> dict[str, Any]:
    """Stratified CV for one model. Returns OOF AUCs and per-fold AUCs."""
    n = len(y)
    oof_probs = np.full(n, np.nan)

    # Determine threshold for two_feature model
    threshold = 0.15  # default fixed

    if use_loo:
        # Leave-one-out CV
        for i in range(n):
            train_mask = np.ones(n, dtype=bool)
            train_mask[i] = False

            if model_type == "two_feature" and not use_fixed_threshold:
                threshold = _tune_threshold_inner_cv(
                    hub[train_mask], mr[train_mask],
                    y[train_mask], seed=seed,
                )

            x_train = _build_features(model_type, hub[train_mask], mr[train_mask],
                                      threshold)
            x_test = _build_features(model_type, hub[~train_mask], mr[~train_mask],
                                     threshold)

            scaler = StandardScaler()
            x_train_s = scaler.fit_transform(x_train)
            x_test_s = scaler.transform(x_test)

            model = _make_model(model_type, seed)
            model.fit(x_train_s, y[train_mask])
            oof_probs[i] = model.predict_proba(x_test_s)[0, 1]
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        for train_idx, test_idx in skf.split(np.zeros(n), y):
            if model_type == "two_feature" and not use_fixed_threshold:
                threshold = _tune_threshold_inner_cv(
                    hub[train_idx], mr[train_idx],
                    y[train_idx], seed=seed,
                )

            x_train = _build_features(model_type, hub[train_idx], mr[train_idx],
                                      threshold)
            x_test = _build_features(model_type, hub[test_idx], mr[test_idx],
                                     threshold)

            scaler = StandardScaler()
            x_train_s = scaler.fit_transform(x_train)
            x_test_s = scaler.transform(x_test)

            model = _make_model(model_type, seed)
            model.fit(x_train_s, y[train_idx])
            oof_probs[test_idx] = model.predict_proba(x_test_s)[:, 1]

    # Compute metrics from OOF predictions
    valid = np.isfinite(oof_probs)
    if valid.sum() < len(y):
        logger.warning("  %s: %d/%d OOF predictions are NaN", model_type,
                        (~valid).sum(), len(y))

    metrics: dict[str, Any] = {}
    if valid.all() and y.sum() > 0 and (1 - y).sum() > 0:
        metrics["auc_roc"] = float(roc_auc_score(y, oof_probs))
        metrics["auc_pr"] = float(average_precision_score(y, oof_probs))

        # Precision@k
        for k in [25, 50]:
            if k <= len(y):
                top_k_idx = np.argsort(oof_probs)[-k:]
                prec_at_k = float(y[top_k_idx].sum() / k)
                metrics[f"precision_at_{k}"] = prec_at_k
    else:
        metrics["auc_roc"] = float("nan")
        metrics["auc_pr"] = float("nan")

    if model_type == "two_feature":
        metrics["tuned_threshold"] = threshold

    metrics["_oof_probs"] = oof_probs
    return metrics


def _tune_threshold_inner_cv(
    hub_train: np.ndarray,
    mr_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    n_inner_folds: int = 3,
) -> float:
    """Inner CV to select best low_merge_flag threshold."""
    n = len(y_train)
    n_pos = y_train.sum()

    # Need at least n_inner_folds positives for stratified inner CV
    if n_pos < n_inner_folds:
        return 0.15  # fallback

    best_threshold = 0.15
    best_auc = -1.0

    inner_skf = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=seed + 1)

    for thresh in LOW_MERGE_THRESHOLDS:
        inner_oof = np.full(n, np.nan)

        for tr_idx, val_idx in inner_skf.split(np.zeros(n), y_train):
            x_tr = _build_features("two_feature", hub_train[tr_idx], mr_train[tr_idx],
                                   thresh)
            x_val = _build_features("two_feature", hub_train[val_idx], mr_train[val_idx],
                                    thresh)

            scaler = StandardScaler()
            x_tr_s = scaler.fit_transform(x_tr)
            x_val_s = scaler.transform(x_val)

            model = LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=seed,
            )
            model.fit(x_tr_s, y_train[tr_idx])
            inner_oof[val_idx] = model.predict_proba(x_val_s)[:, 1]

        valid = np.isfinite(inner_oof)
        if valid.all() and y_train.sum() > 0:
            try:
                auc = roc_auc_score(y_train, inner_oof)
                if auc > best_auc:
                    best_auc = auc
                    best_threshold = thresh
            except ValueError:
                pass

    return best_threshold


def _fit_and_inspect(
    model_type: str,
    hub: np.ndarray,
    mr: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit on all data, return coefficients/importances + partial dependence for GBT."""
    features = _build_features(model_type, hub, mr)
    scaler = StandardScaler()
    features_s = scaler.fit_transform(features)
    names = _feature_names(model_type)

    model = _make_model(model_type, seed)
    model.fit(features_s, y)

    result: dict[str, Any] = {}

    if model_type == "gbt":
        importances = dict(zip(names, [float(v) for v in model.feature_importances_], strict=True))
        result["feature_importances"] = importances

        # Partial dependence for merge_rate (index 1)
        mr_idx = names.index("merge_rate")
        pd_result = partial_dependence(model, features_s, features=[mr_idx], kind="average",
                                       grid_resolution=50)
        # Convert grid from scaled back to original merge_rate
        grid_scaled = pd_result["grid_values"][0]
        grid_original = grid_scaled * scaler.scale_[mr_idx] + scaler.mean_[mr_idx]
        result["partial_dependence_merge_rate"] = {
            "grid": [float(v) for v in grid_original],
            "values": [float(v) for v in pd_result["average"][0]],
        }
    else:
        coefficients = dict(zip(names, [float(v) for v in model.coef_[0]], strict=True))
        coefficients["intercept"] = float(model.intercept_[0])
        result["coefficients"] = coefficients

    return result


def aggregate_across_cutoffs(
    per_cutoff: list[dict[str, Any]],
) -> dict[str, Any]:
    """Mean/std/min/max across cutoffs per model. Ranking consistency."""
    if not per_cutoff:
        return {}

    aggregated: dict[str, dict[str, Any]] = {}

    for model_type in MODEL_TYPES:
        aucs_roc: list[float] = []
        aucs_pr: list[float] = []

        for r in per_cutoff:
            model_result = r.get("models", {}).get(model_type, {})
            auc_roc = model_result.get("auc_roc")
            auc_pr = model_result.get("auc_pr")
            if auc_roc is not None and not np.isnan(auc_roc):
                aucs_roc.append(auc_roc)
            if auc_pr is not None and not np.isnan(auc_pr):
                aucs_pr.append(auc_pr)

        agg: dict[str, Any] = {}
        if aucs_roc:
            arr = np.array(aucs_roc)
            agg["mean_auc_roc"] = float(np.mean(arr))
            agg["std_auc_roc"] = float(np.std(arr))
            agg["min_auc_roc"] = float(np.min(arr))
            agg["max_auc_roc"] = float(np.max(arr))
            agg["n_cutoffs_roc"] = len(aucs_roc)
            agg["per_cutoff_auc_roc"] = aucs_roc
        if aucs_pr:
            arr = np.array(aucs_pr)
            agg["mean_auc_pr"] = float(np.mean(arr))
            agg["std_auc_pr"] = float(np.std(arr))
            agg["min_auc_pr"] = float(np.min(arr))
            agg["max_auc_pr"] = float(np.max(arr))
            agg["n_cutoffs_pr"] = len(aucs_pr)
        aggregated[model_type] = agg

    # Ranking consistency: which model wins at each cutoff?
    rankings: list[dict[str, int]] = []
    for r in per_cutoff:
        models = r.get("models", {})
        cutoff_aucs = {
            mt: models.get(mt, {}).get("auc_roc", float("nan"))
            for mt in MODEL_TYPES
        }
        # Rank by AUC descending (1 = best)
        sorted_models = sorted(cutoff_aucs.items(), key=lambda x: -x[1])
        ranking = {mt: rank + 1 for rank, (mt, _) in enumerate(sorted_models)}
        rankings.append(ranking)

    # Count wins (rank == 1)
    wins: dict[str, int] = {mt: 0 for mt in MODEL_TYPES}
    mean_rank: dict[str, float] = {mt: 0.0 for mt in MODEL_TYPES}
    for ranking in rankings:
        for mt, rank in ranking.items():
            if rank == 1:
                wins[mt] += 1
            mean_rank[mt] += rank
    for mt in MODEL_TYPES:
        mean_rank[mt] /= max(len(rankings), 1)

    return {
        "per_model": aggregated,
        "ranking_consistency": {
            "wins": wins,
            "mean_rank": {mt: float(v) for mt, v in mean_rank.items()},
        },
    }


def _print_summary(aggregated: dict[str, Any]) -> None:
    """Print a summary table to the log."""
    per_model = aggregated.get("per_model", {})
    logger.info("\n=== Merge Rate Experiment Summary ===")
    logger.info("%-15s  %8s  %8s  %8s  %8s", "Model", "Mean ROC", "Std ROC",
                "Mean PR", "Std PR")
    logger.info("-" * 55)
    for mt in MODEL_TYPES:
        agg = per_model.get(mt, {})
        logger.info(
            "%-15s  %8.3f  %8.3f  %8.3f  %8.3f",
            mt,
            agg.get("mean_auc_roc", float("nan")),
            agg.get("std_auc_roc", float("nan")),
            agg.get("mean_auc_pr", float("nan")),
            agg.get("std_auc_pr", float("nan")),
        )

    ranking = aggregated.get("ranking_consistency", {})
    wins = ranking.get("wins", {})
    mean_r = ranking.get("mean_rank", {})
    logger.info("\nRanking consistency (wins / mean rank):")
    for mt in MODEL_TYPES:
        logger.info("  %-15s  wins=%d  mean_rank=%.1f", mt,
                     wins.get(mt, 0), mean_r.get(mt, 0))
