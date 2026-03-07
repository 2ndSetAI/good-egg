from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
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

# Features that leak suspension status — exclude from all models.
LEAKED = {
    "account_age_days", "followers", "public_repos",
    "knn_distance_to_seed", "knn_distance_to_seed_clean",
    "knn_distance_to_seed_temporal", "isolation_forest_score",
}

# Differentially missing — exclude.
DIFFERENTIALLY_MISSING = {"inter_pr_cv", "burst_episode_count"}

# Features with high skew that get log-transformed before scaling.
LOG_TRANSFORM = {"median_additions", "median_files_changed", "career_span_days", "total_prs"}

SMALL_FEATURES = [
    "mean_title_length", "rejection_rate", "merge_rate",
    "career_span_days", "hour_entropy",
]

FULL_FEATURES = [
    "mean_title_length", "rejection_rate", "merge_rate",
    "career_span_days", "hour_entropy", "total_prs",
    "bipartite_clustering", "median_files_changed", "median_additions",
    "hub_score", "total_repos", "empty_body_rate", "title_spam_score",
    "weekend_ratio", "isolation_score", "prs_per_active_day",
]

SINGLE_MODELS = [
    "merge_rate_only", "ge_v2_proxy",
    "susp_lr_small", "susp_lr_full", "susp_gbt",
]

COMBINED_MODELS = ["linear_combo", "stacked", "product", "max_score"]

ALL_MODELS = SINGLE_MODELS + COMBINED_MODELS


def run_stage11(
    base_dir: Path,
    config: StudyConfig,
    cutoffs: list[str] | None = None,
) -> dict[str, Any]:
    """Run two-model pipeline experiment across temporal cutoffs."""
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

        logger.info("=== Two-model pipeline: cutoff %s ===", cutoff_str)
        result = evaluate_single_cutoff(parquet_path)
        result["cutoff"] = cutoff_str
        per_cutoff_results.append(result)

    aggregated = aggregate_across_cutoffs(per_cutoff_results)

    output = {
        "per_cutoff": [
            {
                "cutoff": r["cutoff"],
                "n_labeled": r["n_labeled"],
                "n_suspended": r["n_suspended"],
                "cv_strategy": r["cv_strategy"],
                "models": r["models"],
                "best_susp_classifier": r.get("best_susp_classifier"),
                "delong_vs_ge_proxy": r.get("delong_vs_ge_proxy", {}),
            }
            for r in per_cutoff_results
        ],
        "aggregated": aggregated,
    }

    output_path = base_dir / "data" / "temporal_holdout" / "two_model_pipeline.json"
    write_json(output_path, output)
    logger.info("Results written to %s", output_path)

    _print_summary(aggregated)
    return output


def evaluate_single_cutoff(
    parquet_path: Path,
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Run all single + combined models on one cutoff."""
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

    # --- Single models ---
    models_results: dict[str, Any] = {}
    oof_probs: dict[str, np.ndarray] = {}

    # 1. merge_rate_only baseline
    mr = df["merge_rate"].fillna(0).values.astype(float)
    mr_score = 1.0 - mr  # higher = more suspicious
    models_results["merge_rate_only"] = _compute_metrics(y, mr_score)
    oof_probs["merge_rate_only"] = mr_score

    # 2. ge_v2_proxy: LR on hub_score + merge_rate, negated for suspension
    ge_features = _prepare_features(df, ["hub_score", "merge_rate"])
    ge_result = _run_cv_single(
        "ge_v2_proxy", ge_features, y, n_folds=n_folds, seed=seed,
        use_loo=use_loo, model_class="lr",
    )
    # Negate: GE predicts trust, we want suspension score
    ge_oof = ge_result.pop("_oof_probs")
    ge_susp_oof = 1.0 - ge_oof
    ge_result.update(_compute_metrics(y, ge_susp_oof))
    models_results["ge_v2_proxy"] = ge_result
    oof_probs["ge_v2_proxy"] = ge_susp_oof

    # 3. susp_lr_small
    small_features = _prepare_features(df, SMALL_FEATURES)
    small_result = _run_cv_single(
        "susp_lr_small", small_features, y, n_folds=n_folds, seed=seed,
        use_loo=use_loo, model_class="lr",
    )
    oof_probs["susp_lr_small"] = small_result.pop("_oof_probs")
    models_results["susp_lr_small"] = small_result

    # 4. susp_lr_full
    full_features = _prepare_features(df, FULL_FEATURES)
    full_result = _run_cv_single(
        "susp_lr_full", full_features, y, n_folds=n_folds, seed=seed,
        use_loo=use_loo, model_class="lr",
    )
    oof_probs["susp_lr_full"] = full_result.pop("_oof_probs")
    models_results["susp_lr_full"] = full_result

    # 5. susp_gbt
    gbt_result = _run_cv_single(
        "susp_gbt", full_features, y, n_folds=n_folds, seed=seed,
        use_loo=use_loo, model_class="gbt",
    )
    oof_probs["susp_gbt"] = gbt_result.pop("_oof_probs")
    models_results["susp_gbt"] = gbt_result

    # Full-data fit for inspection
    for name, feats, mc in [
        ("susp_lr_small", small_features, "lr"),
        ("susp_lr_full", full_features, "lr"),
        ("susp_gbt", full_features, "gbt"),
    ]:
        inspection = _fit_and_inspect(name, feats, y, seed=seed, model_class=mc)
        models_results[name].update(inspection)

    # --- Pick best single suspension classifier ---
    susp_candidates = ["susp_lr_small", "susp_lr_full", "susp_gbt"]
    best_susp = max(
        susp_candidates,
        key=lambda m: models_results[m].get("auc_roc", float("-inf")),
    )
    logger.info("  Best suspension classifier: %s (AUC %.3f)",
                best_susp, models_results[best_susp].get("auc_roc", float("nan")))

    # --- Combined models ---
    susp_oof = oof_probs[best_susp]
    ge_susp = oof_probs["ge_v2_proxy"]

    for combo_type in COMBINED_MODELS:
        combo_result = _run_cv_combined(
            combo_type, ge_susp, susp_oof, y,
            n_folds=n_folds, seed=seed, use_loo=use_loo,
        )
        combo_oof = combo_result.pop("_oof_probs")
        oof_probs[combo_type] = combo_oof
        models_results[combo_type] = combo_result

    # --- DeLong tests: each model vs ge_v2_proxy ---
    baseline_probs = oof_probs["ge_v2_proxy"]
    delong_results: dict[str, Any] = {}
    p_values_for_correction: dict[str, float] = {}

    compare_models = [m for m in ALL_MODELS if m != "ge_v2_proxy"]
    for model_name in compare_models:
        alt_probs = oof_probs[model_name]
        if np.all(np.isfinite(baseline_probs)) and np.all(np.isfinite(alt_probs)):
            dl = delong_auc_test(y, alt_probs, baseline_probs)
            delong_results[model_name] = dl
            p_values_for_correction[model_name] = dl["p_value"]

    if p_values_for_correction:
        corrected = holm_bonferroni(p_values_for_correction)
        for model_name, correction in corrected.items():
            delong_results[model_name]["adjusted_p"] = correction["adjusted_p"]
            delong_results[model_name]["reject_h0"] = correction["reject"]

    return {
        "n_labeled": n_labeled,
        "n_suspended": n_suspended,
        "cv_strategy": "loo" if use_loo else f"{n_folds}-fold",
        "models": models_results,
        "best_susp_classifier": best_susp,
        "delong_vs_ge_proxy": delong_results,
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


def _make_model(model_class: str, seed: int = 42) -> Any:
    """Create an sklearn model."""
    if model_class == "gbt":
        return GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=seed,
        )
    return LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=seed,
    )


def _run_cv_single(
    model_name: str,
    features: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
    use_loo: bool = False,
    model_class: str = "lr",
) -> dict[str, Any]:
    """CV for a single model. Returns metrics + _oof_probs."""
    n = len(y)
    oof_probs = np.full(n, np.nan)

    if use_loo:
        for i in range(n):
            train_mask = np.ones(n, dtype=bool)
            train_mask[i] = False

            scaler = StandardScaler()
            x_train = scaler.fit_transform(features[train_mask])
            x_test = scaler.transform(features[~train_mask])

            model = _make_model(model_class, seed)
            model.fit(x_train, y[train_mask])
            oof_probs[i] = model.predict_proba(x_test)[0, 1]
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(np.zeros(n), y):
            scaler = StandardScaler()
            x_train = scaler.fit_transform(features[train_idx])
            x_test = scaler.transform(features[test_idx])

            model = _make_model(model_class, seed)
            model.fit(x_train, y[train_idx])
            oof_probs[test_idx] = model.predict_proba(x_test)[:, 1]

    valid = np.isfinite(oof_probs)
    if not valid.all():
        logger.warning("  %s: %d/%d OOF predictions NaN", model_name, (~valid).sum(), n)

    metrics = _compute_metrics(y, oof_probs)
    metrics["_oof_probs"] = oof_probs
    return metrics


def _run_cv_combined(
    combo_type: str,
    ge_oof: np.ndarray,
    susp_oof: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
    use_loo: bool = False,
) -> dict[str, Any]:
    """Combine two OOF prob vectors via the specified strategy."""
    n = len(y)

    if combo_type == "product":
        combined = ge_oof * susp_oof
        metrics = _compute_metrics(y, combined)
        metrics["_oof_probs"] = combined
        return metrics

    if combo_type == "max_score":
        combined = np.maximum(ge_oof, susp_oof)
        metrics = _compute_metrics(y, combined)
        metrics["_oof_probs"] = combined
        return metrics

    if combo_type == "linear_combo":
        # Tune alpha via inner CV on training folds
        combined = np.full(n, np.nan)
        best_alphas: list[float] = []

        if use_loo:
            # For LOO, tune alpha once on all data (inner 3-fold CV)
            alpha = _tune_alpha_inner(ge_oof, susp_oof, y, seed=seed)
            combined = alpha * ge_oof + (1 - alpha) * susp_oof
            best_alphas.append(alpha)
        else:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            for train_idx, test_idx in skf.split(np.zeros(n), y):
                alpha = _tune_alpha_inner(
                    ge_oof[train_idx], susp_oof[train_idx], y[train_idx], seed=seed,
                )
                combined[test_idx] = alpha * ge_oof[test_idx] + (1 - alpha) * susp_oof[test_idx]
                best_alphas.append(alpha)

        metrics = _compute_metrics(y, combined)
        metrics["mean_alpha"] = float(np.mean(best_alphas))
        metrics["_oof_probs"] = combined
        return metrics

    if combo_type == "stacked":
        # Second-stage LR on the two OOF prob vectors
        combined = np.full(n, np.nan)
        stack_features = np.column_stack([ge_oof, susp_oof])

        if use_loo:
            for i in range(n):
                train_mask = np.ones(n, dtype=bool)
                train_mask[i] = False
                lr = LogisticRegression(max_iter=1000, random_state=seed)
                lr.fit(stack_features[train_mask], y[train_mask])
                combined[i] = lr.predict_proba(stack_features[~train_mask])[0, 1]
        else:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            for train_idx, test_idx in skf.split(np.zeros(n), y):
                lr = LogisticRegression(max_iter=1000, random_state=seed)
                lr.fit(stack_features[train_idx], y[train_idx])
                combined[test_idx] = lr.predict_proba(stack_features[test_idx])[:, 1]

        metrics = _compute_metrics(y, combined)
        metrics["_oof_probs"] = combined
        return metrics

    raise ValueError(f"Unknown combo type: {combo_type}")


def _tune_alpha_inner(
    ge_probs: np.ndarray,
    susp_probs: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    n_inner_folds: int = 3,
) -> float:
    """Inner CV to tune alpha for linear_combo."""
    n = len(y)
    n_pos = int(y.sum())

    if n_pos < n_inner_folds:
        return 0.5  # fallback

    alphas = np.arange(0.1, 1.0, 0.1)
    best_alpha = 0.5
    best_auc = -1.0

    inner_skf = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=seed + 1)

    for alpha in alphas:
        inner_oof = np.full(n, np.nan)
        for _tr_idx, val_idx in inner_skf.split(np.zeros(n), y):
            combo = alpha * ge_probs[val_idx] + (1 - alpha) * susp_probs[val_idx]
            inner_oof[val_idx] = combo

        if np.all(np.isfinite(inner_oof)) and y.sum() > 0:
            try:
                auc = roc_auc_score(y, inner_oof)
                if auc > best_auc:
                    best_auc = auc
                    best_alpha = float(alpha)
            except ValueError:
                pass

    return best_alpha


def _fit_and_inspect(
    model_name: str,
    features: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    model_class: str = "lr",
) -> dict[str, Any]:
    """Full-data fit for coefficient/importance inspection."""
    scaler = StandardScaler()
    features_s = scaler.fit_transform(features)

    model = _make_model(model_class, seed)
    model.fit(features_s, y)

    result: dict[str, Any] = {}

    if model_class == "gbt":
        # Use FULL_FEATURES names for gbt
        names = FULL_FEATURES
        importances = dict(zip(names, [float(v) for v in model.feature_importances_], strict=True))
        result["feature_importances"] = importances
    else:
        # Determine feature names from model_name
        names = SMALL_FEATURES if "small" in model_name else FULL_FEATURES
        coefficients = dict(zip(names, [float(v) for v in model.coef_[0]], strict=True))
        coefficients["intercept"] = float(model.intercept_[0])
        result["coefficients"] = coefficients

    return result


def aggregate_across_cutoffs(
    per_cutoff: list[dict[str, Any]],
) -> dict[str, Any]:
    """Mean/std across cutoffs per model. Ranking consistency."""
    if not per_cutoff:
        return {}

    aggregated: dict[str, dict[str, Any]] = {}

    for model_name in ALL_MODELS:
        aucs_roc: list[float] = []
        aucs_pr: list[float] = []
        prec25: list[float] = []
        prec50: list[float] = []

        for r in per_cutoff:
            model_result = r.get("models", {}).get(model_name, {})
            for val, dest in [
                (model_result.get("auc_roc"), aucs_roc),
                (model_result.get("auc_pr"), aucs_pr),
                (model_result.get("precision_at_25"), prec25),
                (model_result.get("precision_at_50"), prec50),
            ]:
                if val is not None and not np.isnan(val):
                    dest.append(val)

        agg: dict[str, Any] = {}
        for metric_name, values in [
            ("auc_roc", aucs_roc), ("auc_pr", aucs_pr),
            ("precision_at_25", prec25), ("precision_at_50", prec50),
        ]:
            if values:
                arr = np.array(values)
                agg[f"mean_{metric_name}"] = float(np.mean(arr))
                agg[f"std_{metric_name}"] = float(np.std(arr))
                agg[f"n_cutoffs_{metric_name}"] = len(values)
        aggregated[model_name] = agg

    # Ranking consistency by AUC-ROC
    rankings: list[dict[str, int]] = []
    for r in per_cutoff:
        models = r.get("models", {})
        cutoff_aucs = {
            m: models.get(m, {}).get("auc_roc", float("nan"))
            for m in ALL_MODELS
        }
        sorted_models = sorted(cutoff_aucs.items(), key=lambda x: -x[1])
        ranking = {m: rank + 1 for rank, (m, _) in enumerate(sorted_models)}
        rankings.append(ranking)

    wins: dict[str, int] = {m: 0 for m in ALL_MODELS}
    mean_rank: dict[str, float] = {m: 0.0 for m in ALL_MODELS}
    for ranking in rankings:
        for m, rank in ranking.items():
            if rank == 1:
                wins[m] += 1
            mean_rank[m] += rank
    for m in ALL_MODELS:
        mean_rank[m] /= max(len(rankings), 1)

    return {
        "per_model": aggregated,
        "ranking_consistency": {
            "wins": wins,
            "mean_rank": {m: float(v) for m, v in mean_rank.items()},
        },
    }


def _print_summary(aggregated: dict[str, Any]) -> None:
    """Print a summary table to the log."""
    per_model = aggregated.get("per_model", {})
    logger.info("\n=== Two-Model Pipeline Summary ===")
    logger.info(
        "%-20s  %8s  %8s  %8s  %8s",
        "Model", "Mean ROC", "Std ROC", "Mean PR", "Std PR",
    )
    logger.info("-" * 60)
    for m in ALL_MODELS:
        agg = per_model.get(m, {})
        logger.info(
            "%-20s  %8.3f  %8.3f  %8.3f  %8.3f",
            m,
            agg.get("mean_auc_roc", float("nan")),
            agg.get("std_auc_roc", float("nan")),
            agg.get("mean_auc_pr", float("nan")),
            agg.get("std_auc_pr", float("nan")),
        )

    ranking = aggregated.get("ranking_consistency", {})
    wins = ranking.get("wins", {})
    mean_r = ranking.get("mean_rank", {})
    logger.info("\nRanking consistency (wins / mean rank):")
    for m in ALL_MODELS:
        logger.info(
            "  %-20s  wins=%d  mean_rank=%.1f",
            m, wins.get(m, 0), mean_r.get(m, 0),
        )
