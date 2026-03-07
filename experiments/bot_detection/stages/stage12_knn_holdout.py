from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from experiments.bot_detection.checkpoint import write_json
from experiments.bot_detection.models import StudyConfig
from experiments.bot_detection.stats import delong_auc_test, holm_bonferroni

logger = logging.getLogger(__name__)

DEFAULT_CUTOFFS = [
    "2020-01-01", "2021-01-01", "2022-01-01",
    "2022-07-01", "2023-01-01", "2024-01-01",
]

LOG_TRANSFORM = {"median_additions", "median_files_changed", "career_span_days", "total_prs"}

FULL_FEATURES = [
    "mean_title_length", "rejection_rate", "merge_rate",
    "career_span_days", "hour_entropy", "total_prs",
    "bipartite_clustering", "median_files_changed", "median_additions",
    "hub_score", "total_repos", "empty_body_rate", "title_spam_score",
    "weekend_ratio", "isolation_score", "prs_per_active_day",
]

FEATURES_NO_MR = [f for f in FULL_FEATURES if f != "merge_rate"]

KNN_VARIANTS = [
    {"name": "knn_safe16_euclidean", "features": "full", "metric": "euclidean", "k": 5},
    {"name": "knn_safe15_no_mr", "features": "no_mr", "metric": "euclidean", "k": 5},
    {"name": "knn_safe16_cosine", "features": "full", "metric": "cosine", "k": 5},
]

K_SWEEP_VALUES = [3, 5, 10, 15]


def run_stage12(
    base_dir: Path,
    config: StudyConfig,
    cutoffs: list[str] | None = None,
) -> dict[str, Any]:
    """Run k-NN holdout experiment across temporal cutoffs."""
    if cutoffs is None:
        holdout_config = config.author_analysis.get("temporal_holdout", {})
        cutoffs = holdout_config.get("cutoffs", DEFAULT_CUTOFFS)

    per_cutoff_a: list[dict[str, Any]] = []
    per_cutoff_b: list[dict[str, Any]] = []

    for cutoff_str in cutoffs:
        parquet_path = (
            base_dir / "data" / "temporal_holdout"
            / f"T_{cutoff_str}" / "author_features.parquet"
        )
        if not parquet_path.exists():
            logger.warning("No parquet at %s, skipping cutoff %s", parquet_path, cutoff_str)
            continue

        logger.info("=== k-NN holdout: cutoff %s ===", cutoff_str)

        result_a = evaluate_cutoff_experiment_a(parquet_path)
        result_a["cutoff"] = cutoff_str
        per_cutoff_a.append(result_a)

        result_b = evaluate_cutoff_experiment_b(parquet_path)
        result_b["cutoff"] = cutoff_str
        per_cutoff_b.append(result_b)

    agg_a = aggregate_experiment_a(per_cutoff_a)
    agg_b = aggregate_experiment_b(per_cutoff_b)

    output = {
        "experiment_a": {
            "per_cutoff": per_cutoff_a,
            "aggregated": agg_a,
        },
        "experiment_b": {
            "per_cutoff": per_cutoff_b,
            "aggregated": agg_b,
        },
    }

    output_path = base_dir / "data" / "temporal_holdout" / "knn_holdout_experiment.json"
    write_json(output_path, output)
    logger.info("Results written to %s", output_path)

    _print_summary(agg_a, agg_b)
    return output


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


def _knn_score(
    seed_features: np.ndarray,
    eval_features: np.ndarray,
    k: int,
    metric: str,
) -> np.ndarray:
    """Fit NearestNeighbors on seeds, score eval set.

    Returns negative mean distance (higher = more suspicious = closer to seeds).
    """
    effective_k = min(k, len(seed_features))
    nn = NearestNeighbors(n_neighbors=effective_k, metric=metric)
    nn.fit(seed_features)
    distances, _ = nn.kneighbors(eval_features)
    mean_distances = distances.mean(axis=1)
    return -mean_distances


def _run_knn_cv(
    features: np.ndarray,
    y: np.ndarray,
    k: int,
    metric: str,
    n_folds: int = 5,
    seed: int = 42,
    use_loo: bool = False,
) -> dict[str, Any]:
    """k-NN CV with suspended-only splitting.

    Splits ONLY suspended accounts into seed/eval folds.
    Active accounts appear in every fold's eval set.
    """
    susp_idx = np.where(y == 1)[0]
    active_idx = np.where(y == 0)[0]
    n_susp = len(susp_idx)

    active_score_accum = np.zeros(len(active_idx))
    active_fold_count = np.zeros(len(active_idx))
    susp_oof_scores = np.full(n_susp, np.nan)

    per_fold_auc: list[float] = []

    if use_loo:
        for i in range(n_susp):
            held_out_susp_pos = np.array([i])
            seed_susp_pos = np.array([j for j in range(n_susp) if j != i])

            if len(seed_susp_pos) == 0:
                continue

            seed_susp_idx = susp_idx[seed_susp_pos]
            held_out_susp_idx = susp_idx[held_out_susp_pos]

            effective_k = min(k, len(seed_susp_idx))
            if effective_k == 0:
                continue

            train_idx = np.concatenate([seed_susp_idx, active_idx])
            scaler = StandardScaler()
            scaler.fit(features[train_idx])

            seed_features_scaled = scaler.transform(features[seed_susp_idx])

            eval_idx = np.concatenate([held_out_susp_idx, active_idx])
            eval_features_scaled = scaler.transform(features[eval_idx])

            scores = _knn_score(seed_features_scaled, eval_features_scaled, k, metric)

            n_held_out = len(held_out_susp_idx)
            susp_oof_scores[held_out_susp_pos] = scores[:n_held_out]
            active_score_accum += scores[n_held_out:]
            active_fold_count += 1

            # Per-fold AUC: 1 held-out suspended + all active
            fold_y = np.concatenate([np.ones(n_held_out), np.zeros(len(active_idx))])
            fold_scores = scores
            if fold_y.sum() > 0 and (1 - fold_y).sum() > 0:
                with contextlib.suppress(ValueError):
                    per_fold_auc.append(float(roc_auc_score(fold_y, fold_scores)))
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold_train_pos, fold_test_pos in kf.split(np.arange(n_susp)):
            seed_susp_idx = susp_idx[fold_train_pos]
            held_out_susp_idx = susp_idx[fold_test_pos]

            effective_k = min(k, len(seed_susp_idx))
            if effective_k == 0:
                continue

            train_idx = np.concatenate([seed_susp_idx, active_idx])
            scaler = StandardScaler()
            scaler.fit(features[train_idx])

            seed_features_scaled = scaler.transform(features[seed_susp_idx])

            eval_idx = np.concatenate([held_out_susp_idx, active_idx])
            eval_features_scaled = scaler.transform(features[eval_idx])

            scores = _knn_score(seed_features_scaled, eval_features_scaled, k, metric)

            n_held_out = len(held_out_susp_idx)
            susp_oof_scores[fold_test_pos] = scores[:n_held_out]
            active_score_accum += scores[n_held_out:]
            active_fold_count += 1

            fold_y = np.concatenate([np.ones(n_held_out), np.zeros(len(active_idx))])
            fold_scores = scores
            if fold_y.sum() > 0 and (1 - fold_y).sum() > 0:
                with contextlib.suppress(ValueError):
                    per_fold_auc.append(float(roc_auc_score(fold_y, fold_scores)))

    # Average active scores across folds
    safe_count = np.where(active_fold_count > 0, active_fold_count, 1.0)
    active_avg_scores = active_score_accum / safe_count

    # Combine into full OOF score array
    oof_scores = np.full(len(y), np.nan)
    oof_scores[susp_idx] = susp_oof_scores
    oof_scores[active_idx] = active_avg_scores

    metrics = _compute_metrics(y, oof_scores)
    metrics["per_fold_auc_roc"] = per_fold_auc
    metrics["_oof_scores"] = oof_scores
    return metrics


def _run_lr_cv(
    features: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
    use_loo: bool = False,
) -> np.ndarray:
    """Standard LR CV baseline. Returns OOF probability array."""
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

    return oof_probs


def evaluate_cutoff_experiment_a(
    parquet_path: Path,
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Experiment A: k-NN with safe CV splitting vs baselines."""
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
        logger.info("  Using %d-fold CV", n_folds)

    # --- k-NN variants ---
    variants_results: dict[str, Any] = {}
    oof_scores: dict[str, np.ndarray] = {}

    for variant in KNN_VARIANTS:
        name = variant["name"]
        feat_list = FULL_FEATURES if variant["features"] == "full" else FEATURES_NO_MR
        features = _prepare_features(df, feat_list)

        result = _run_knn_cv(
            features, y, k=variant["k"], metric=variant["metric"],
            n_folds=n_folds, seed=seed, use_loo=use_loo,
        )
        oof = result.pop("_oof_scores")
        oof_scores[name] = oof
        variants_results[name] = result
        logger.info("  %s: AUC-ROC=%.3f", name, result.get("auc_roc", float("nan")))

    # --- k sweep on best variant ---
    best_variant_name = max(
        variants_results,
        key=lambda m: variants_results[m].get("auc_roc", float("-inf")),
    )
    best_variant = next(v for v in KNN_VARIANTS if v["name"] == best_variant_name)
    best_feat_list = FULL_FEATURES if best_variant["features"] == "full" else FEATURES_NO_MR
    best_features = _prepare_features(df, best_feat_list)

    # Min seed set size across folds
    min_seeds = n_suspended - 1 if use_loo else n_suspended - n_suspended // n_folds

    k_sweep_results: dict[str, Any] = {}
    for k_val in K_SWEEP_VALUES:
        if k_val > min_seeds:
            logger.info("  Skipping k=%d (> min_seeds=%d)", k_val, min_seeds)
            continue
        result = _run_knn_cv(
            best_features, y, k=k_val, metric=best_variant["metric"],
            n_folds=n_folds, seed=seed, use_loo=use_loo,
        )
        result.pop("_oof_scores", None)
        k_sweep_results[f"k={k_val}"] = result
        logger.info(
            "  k=%d: AUC-ROC=%.3f", k_val, result.get("auc_roc", float("nan")),
        )

    # --- Baselines ---
    baselines: dict[str, Any] = {}

    # merge_rate_only
    mr = df["merge_rate"].fillna(0).values.astype(float)
    mr_score = 1.0 - mr
    baselines["merge_rate_only"] = _compute_metrics(y, mr_score)
    oof_scores["merge_rate_only"] = mr_score

    # susp_lr_full
    full_features = _prepare_features(df, FULL_FEATURES)
    lr_oof = _run_lr_cv(full_features, y, n_folds=n_folds, seed=seed, use_loo=use_loo)
    baselines["susp_lr_full"] = _compute_metrics(y, lr_oof)
    oof_scores["susp_lr_full"] = lr_oof

    # --- DeLong tests ---
    delong_results: dict[str, Any] = {}
    p_values_for_correction: dict[str, float] = {}

    for variant_name in variants_results:
        variant_oof = oof_scores[variant_name]

        for baseline_name in ["merge_rate_only", "susp_lr_full"]:
            baseline_oof = oof_scores[baseline_name]
            test_key = f"{variant_name}_vs_{baseline_name}"

            if np.all(np.isfinite(variant_oof)) and np.all(np.isfinite(baseline_oof)):
                dl = delong_auc_test(y, variant_oof, baseline_oof)
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
        "low_power": n_suspended < 15,
        "variants": variants_results,
        "best_variant": best_variant_name,
        "k_sweep": k_sweep_results,
        "baselines": baselines,
        "delong_tests": delong_results,
    }


def evaluate_cutoff_experiment_b(
    parquet_path: Path,
    k: int = 5,
) -> dict[str, Any]:
    """Experiment B: k-NN distance predicts merge_rate among active accounts."""
    df = pd.read_parquet(parquet_path)
    df = df[df["account_status"].isin(["active", "suspended"])].copy()
    y = (df["account_status"] == "suspended").astype(int).values

    susp_mask = y == 1
    active_mask = y == 0
    n_seeds = int(susp_mask.sum())
    n_active = int(active_mask.sum())

    if n_seeds == 0 or n_active == 0:
        return {
            "n_active": n_active,
            "n_seeds": n_seeds,
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
            "quartile_merge_rates": [],
            "binary_auc": float("nan"),
        }

    # Use FEATURES_NO_MR (no merge_rate) to avoid circularity
    features = _prepare_features(df, FEATURES_NO_MR)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    seed_features = features_scaled[susp_mask]
    active_features = features_scaled[active_mask]

    scores = _knn_score(seed_features, active_features, k, "euclidean")

    active_merge_rate = df.loc[active_mask, "merge_rate"].fillna(0).values.astype(float)

    # Spearman correlation
    rho, p_val = sp_stats.spearmanr(scores, active_merge_rate)

    # Quartile analysis
    quartile_bins = np.percentile(scores, [0, 25, 50, 75, 100])
    quartile_merge_rates: list[dict[str, Any]] = []
    for q in range(4):
        low = quartile_bins[q]
        high = quartile_bins[q + 1]
        mask = (scores >= low) & (scores < high) if q < 3 else (scores >= low) & (scores <= high)
        if mask.sum() > 0:
            quartile_merge_rates.append({
                "quartile": q + 1,
                "n": int(mask.sum()),
                "mean_merge_rate": float(active_merge_rate[mask].mean()),
                "score_range": [float(low), float(high)],
            })

    # Binary AUC: predict merge_rate < 0.3
    low_mr = (active_merge_rate < 0.3).astype(int)
    binary_auc = float("nan")
    if low_mr.sum() > 0 and (1 - low_mr).sum() > 0:
        with contextlib.suppress(ValueError):
            binary_auc = float(roc_auc_score(low_mr, scores))

    return {
        "n_active": n_active,
        "n_seeds": n_seeds,
        "spearman_rho": float(rho),
        "spearman_p": float(p_val),
        "quartile_merge_rates": quartile_merge_rates,
        "binary_auc": binary_auc,
        "note": "scores use FEATURES_NO_MR; check rejection_rate correlation separately",
    }


def aggregate_experiment_a(
    per_cutoff: list[dict[str, Any]],
) -> dict[str, Any]:
    """Mean/std across cutoffs for each variant + baselines."""
    if not per_cutoff:
        return {}

    all_model_names: list[str] = []
    for r in per_cutoff:
        for name in r.get("variants", {}):
            if name not in all_model_names:
                all_model_names.append(name)
    for name in ["merge_rate_only", "susp_lr_full"]:
        if name not in all_model_names:
            all_model_names.append(name)

    per_model: dict[str, dict[str, Any]] = {}

    for model_name in all_model_names:
        aucs_roc: list[float] = []
        aucs_pr: list[float] = []
        prec25: list[float] = []
        prec50: list[float] = []

        for r in per_cutoff:
            # Check variants first, then baselines
            model_result = r.get("variants", {}).get(model_name, {})
            if not model_result:
                model_result = r.get("baselines", {}).get(model_name, {})
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
        per_model[model_name] = agg

    # Stable-only subset (n_suspended >= 30)
    stable_cutoffs = [r for r in per_cutoff if r.get("n_suspended", 0) >= 30]
    stable_only: dict[str, dict[str, Any]] = {}
    if stable_cutoffs:
        for model_name in all_model_names:
            aucs: list[float] = []
            for r in stable_cutoffs:
                model_result = r.get("variants", {}).get(model_name, {})
                if not model_result:
                    model_result = r.get("baselines", {}).get(model_name, {})
                val = model_result.get("auc_roc")
                if val is not None and not np.isnan(val):
                    aucs.append(val)
            if aucs:
                arr = np.array(aucs)
                stable_only[model_name] = {
                    "mean_auc_roc": float(np.mean(arr)),
                    "std_auc_roc": float(np.std(arr)),
                    "n_cutoffs": len(aucs),
                }

    # Ranking consistency by AUC-ROC
    rankings: list[dict[str, int]] = []
    for r in per_cutoff:
        cutoff_aucs: dict[str, float] = {}
        for name in all_model_names:
            model_result = r.get("variants", {}).get(name, {})
            if not model_result:
                model_result = r.get("baselines", {}).get(name, {})
            cutoff_aucs[name] = model_result.get("auc_roc", float("nan"))
        sorted_models = sorted(cutoff_aucs.items(), key=lambda x: -x[1])
        ranking = {m: rank + 1 for rank, (m, _) in enumerate(sorted_models)}
        rankings.append(ranking)

    wins: dict[str, int] = {m: 0 for m in all_model_names}
    mean_rank: dict[str, float] = {m: 0.0 for m in all_model_names}
    for ranking in rankings:
        for m, rank in ranking.items():
            if rank == 1:
                wins[m] += 1
            mean_rank[m] += rank
    for m in all_model_names:
        mean_rank[m] /= max(len(rankings), 1)

    return {
        "per_model": per_model,
        "stable_only": stable_only,
        "ranking_consistency": {
            "wins": wins,
            "mean_rank": {m: float(v) for m, v in mean_rank.items()},
        },
    }


def aggregate_experiment_b(
    per_cutoff: list[dict[str, Any]],
) -> dict[str, Any]:
    """Mean/std of Spearman rho, quartile stats, binary AUC across cutoffs."""
    if not per_cutoff:
        return {}

    rhos: list[float] = []
    binary_aucs: list[float] = []

    for r in per_cutoff:
        rho = r.get("spearman_rho")
        if rho is not None and not np.isnan(rho):
            rhos.append(rho)
        ba = r.get("binary_auc")
        if ba is not None and not np.isnan(ba):
            binary_aucs.append(ba)

    # Aggregate quartile merge rates
    max_quartiles = 4
    quartile_means: dict[int, list[float]] = {q: [] for q in range(1, max_quartiles + 1)}
    for r in per_cutoff:
        for qr in r.get("quartile_merge_rates", []):
            q = qr["quartile"]
            quartile_means[q].append(qr["mean_merge_rate"])

    quartile_summary: list[dict[str, Any]] = []
    for q in range(1, max_quartiles + 1):
        vals = quartile_means[q]
        if vals:
            arr = np.array(vals)
            quartile_summary.append({
                "quartile": q,
                "mean_merge_rate": float(np.mean(arr)),
                "std_merge_rate": float(np.std(arr)),
                "n_cutoffs": len(vals),
            })

    result: dict[str, Any] = {
        "n_cutoffs": len(per_cutoff),
        "quartile_summary": quartile_summary,
    }

    if rhos:
        arr = np.array(rhos)
        result["mean_spearman_rho"] = float(np.mean(arr))
        result["std_spearman_rho"] = float(np.std(arr))

    if binary_aucs:
        arr = np.array(binary_aucs)
        result["mean_binary_auc"] = float(np.mean(arr))
        result["std_binary_auc"] = float(np.std(arr))

    return result


def _print_summary(agg_a: dict[str, Any], agg_b: dict[str, Any]) -> None:
    """Print summary tables for both experiments."""
    # Experiment A
    per_model = agg_a.get("per_model", {})
    logger.info("\n=== Experiment A: k-NN Holdout CV ===")
    logger.info(
        "%-25s  %8s  %8s  %8s  %8s",
        "Model", "Mean ROC", "Std ROC", "Mean PR", "Std PR",
    )
    logger.info("-" * 65)
    for m, agg in per_model.items():
        logger.info(
            "%-25s  %8.3f  %8.3f  %8.3f  %8.3f",
            m,
            agg.get("mean_auc_roc", float("nan")),
            agg.get("std_auc_roc", float("nan")),
            agg.get("mean_auc_pr", float("nan")),
            agg.get("std_auc_pr", float("nan")),
        )

    stable = agg_a.get("stable_only", {})
    if stable:
        logger.info("\nStable cutoffs (n_suspended >= 30):")
        for m, s in stable.items():
            logger.info(
                "  %-25s  mean_auc=%.3f  std=%.3f  n=%d",
                m, s.get("mean_auc_roc", float("nan")),
                s.get("std_auc_roc", float("nan")),
                s.get("n_cutoffs", 0),
            )

    ranking = agg_a.get("ranking_consistency", {})
    wins = ranking.get("wins", {})
    mean_r = ranking.get("mean_rank", {})
    if wins:
        logger.info("\nRanking consistency (wins / mean rank):")
        for m in per_model:
            logger.info(
                "  %-25s  wins=%d  mean_rank=%.1f",
                m, wins.get(m, 0), mean_r.get(m, 0),
            )

    # Experiment B
    logger.info("\n=== Experiment B: k-NN Distance vs Merge Rate ===")
    logger.info(
        "Mean Spearman rho: %.3f (std %.3f)",
        agg_b.get("mean_spearman_rho", float("nan")),
        agg_b.get("std_spearman_rho", float("nan")),
    )
    logger.info(
        "Mean binary AUC (mr<0.3): %.3f (std %.3f)",
        agg_b.get("mean_binary_auc", float("nan")),
        agg_b.get("std_binary_auc", float("nan")),
    )
    for qs in agg_b.get("quartile_summary", []):
        logger.info(
            "  Q%d: mean_merge_rate=%.3f (std %.3f, n=%d cutoffs)",
            qs["quartile"],
            qs["mean_merge_rate"],
            qs["std_merge_rate"],
            qs["n_cutoffs"],
        )
