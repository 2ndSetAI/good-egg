from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
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

WINDOW_VARIANTS = ["mr_alltime", "mr_1yr", "mr_6mo", "mr_3mo", "mr_weighted"]


def run_stage13(
    base_dir: Path,
    config: StudyConfig,
    cutoffs: list[str] | None = None,
) -> dict[str, Any]:
    """Run merge prediction experiments (A, C, D) across temporal cutoffs."""
    if cutoffs is None:
        holdout_config = config.author_analysis.get("temporal_holdout", {})
        cutoffs = holdout_config.get("cutoffs", DEFAULT_CUTOFFS)

    db_path = base_dir / "data" / "bot_detection.duckdb"

    per_cutoff_a: list[dict[str, Any]] = []
    per_cutoff_c: list[dict[str, Any]] = []
    per_cutoff_d: list[dict[str, Any]] = []

    for cutoff_str in cutoffs:
        parquet_path = (
            base_dir / "data" / "temporal_holdout"
            / f"T_{cutoff_str}" / "author_features.parquet"
        )
        if not parquet_path.exists():
            logger.warning("No parquet at %s, skipping cutoff %s", parquet_path, cutoff_str)
            continue

        logger.info("=== Merge prediction: cutoff %s ===", cutoff_str)

        post_mr_df = _query_post_cutoff_merge_rates(db_path, cutoff_str)
        if post_mr_df.empty:
            logger.warning("  No post-cutoff PRs for %s, skipping", cutoff_str)
            continue

        df = pd.read_parquet(parquet_path)
        merged = df.merge(post_mr_df, on="login", how="inner")
        merged = merged[merged["post_total_prs"] >= 1].copy()

        if len(merged) < 10:
            logger.warning("  Only %d authors with post-cutoff PRs, skipping", len(merged))
            continue

        logger.info("  %d authors with post-cutoff PRs", len(merged))

        result_a = _run_experiment_a(merged)
        result_a["cutoff"] = cutoff_str
        per_cutoff_a.append(result_a)

        result_c = _run_experiment_c(merged, df)
        result_c["cutoff"] = cutoff_str
        per_cutoff_c.append(result_c)

        windowed_df = _query_windowed_merge_rates(db_path, cutoff_str)
        result_d = _run_experiment_d(merged, windowed_df)
        result_d["cutoff"] = cutoff_str
        per_cutoff_d.append(result_d)

    agg_a = _aggregate_experiment_a(per_cutoff_a)
    agg_c = _aggregate_experiment_c(per_cutoff_c)
    agg_d = _aggregate_experiment_d(per_cutoff_d)

    output = {
        "experiment_a": {"per_cutoff": per_cutoff_a, "aggregated": agg_a},
        "experiment_c": {"per_cutoff": per_cutoff_c, "aggregated": agg_c},
        "experiment_d": {"per_cutoff": per_cutoff_d, "aggregated": agg_d},
    }

    output_path = base_dir / "data" / "temporal_holdout" / "merge_prediction_experiment.json"
    write_json(output_path, output)
    logger.info("Results written to %s", output_path)

    _print_summary(agg_a, agg_c, agg_d)
    return output


# ---------------------------------------------------------------------------
# Data queries
# ---------------------------------------------------------------------------

def _query_post_cutoff_merge_rates(db_path: Path, cutoff_str: str) -> pd.DataFrame:
    """Query post-cutoff merge rates from DuckDB."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        result = con.execute(
            """
            SELECT author AS login,
                   AVG(CASE WHEN state='MERGED' THEN 1.0 ELSE 0.0 END) AS post_mr,
                   COUNT(*) AS post_total_prs,
                   SUM(CASE WHEN state='MERGED' THEN 1 ELSE 0 END) AS post_merged_prs
            FROM prs
            WHERE created_at >= ?::TIMESTAMP
            GROUP BY author
            """,
            [cutoff_str],
        ).fetchdf()
    finally:
        con.close()
    return result


def _query_windowed_merge_rates(db_path: Path, cutoff_str: str) -> pd.DataFrame:
    """Query per-author merge rates in lookback windows before cutoff."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        # Alltime and fixed windows via SQL
        agg_df = con.execute(
            """
            SELECT
                author AS login,
                AVG(CASE WHEN state='MERGED' THEN 1.0 ELSE 0.0 END) AS mr_alltime,
                COUNT(*) AS n_alltime,
                AVG(CASE WHEN created_at >= ?::TIMESTAMP - INTERVAL '1 year'
                    THEN (CASE WHEN state='MERGED' THEN 1.0 ELSE 0.0 END)
                    ELSE NULL END) AS mr_1yr,
                SUM(CASE WHEN created_at >= ?::TIMESTAMP - INTERVAL '1 year'
                    THEN 1 ELSE 0 END) AS n_1yr,
                AVG(CASE WHEN created_at >= ?::TIMESTAMP - INTERVAL '6 months'
                    THEN (CASE WHEN state='MERGED' THEN 1.0 ELSE 0.0 END)
                    ELSE NULL END) AS mr_6mo,
                SUM(CASE WHEN created_at >= ?::TIMESTAMP - INTERVAL '6 months'
                    THEN 1 ELSE 0 END) AS n_6mo,
                AVG(CASE WHEN created_at >= ?::TIMESTAMP - INTERVAL '3 months'
                    THEN (CASE WHEN state='MERGED' THEN 1.0 ELSE 0.0 END)
                    ELSE NULL END) AS mr_3mo,
                SUM(CASE WHEN created_at >= ?::TIMESTAMP - INTERVAL '3 months'
                    THEN 1 ELSE 0 END) AS n_3mo
            FROM prs
            WHERE created_at < ?::TIMESTAMP
            GROUP BY author
            """,
            [cutoff_str] * 7,
        ).fetchdf()

        # Exponentially weighted merge rate: fetch individual PRs
        pr_df = con.execute(
            """
            SELECT author AS login,
                   created_at,
                   CASE WHEN state='MERGED' THEN 1.0 ELSE 0.0 END AS merged
            FROM prs
            WHERE created_at < ?::TIMESTAMP
            """,
            [cutoff_str],
        ).fetchdf()
    finally:
        con.close()

    # Compute exponentially weighted merge rate (half-life 180 days)
    if not pr_df.empty:
        cutoff_ts = pd.Timestamp(cutoff_str)
        pr_df["age_days"] = (cutoff_ts - pr_df["created_at"]).dt.total_seconds() / 86400.0
        half_life = 180.0
        pr_df["weight"] = np.exp(-np.log(2) * pr_df["age_days"] / half_life)

        weighted = pr_df.groupby("login").apply(
            lambda g: np.average(g["merged"], weights=g["weight"]),
            include_groups=False,
        ).rename("mr_weighted")
        agg_df = agg_df.merge(weighted.reset_index(), on="login", how="left")
    else:
        agg_df["mr_weighted"] = np.nan

    return agg_df


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _prepare_features(df: pd.DataFrame, feature_list: list[str]) -> np.ndarray:
    """Extract columns, log-transform skewed ones, fill NaN with 0."""
    arrays = []
    for col in feature_list:
        vals = df[col].fillna(0).values.astype(float)
        if col in LOG_TRANSFORM:
            vals = np.log1p(np.abs(vals)) * np.sign(vals)
        arrays.append(vals)
    return np.column_stack(arrays)


def _compute_metrics(
    y: np.ndarray, scores: np.ndarray, post_mr: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute AUC-ROC, AUC-PR, and optionally Spearman correlation."""
    metrics: dict[str, Any] = {}
    if y.sum() > 0 and (1 - y).sum() > 0 and np.all(np.isfinite(scores)):
        metrics["auc_roc"] = float(roc_auc_score(y, scores))
        metrics["auc_pr"] = float(average_precision_score(y, scores))
    else:
        metrics["auc_roc"] = float("nan")
        metrics["auc_pr"] = float("nan")

    if post_mr is not None and np.all(np.isfinite(scores)):
        rho, p_val = sp_stats.spearmanr(scores, post_mr)
        metrics["spearman_rho"] = float(rho)
        metrics["spearman_p"] = float(p_val)

    return metrics


def _run_cv_single(
    features: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
    use_loo: bool = False,
) -> np.ndarray:
    """LR CV returning OOF probability array."""
    n = len(y)
    oof_probs = np.full(n, np.nan)

    if use_loo:
        for i in range(n):
            train_mask = np.ones(n, dtype=bool)
            train_mask[i] = False
            scaler = StandardScaler()
            x_train = scaler.fit_transform(features[train_mask])
            x_test = scaler.transform(features[~train_mask])
            model = LogisticRegression(max_iter=1000, random_state=seed)
            model.fit(x_train, y[train_mask])
            oof_probs[i] = model.predict_proba(x_test)[0, 1]
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(np.zeros(n), y):
            scaler = StandardScaler()
            x_train = scaler.fit_transform(features[train_idx])
            x_test = scaler.transform(features[test_idx])
            model = LogisticRegression(max_iter=1000, random_state=seed)
            model.fit(x_train, y[train_idx])
            oof_probs[test_idx] = model.predict_proba(x_test)[:, 1]

    return oof_probs


def _knn_score(
    seed_features: np.ndarray,
    eval_features: np.ndarray,
    k: int,
    metric: str,
) -> np.ndarray:
    """Fit NearestNeighbors on seeds, score eval set.

    Returns negative mean distance (higher = closer to seeds).
    """
    effective_k = min(k, len(seed_features))
    nn = NearestNeighbors(n_neighbors=effective_k, metric=metric)
    nn.fit(seed_features)
    distances, _ = nn.kneighbors(eval_features)
    return -distances.mean(axis=1)


# ---------------------------------------------------------------------------
# Experiment A: Merge prediction with multiple models
# ---------------------------------------------------------------------------

def _run_experiment_a(
    df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Experiment A: predict post-cutoff merge rate >= 0.5."""
    y = (df["post_mr"] >= 0.5).astype(int).values
    post_mr = df["post_mr"].values.astype(float)
    n_pos = int(y.sum())
    n_total = len(y)
    use_loo = n_pos < 30

    logger.info(
        "  Exp A: %d authors, %d high-merge (>=0.5), cv=%s",
        n_total, n_pos, "loo" if use_loo else f"{n_folds}-fold",
    )

    models_results: dict[str, Any] = {}
    oof_scores: dict[str, np.ndarray] = {}

    # 1. merge_rate_only: raw pre-cutoff merge_rate as score
    mr = df["merge_rate"].fillna(0).values.astype(float)
    models_results["merge_rate_only"] = _compute_metrics(y, mr, post_mr)
    oof_scores["merge_rate_only"] = mr

    # 2. ge_v2_proxy: LR(hub_score, merge_rate)
    ge_features = _prepare_features(df, ["hub_score", "merge_rate"])
    ge_oof = _run_cv_single(ge_features, y, n_folds=n_folds, seed=seed, use_loo=use_loo)
    models_results["ge_v2_proxy"] = _compute_metrics(y, ge_oof, post_mr)
    oof_scores["ge_v2_proxy"] = ge_oof

    # 3. lr_full: LR on all 16 features
    full_features = _prepare_features(df, FULL_FEATURES)
    lr_full_oof = _run_cv_single(
        full_features, y, n_folds=n_folds, seed=seed, use_loo=use_loo,
    )
    models_results["lr_full"] = _compute_metrics(y, lr_full_oof, post_mr)
    oof_scores["lr_full"] = lr_full_oof

    # 4. lr_full_no_mr: LR on 15 features (no merge_rate)
    no_mr_features = _prepare_features(df, FEATURES_NO_MR)
    lr_no_mr_oof = _run_cv_single(
        no_mr_features, y, n_folds=n_folds, seed=seed, use_loo=use_loo,
    )
    models_results["lr_full_no_mr"] = _compute_metrics(y, lr_no_mr_oof, post_mr)
    oof_scores["lr_full_no_mr"] = lr_no_mr_oof

    # 5. knn_cosine: k-NN with cosine distance on FEATURES_NO_MR
    knn_oof = _run_knn_merge_cv(
        no_mr_features, y, k=5, metric="cosine",
        n_folds=n_folds, seed=seed, use_loo=use_loo,
    )
    models_results["knn_cosine"] = _compute_metrics(y, knn_oof, post_mr)
    oof_scores["knn_cosine"] = knn_oof

    # DeLong tests: each model vs merge_rate_only AND vs ge_v2_proxy
    delong_results: dict[str, Any] = {}
    p_values_for_correction: dict[str, float] = {}

    compare_models = ["ge_v2_proxy", "lr_full", "lr_full_no_mr", "knn_cosine"]
    baselines = ["merge_rate_only", "ge_v2_proxy"]

    for model_name in compare_models:
        for baseline_name in baselines:
            if model_name == baseline_name:
                continue
            test_key = f"{model_name}_vs_{baseline_name}"
            alt = oof_scores[model_name]
            base = oof_scores[baseline_name]
            if np.all(np.isfinite(alt)) and np.all(np.isfinite(base)):
                with contextlib.suppress(ValueError):
                    dl = delong_auc_test(y, alt, base)
                    delong_results[test_key] = dl
                    p_values_for_correction[test_key] = dl["p_value"]

    # Also test merge_rate_only vs ge_v2_proxy
    mr_base = oof_scores["merge_rate_only"]
    ge_base = oof_scores["ge_v2_proxy"]
    if np.all(np.isfinite(mr_base)) and np.all(np.isfinite(ge_base)):
        with contextlib.suppress(ValueError):
            dl = delong_auc_test(y, mr_base, ge_base)
            delong_results["merge_rate_only_vs_ge_v2_proxy"] = dl
            p_values_for_correction["merge_rate_only_vs_ge_v2_proxy"] = dl["p_value"]

    if p_values_for_correction:
        corrected = holm_bonferroni(p_values_for_correction)
        for test_key, correction in corrected.items():
            delong_results[test_key]["adjusted_p"] = correction["adjusted_p"]
            delong_results[test_key]["reject_h0"] = correction["reject"]

    return {
        "n_authors": n_total,
        "n_high_merge": n_pos,
        "cv_strategy": "loo" if use_loo else f"{n_folds}-fold",
        "models": models_results,
        "delong_tests": delong_results,
    }


def _run_knn_merge_cv(
    features: np.ndarray,
    y: np.ndarray,
    k: int,
    metric: str,
    n_folds: int = 5,
    seed: int = 42,
    use_loo: bool = False,
) -> np.ndarray:
    """k-NN CV for merge prediction. Standard stratified splitting."""
    n = len(y)
    oof_scores = np.full(n, np.nan)

    if use_loo:
        for i in range(n):
            train_mask = np.ones(n, dtype=bool)
            train_mask[i] = False
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(features[train_mask])
            test_scaled = scaler.transform(features[~train_mask])

            # Use high-merge authors as "seeds"
            train_y = y[train_mask]
            pos_idx = np.where(train_y == 1)[0]
            if len(pos_idx) == 0:
                continue
            seed_features = train_scaled[pos_idx]
            scores = _knn_score(seed_features, test_scaled, k, metric)
            oof_scores[i] = scores[0]
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(np.zeros(n), y):
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(features[train_idx])
            test_scaled = scaler.transform(features[test_idx])

            train_y = y[train_idx]
            pos_idx = np.where(train_y == 1)[0]
            if len(pos_idx) == 0:
                continue
            seed_features = train_scaled[pos_idx]
            scores = _knn_score(seed_features, test_scaled, k, metric)
            oof_scores[test_idx] = scores

    return oof_scores


# ---------------------------------------------------------------------------
# Experiment C: k-NN bot proximity as feature
# ---------------------------------------------------------------------------

def _run_experiment_c(
    merge_df: pd.DataFrame,
    full_df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Experiment C: add bot_proximity_score to ge_v2_proxy."""
    y = (merge_df["post_mr"] >= 0.5).astype(int).values
    post_mr = merge_df["post_mr"].values.astype(float)
    n_pos = int(y.sum())
    use_loo = n_pos < 30

    # Get suspended accounts from full parquet as seeds
    susp_df = full_df[full_df["account_status"] == "suspended"].copy()
    n_seeds = len(susp_df)

    if n_seeds == 0:
        return {
            "n_authors": len(merge_df),
            "n_seeds": 0,
            "note": "no suspended accounts for bot proximity",
        }

    # Scale features on full dataset, compute bot proximity for merge authors
    all_features_raw = _prepare_features(full_df, FEATURES_NO_MR)
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features_raw)

    susp_mask = full_df["account_status"] == "suspended"
    seed_features = all_features_scaled[susp_mask.values]

    # Map merge_df logins to full_df indices for feature lookup
    merge_features_raw = _prepare_features(merge_df, FEATURES_NO_MR)
    merge_features_scaled = scaler.transform(merge_features_raw)

    bot_prox = _knn_score(seed_features, merge_features_scaled, k=5, metric="cosine")

    # ge_v2_proxy baseline
    ge_features = _prepare_features(merge_df, ["hub_score", "merge_rate"])
    ge_oof = _run_cv_single(ge_features, y, n_folds=n_folds, seed=seed, use_loo=use_loo)
    ge_metrics = _compute_metrics(y, ge_oof, post_mr)

    # ge_v2_proxy + bot_proximity_score
    ge_plus_bot = np.column_stack([
        _prepare_features(merge_df, ["hub_score", "merge_rate"]),
        bot_prox.reshape(-1, 1),
    ])
    ge_plus_oof = _run_cv_single(
        ge_plus_bot, y, n_folds=n_folds, seed=seed, use_loo=use_loo,
    )
    ge_plus_metrics = _compute_metrics(y, ge_plus_oof, post_mr)

    # DeLong test
    delong_result: dict[str, Any] = {}
    if np.all(np.isfinite(ge_oof)) and np.all(np.isfinite(ge_plus_oof)):
        with contextlib.suppress(ValueError):
            delong_result = delong_auc_test(y, ge_plus_oof, ge_oof)

    return {
        "n_authors": len(merge_df),
        "n_seeds": n_seeds,
        "n_high_merge": n_pos,
        "cv_strategy": "loo" if use_loo else f"{n_folds}-fold",
        "ge_v2_proxy": ge_metrics,
        "ge_v2_proxy_plus_bot_prox": ge_plus_metrics,
        "delong_test": delong_result,
    }


# ---------------------------------------------------------------------------
# Experiment D: Temporal windowing
# ---------------------------------------------------------------------------

def _run_experiment_d(
    merge_df: pd.DataFrame,
    windowed_df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Experiment D: compare windowed merge rate variants."""
    joined = merge_df.merge(windowed_df, on="login", how="left")
    y = (joined["post_mr"] >= 0.5).astype(int).values
    post_mr = joined["post_mr"].values.astype(float)
    n_pos = int(y.sum())

    variants_results: dict[str, Any] = {}

    for variant in WINDOW_VARIANTS:
        if variant not in joined.columns:
            continue

        scores_raw = joined[variant].fillna(0).values.astype(float)

        # Population filter: authors with >=2 PRs in the window
        n_col = f"n_{variant.replace('mr_', '')}" if variant != "mr_weighted" else None
        if n_col and n_col in joined.columns:
            mask = joined[n_col].fillna(0).values >= 2
        else:
            mask = np.ones(len(joined), dtype=bool)

        n_eligible = int(mask.sum())
        if n_eligible < 10:
            variants_results[variant] = {"n_eligible": n_eligible, "note": "too few authors"}
            continue

        y_sub = y[mask]
        post_mr_sub = post_mr[mask]
        scores_sub = scores_raw[mask]
        n_pos_sub = int(y_sub.sum())
        use_loo_sub = n_pos_sub < 30

        # Univariate: AUC + Spearman using raw window merge rate as score
        uni_metrics = _compute_metrics(y_sub, scores_sub, post_mr_sub)

        # v2-style LR: LR(hub_score, mr_variant)
        hub_sub = joined.loc[mask, "hub_score"].fillna(0).values.astype(float)
        if "hub_score" in LOG_TRANSFORM:
            hub_sub = np.log1p(np.abs(hub_sub)) * np.sign(hub_sub)
        v2_features = np.column_stack([hub_sub, scores_sub])
        v2_oof = _run_cv_single(
            v2_features, y_sub, n_folds=n_folds, seed=seed, use_loo=use_loo_sub,
        )
        v2_metrics = _compute_metrics(y_sub, v2_oof, post_mr_sub)

        variants_results[variant] = {
            "n_eligible": n_eligible,
            "n_high_merge": n_pos_sub,
            "cv_strategy": "loo" if use_loo_sub else f"{n_folds}-fold",
            "univariate": uni_metrics,
            "v2_style_lr": v2_metrics,
        }

    return {
        "n_authors": len(joined),
        "n_high_merge": n_pos,
        "variants": variants_results,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_experiment_a(per_cutoff: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean/std across cutoffs for each model."""
    if not per_cutoff:
        return {}

    model_names = ["merge_rate_only", "ge_v2_proxy", "lr_full", "lr_full_no_mr", "knn_cosine"]
    per_model: dict[str, dict[str, Any]] = {}

    for model_name in model_names:
        aucs: list[float] = []
        rhos: list[float] = []
        for r in per_cutoff:
            m = r.get("models", {}).get(model_name, {})
            val = m.get("auc_roc")
            if val is not None and not np.isnan(val):
                aucs.append(val)
            rho = m.get("spearman_rho")
            if rho is not None and not np.isnan(rho):
                rhos.append(rho)

        agg: dict[str, Any] = {}
        if aucs:
            arr = np.array(aucs)
            agg["mean_auc_roc"] = float(np.mean(arr))
            agg["std_auc_roc"] = float(np.std(arr))
            agg["n_cutoffs"] = len(aucs)
        if rhos:
            arr = np.array(rhos)
            agg["mean_spearman_rho"] = float(np.mean(arr))
            agg["std_spearman_rho"] = float(np.std(arr))
        per_model[model_name] = agg

    return {"per_model": per_model}


def _aggregate_experiment_c(per_cutoff: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean/std of AUC improvement from bot proximity."""
    if not per_cutoff:
        return {}

    base_aucs: list[float] = []
    plus_aucs: list[float] = []

    for r in per_cutoff:
        base = r.get("ge_v2_proxy", {}).get("auc_roc")
        plus = r.get("ge_v2_proxy_plus_bot_prox", {}).get("auc_roc")
        if base is not None and plus is not None and not np.isnan(base) and not np.isnan(plus):
            base_aucs.append(base)
            plus_aucs.append(plus)

    if not base_aucs:
        return {}

    base_arr = np.array(base_aucs)
    plus_arr = np.array(plus_aucs)
    delta = plus_arr - base_arr

    return {
        "mean_auc_base": float(np.mean(base_arr)),
        "mean_auc_plus_bot_prox": float(np.mean(plus_arr)),
        "mean_delta_auc": float(np.mean(delta)),
        "std_delta_auc": float(np.std(delta)),
        "n_cutoffs": len(base_aucs),
    }


def _aggregate_experiment_d(per_cutoff: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean/std AUC per window variant."""
    if not per_cutoff:
        return {}

    per_variant: dict[str, dict[str, Any]] = {}

    for variant in WINDOW_VARIANTS:
        uni_aucs: list[float] = []
        v2_aucs: list[float] = []
        rhos: list[float] = []

        for r in per_cutoff:
            v = r.get("variants", {}).get(variant, {})
            uni = v.get("univariate", {})
            v2 = v.get("v2_style_lr", {})

            val = uni.get("auc_roc")
            if val is not None and not np.isnan(val):
                uni_aucs.append(val)
            val = v2.get("auc_roc")
            if val is not None and not np.isnan(val):
                v2_aucs.append(val)
            rho = uni.get("spearman_rho")
            if rho is not None and not np.isnan(rho):
                rhos.append(rho)

        agg: dict[str, Any] = {}
        if uni_aucs:
            arr = np.array(uni_aucs)
            agg["mean_univariate_auc"] = float(np.mean(arr))
            agg["std_univariate_auc"] = float(np.std(arr))
        if v2_aucs:
            arr = np.array(v2_aucs)
            agg["mean_v2_auc"] = float(np.mean(arr))
            agg["std_v2_auc"] = float(np.std(arr))
        if rhos:
            arr = np.array(rhos)
            agg["mean_spearman_rho"] = float(np.mean(arr))
            agg["std_spearman_rho"] = float(np.std(arr))
        agg["n_cutoffs"] = max(len(uni_aucs), len(v2_aucs))
        per_variant[variant] = agg

    return {"per_variant": per_variant}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(
    agg_a: dict[str, Any],
    agg_c: dict[str, Any],
    agg_d: dict[str, Any],
) -> None:
    """Print summary tables."""
    # Experiment A
    per_model = agg_a.get("per_model", {})
    logger.info("\n=== Experiment A: Merge Prediction ===")
    logger.info("%-20s  %8s  %8s  %8s", "Model", "Mean AUC", "Std AUC", "Mean Rho")
    logger.info("-" * 55)
    for m, agg in per_model.items():
        logger.info(
            "%-20s  %8.3f  %8.3f  %8.3f",
            m,
            agg.get("mean_auc_roc", float("nan")),
            agg.get("std_auc_roc", float("nan")),
            agg.get("mean_spearman_rho", float("nan")),
        )

    # Experiment C
    logger.info("\n=== Experiment C: Bot Proximity Feature ===")
    logger.info(
        "Base AUC: %.3f  +BotProx AUC: %.3f  Delta: %.3f (std %.3f)",
        agg_c.get("mean_auc_base", float("nan")),
        agg_c.get("mean_auc_plus_bot_prox", float("nan")),
        agg_c.get("mean_delta_auc", float("nan")),
        agg_c.get("std_delta_auc", float("nan")),
    )

    # Experiment D
    per_variant = agg_d.get("per_variant", {})
    logger.info("\n=== Experiment D: Temporal Windowing ===")
    logger.info("%-15s  %8s  %8s  %8s", "Variant", "Uni AUC", "V2 AUC", "Rho")
    logger.info("-" * 45)
    for v, agg in per_variant.items():
        logger.info(
            "%-15s  %8.3f  %8.3f  %8.3f",
            v,
            agg.get("mean_univariate_auc", float("nan")),
            agg.get("mean_v2_auc", float("nan")),
            agg.get("mean_spearman_rho", float("nan")),
        )
