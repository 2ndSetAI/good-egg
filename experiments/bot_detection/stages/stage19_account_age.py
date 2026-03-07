from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
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

MODEL_VARIANTS = ["mr_only", "mr_age", "age_only"]

VARIANT_FEATURES: dict[str, list[str]] = {
    "mr_only": ["merge_rate"],
    "mr_age": ["merge_rate", "log_account_age"],
    "age_only": ["log_account_age"],
}

DELONG_COMPARISONS = [
    ("mr_age", "mr_only"),
    ("age_only", "mr_only"),
]

REPO_TIERS: dict[str, tuple[int, int | None]] = {
    "medium": (100, 499),
    "large": (500, 1999),
    "xl": (2000, None),
}


def run_stage19(
    base_dir: Path,
    config: StudyConfig,
    cutoffs: list[str] | None = None,
) -> dict[str, Any]:
    """Run account_age experiment on unknown contributors across cutoffs."""
    if cutoffs is None:
        holdout_config = config.author_analysis.get("temporal_holdout", {})
        cutoffs = holdout_config.get("cutoffs", DEFAULT_CUTOFFS)

    db_path = base_dir / "data" / "bot_detection.duckdb"
    per_cutoff: list[dict[str, Any]] = []

    for cutoff_str in cutoffs:
        parquet_path = (
            base_dir / "data" / "temporal_holdout"
            / f"T_{cutoff_str}" / "author_features.parquet"
        )
        if not parquet_path.exists():
            logger.warning(
                "No parquet at %s, skipping cutoff %s",
                parquet_path, cutoff_str,
            )
            continue

        logger.info("=== Account age unknown contributors: cutoff %s ===", cutoff_str)
        result = evaluate_account_age_cutoff(parquet_path, db_path, cutoff_str)
        result["cutoff"] = cutoff_str
        per_cutoff.append(result)

    aggregated = _aggregate_results(per_cutoff)
    output = {"per_cutoff": per_cutoff, "aggregated": aggregated}

    output_path = (
        base_dir / "data" / "temporal_holdout"
        / "account_age_unknown_experiment.json"
    )
    write_json(output_path, output)
    logger.info("Results written to %s", output_path)

    _print_summary(aggregated)
    return output


def evaluate_account_age_cutoff(
    parquet_path: Path,
    db_path: Path,
    cutoff_str: str,
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Run models on unknown contributors for one cutoff."""
    repo_pre = _query_pre_cutoff_repo_stats(db_path, cutoff_str)
    repo_post = _query_post_cutoff_repo_targets(db_path, cutoff_str)
    repo_sizes = _query_repo_sizes(db_path, cutoff_str)

    if repo_pre.empty or repo_post.empty:
        return {"n_pairs": 0, "note": "no data from DuckDB"}

    dataset = _build_unknown_dataset(
        parquet_path, repo_pre, repo_post, repo_sizes,
    )
    if dataset.empty:
        return {"n_pairs": 0, "note": "no qualifying unknown author-repo pairs"}

    all_result = _evaluate_tier(dataset, "all_medium_plus", n_folds, seed)

    tier_results: dict[str, Any] = {}
    for tier_name, (lo, hi) in REPO_TIERS.items():
        mask = dataset["repo_prs"] >= lo
        if hi is not None:
            mask = mask & (dataset["repo_prs"] <= hi)
        tier_df = dataset[mask].copy()
        tier_results[tier_name] = _evaluate_tier(
            tier_df, tier_name, n_folds, seed,
        )

    return {
        "n_pairs": len(dataset),
        "n_have_age": int(dataset["log_account_age"].notna().sum()),
        "all": all_result,
        "tiers": tier_results,
    }


def _evaluate_tier(
    dataset: pd.DataFrame,
    tier_name: str,
    n_folds: int,
    seed: int,
) -> dict[str, Any]:
    """Run all models + DeLong tests on a dataset slice."""
    # Filter to rows that have account_age_days
    dataset = dataset[dataset["log_account_age"].notna()].copy()

    if len(dataset) < 10:
        return {"n_pairs": len(dataset), "note": "too few pairs"}

    y = (dataset["post_mr_in_R"] >= 0.5).astype(int).values
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos

    if n_pos < 2 or n_neg < 2:
        return {
            "n_pairs": len(dataset),
            "n_positive": n_pos,
            "note": "insufficient class balance",
        }

    use_loo = n_pos < 30
    n_repos = dataset["repo"].nunique()

    logger.info(
        "  [%s] %d pairs (%d pos, %d neg), %d repos, cv=%s",
        tier_name, len(y), n_pos, n_neg, n_repos,
        "loo" if use_loo else f"{n_folds}-fold",
    )

    models_results: dict[str, Any] = {}
    oof_scores: dict[str, np.ndarray] = {}

    for variant in MODEL_VARIANTS:
        features = _prepare_features(dataset, VARIANT_FEATURES[variant])
        if len(VARIANT_FEATURES[variant]) == 1:
            scores = features.ravel()
        else:
            scores = _run_cv_single(
                features, y, n_folds=n_folds, seed=seed, use_loo=use_loo,
            )
        oof_scores[variant] = scores
        models_results[variant] = _compute_metrics(y, scores)

    # DeLong tests
    delong_results: dict[str, Any] = {}
    p_values_for_correction: dict[str, float] = {}

    for alt_name, base_name in DELONG_COMPARISONS:
        test_key = f"{alt_name}_vs_{base_name}"
        alt = oof_scores[alt_name]
        base = oof_scores[base_name]
        if np.all(np.isfinite(alt)) and np.all(np.isfinite(base)):
            with contextlib.suppress(ValueError):
                dl = delong_auc_test(y, alt, base)
                delong_results[test_key] = dl
                p_values_for_correction[test_key] = dl["p_value"]

    if p_values_for_correction:
        corrected = holm_bonferroni(p_values_for_correction)
        for test_key, correction in corrected.items():
            delong_results[test_key]["adjusted_p"] = correction["adjusted_p"]
            delong_results[test_key]["reject_h0"] = correction["reject"]

    return {
        "n_pairs": len(y),
        "n_positive": n_pos,
        "n_repos": n_repos,
        "cv_strategy": "loo" if use_loo else f"{n_folds}-fold",
        "models": models_results,
        "delong_tests": delong_results,
    }


# ---------------------------------------------------------------------------
# DuckDB queries (reused from stage17)
# ---------------------------------------------------------------------------

def _query_pre_cutoff_repo_stats(
    db_path: Path, cutoff_str: str,
) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        result = con.execute(
            """
            SELECT author AS login, repo,
                   MAX(CASE WHEN state='MERGED' THEN 1 ELSE 0 END)
                       AS has_prior_merged_in_R
            FROM prs
            WHERE created_at < ?::TIMESTAMP
            GROUP BY author, repo
            """,
            [cutoff_str],
        ).fetchdf()
    finally:
        con.close()
    return result


def _query_post_cutoff_repo_targets(
    db_path: Path, cutoff_str: str,
) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        result = con.execute(
            """
            SELECT author AS login, repo,
                   AVG(CASE WHEN state='MERGED' THEN 1.0 ELSE 0.0 END)
                       AS post_mr_in_R,
                   COUNT(*) AS post_prs_in_R
            FROM prs
            WHERE created_at >= ?::TIMESTAMP
            GROUP BY author, repo
            """,
            [cutoff_str],
        ).fetchdf()
    finally:
        con.close()
    return result


def _query_repo_sizes(
    db_path: Path, cutoff_str: str,
) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        result = con.execute(
            """
            SELECT repo, COUNT(*) AS repo_prs
            FROM prs
            WHERE created_at < ?::TIMESTAMP
            GROUP BY repo
            """,
            [cutoff_str],
        ).fetchdf()
    finally:
        con.close()
    return result


def _build_unknown_dataset(
    parquet_path: Path,
    repo_pre: pd.DataFrame,
    repo_post: pd.DataFrame,
    repo_sizes: pd.DataFrame,
    min_repo_prs: int = 100,
    min_pairs_per_repo: int = 10,
) -> pd.DataFrame:
    """Build dataset of unknown-to-repo author-repo pairs with account_age."""
    qualifying_repos = repo_sizes[repo_sizes["repo_prs"] >= min_repo_prs]

    known_merged = repo_pre[repo_pre["has_prior_merged_in_R"] == 1][
        ["login", "repo"]
    ]

    pairs = repo_post[repo_post["post_prs_in_R"] >= 1].copy()

    pairs = pairs.merge(
        known_merged, on=["login", "repo"], how="left", indicator=True,
    )
    pairs = pairs[pairs["_merge"] == "left_only"].drop(columns=["_merge"])

    pairs = pairs.merge(qualifying_repos, on="repo", how="inner")

    global_df = pd.read_parquet(parquet_path)
    global_df = global_df[["login", "merge_rate", "account_age_days"]].copy()
    global_df["log_account_age"] = np.log1p(
        global_df["account_age_days"].clip(lower=0),
    )

    dataset = pairs.merge(global_df, on="login", how="inner")

    repo_counts = dataset.groupby("repo")["login"].nunique()
    good_repos = repo_counts[repo_counts >= min_pairs_per_repo].index
    dataset = dataset[dataset["repo"].isin(good_repos)].copy()

    n_repos = dataset["repo"].nunique()
    n_have_age = dataset["log_account_age"].notna().sum()
    logger.info(
        "  Unknown dataset: %d pairs across %d repos (%d have account_age)",
        len(dataset), n_repos, n_have_age,
    )

    return dataset


# ---------------------------------------------------------------------------
# ML helpers
# ---------------------------------------------------------------------------

def _prepare_features(
    df: pd.DataFrame, feature_list: list[str],
) -> np.ndarray:
    arrays = []
    for col in feature_list:
        vals = df[col].fillna(0).values.astype(float)
        arrays.append(vals)
    return np.column_stack(arrays)


def _run_cv_single(
    features: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
    use_loo: bool = False,
) -> np.ndarray:
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
        skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=seed,
        )
        for train_idx, test_idx in skf.split(np.zeros(n), y):
            scaler = StandardScaler()
            x_train = scaler.fit_transform(features[train_idx])
            x_test = scaler.transform(features[test_idx])
            model = LogisticRegression(max_iter=1000, random_state=seed)
            model.fit(x_train, y[train_idx])
            oof_probs[test_idx] = model.predict_proba(x_test)[:, 1]

    return oof_probs


def _compute_metrics(
    y: np.ndarray, scores: np.ndarray,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if y.sum() > 0 and (1 - y).sum() > 0 and np.all(np.isfinite(scores)):
        metrics["auc_roc"] = float(roc_auc_score(y, scores))
        metrics["auc_pr"] = float(average_precision_score(y, scores))
    else:
        metrics["auc_roc"] = float("nan")
        metrics["auc_pr"] = float("nan")
    return metrics


# ---------------------------------------------------------------------------
# Aggregation and summary
# ---------------------------------------------------------------------------

def _aggregate_results(
    per_cutoff: list[dict[str, Any]],
) -> dict[str, Any]:
    if not per_cutoff:
        return {}

    sections = ["all"] + list(REPO_TIERS.keys())
    aggregated: dict[str, Any] = {}

    for section in sections:
        per_model: dict[str, dict[str, Any]] = {}

        for variant in MODEL_VARIANTS:
            aucs: list[float] = []
            for r in per_cutoff:
                src = (
                    r.get("all", {}) if section == "all"
                    else r.get("tiers", {}).get(section, {})
                )
                m = src.get("models", {}).get(variant, {})
                val = m.get("auc_roc")
                if val is not None and not np.isnan(val):
                    aucs.append(val)

            agg: dict[str, Any] = {}
            if aucs:
                arr = np.array(aucs)
                agg["mean_auc_roc"] = float(np.mean(arr))
                agg["std_auc_roc"] = float(np.std(arr))
                agg["n_cutoffs"] = len(aucs)
            per_model[variant] = agg

        delong_summary: dict[str, dict[str, int]] = {}
        for alt_name, base_name in DELONG_COMPARISONS:
            test_key = f"{alt_name}_vs_{base_name}"
            n_sig = 0
            n_tested = 0
            for r in per_cutoff:
                src = (
                    r.get("all", {}) if section == "all"
                    else r.get("tiers", {}).get(section, {})
                )
                dl = src.get("delong_tests", {}).get(test_key, {})
                if "p_value" in dl:
                    n_tested += 1
                    if dl.get("reject_h0", False):
                        n_sig += 1
            delong_summary[test_key] = {
                "n_significant": n_sig, "n_tested": n_tested,
            }

        aggregated[section] = {
            "per_model": per_model,
            "delong_summary": delong_summary,
        }

    return aggregated


def _print_summary(aggregated: dict[str, Any]) -> None:
    sections = ["all"] + list(REPO_TIERS.keys())

    for section in sections:
        data = aggregated.get(section, {})
        per_model = data.get("per_model", {})
        if not per_model:
            continue

        label = section.upper() if section != "all" else "ALL (medium+)"
        logger.info("\n=== Account Age Unknown Contributors: %s ===", label)
        logger.info(
            "%-20s  %8s  %8s  %s",
            "Model", "Mean ROC", "Std ROC", "N",
        )
        logger.info("-" * 50)
        for variant in MODEL_VARIANTS:
            agg = per_model.get(variant, {})
            logger.info(
                "%-20s  %8.3f  %8.3f  %d",
                variant,
                agg.get("mean_auc_roc", float("nan")),
                agg.get("std_auc_roc", float("nan")),
                agg.get("n_cutoffs", 0),
            )

        delong_summary = data.get("delong_summary", {})
        if delong_summary:
            logger.info("  DeLong (Holm-Bonferroni):")
            for test_key, s in delong_summary.items():
                logger.info(
                    "    %-35s  %d/%d significant",
                    test_key, s["n_significant"], s["n_tested"],
                )
