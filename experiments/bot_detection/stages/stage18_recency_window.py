from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.bot_detection.checkpoint import write_json
from experiments.bot_detection.models import StudyConfig
from experiments.bot_detection.stats import delong_auc_test, holm_bonferroni

logger = logging.getLogger(__name__)

DEFAULT_CUTOFFS = [
    "2020-01-01", "2021-01-01", "2022-01-01",
    "2022-07-01", "2023-01-01", "2024-01-01",
]

# Merge rate variants to compare (all univariate -- raw score as predictor)
MR_VARIANTS = [
    "mr_alltime", "mr_2yr", "mr_1yr", "mr_6mo", "mr_3mo", "mr_weighted",
]

DELONG_COMPARISONS = [
    ("mr_3mo", "mr_alltime"),
    ("mr_6mo", "mr_alltime"),
    ("mr_1yr", "mr_alltime"),
    ("mr_2yr", "mr_alltime"),
    ("mr_weighted", "mr_alltime"),
    ("mr_3mo", "mr_6mo"),
    ("mr_6mo", "mr_1yr"),
]

# Minimum PRs in window to use that window's MR (otherwise fall back)
MIN_PRS_IN_WINDOW = 2


def run_stage18(
    base_dir: Path,
    config: StudyConfig,
    cutoffs: list[str] | None = None,
) -> dict[str, Any]:
    """Run recency window experiment on unknown contributors."""
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

        logger.info("=== Recency window: cutoff %s ===", cutoff_str)
        result = evaluate_recency_cutoff(parquet_path, db_path, cutoff_str)
        result["cutoff"] = cutoff_str
        per_cutoff.append(result)

    aggregated = _aggregate_results(per_cutoff)
    output = {"per_cutoff": per_cutoff, "aggregated": aggregated}

    output_path = (
        base_dir / "data" / "temporal_holdout"
        / "recency_window_experiment.json"
    )
    write_json(output_path, output)
    logger.info("Results written to %s", output_path)

    _print_summary(aggregated)
    return output


def evaluate_recency_cutoff(
    parquet_path: Path,
    db_path: Path,
    cutoff_str: str,
) -> dict[str, Any]:
    """Run recency window comparison for one cutoff.

    Population: authors with zero merged PRs in repo R before cutoff
    (the GE scoring population), in repos with >=100 pre-cutoff PRs.
    All merge rate variants are univariate (raw score as predictor).
    """
    # Get windowed global merge rates
    windowed_df = _query_windowed_merge_rates(db_path, cutoff_str)
    if windowed_df.empty:
        return {"n_pairs": 0, "note": "no windowed MR data"}

    # Get unknown author-repo pairs
    repo_pre = _query_has_prior_merged(db_path, cutoff_str)
    repo_post = _query_post_cutoff_targets(db_path, cutoff_str)
    repo_sizes = _query_repo_sizes(db_path, cutoff_str)

    if repo_post.empty:
        return {"n_pairs": 0, "note": "no post-cutoff data"}

    dataset = _build_unknown_dataset(
        parquet_path, windowed_df, repo_pre, repo_post, repo_sizes,
    )
    if dataset.empty:
        return {"n_pairs": 0, "note": "no qualifying pairs"}

    # Also build fallback variants: use windowed MR if enough PRs,
    # else fall back to alltime
    for window in ["3mo", "6mo", "1yr", "2yr"]:
        mr_col = f"mr_{window}"
        n_col = f"n_{window}"
        fb_col = f"mr_{window}_fb"
        if mr_col in dataset.columns and n_col in dataset.columns:
            has_enough = dataset[n_col].fillna(0) >= MIN_PRS_IN_WINDOW
            dataset[fb_col] = np.where(
                has_enough,
                dataset[mr_col].fillna(0),
                dataset["mr_alltime"].fillna(0),
            )

    # Evaluate on all medium+ repos pooled
    all_result = _evaluate_variants(dataset, "all_medium_plus")

    # Per-tier
    tier_results: dict[str, Any] = {}
    tiers = {"medium": (100, 499), "large": (500, 1999), "xl": (2000, None)}
    for tier_name, (lo, hi) in tiers.items():
        mask = dataset["repo_prs"] >= lo
        if hi is not None:
            mask = mask & (dataset["repo_prs"] <= hi)
        tier_df = dataset[mask].copy()
        tier_results[tier_name] = _evaluate_variants(tier_df, tier_name)

    return {
        "n_pairs": len(dataset),
        "all": all_result,
        "tiers": tier_results,
    }


def _evaluate_variants(
    dataset: pd.DataFrame,
    label: str,
) -> dict[str, Any]:
    """Compare MR variants as univariate predictors."""
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

    n_repos = dataset["repo"].nunique()

    logger.info(
        "  [%s] %d pairs (%d pos, %d neg), %d repos",
        label, len(y), n_pos, n_neg, n_repos,
    )

    # Evaluate each variant (univariate: raw MR as score)
    models_results: dict[str, Any] = {}
    scores_map: dict[str, np.ndarray] = {}

    # Standard variants
    for variant in MR_VARIANTS:
        if variant not in dataset.columns:
            continue
        scores = dataset[variant].fillna(0).values.astype(float)
        # Count eligible (have data in this window)
        n_col = f"n_{variant.replace('mr_', '')}"
        if n_col in dataset.columns:
            n_eligible = int(
                (dataset[n_col].fillna(0) >= MIN_PRS_IN_WINDOW).sum(),
            )
        else:
            n_eligible = len(dataset)

        metrics = _compute_metrics(y, scores)
        metrics["n_eligible"] = n_eligible
        metrics["pct_eligible"] = (
            float(n_eligible / len(dataset)) if len(dataset) > 0 else 0.0
        )
        models_results[variant] = metrics
        scores_map[variant] = scores

    # Fallback variants (window MR if enough PRs, else alltime)
    for window in ["3mo", "6mo", "1yr", "2yr"]:
        fb_col = f"mr_{window}_fb"
        if fb_col in dataset.columns:
            scores = dataset[fb_col].values.astype(float)
            metrics = _compute_metrics(y, scores)
            metrics["note"] = (
                f"{window} MR if >= {MIN_PRS_IN_WINDOW} PRs, else alltime"
            )
            models_results[fb_col] = metrics
            scores_map[fb_col] = scores

    # DeLong tests
    delong_results: dict[str, Any] = {}
    p_values_for_correction: dict[str, float] = {}

    # Standard comparisons
    for alt_name, base_name in DELONG_COMPARISONS:
        if alt_name not in scores_map or base_name not in scores_map:
            continue
        test_key = f"{alt_name}_vs_{base_name}"
        alt = scores_map[alt_name]
        base = scores_map[base_name]
        if np.all(np.isfinite(alt)) and np.all(np.isfinite(base)):
            with contextlib.suppress(ValueError):
                dl = delong_auc_test(y, alt, base)
                delong_results[test_key] = dl
                p_values_for_correction[test_key] = dl["p_value"]

    # Also test fallback variants vs alltime
    for window in ["3mo", "6mo"]:
        fb_col = f"mr_{window}_fb"
        if fb_col in scores_map and "mr_alltime" in scores_map:
            test_key = f"{fb_col}_vs_mr_alltime"
            alt = scores_map[fb_col]
            base = scores_map["mr_alltime"]
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
        "models": models_results,
        "delong_tests": delong_results,
    }


# ---------------------------------------------------------------------------
# DuckDB queries
# ---------------------------------------------------------------------------

def _query_windowed_merge_rates(
    db_path: Path, cutoff_str: str,
) -> pd.DataFrame:
    """Per-author windowed merge rates before cutoff."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        agg_df = con.execute(
            """
            SELECT
                author AS login,
                AVG(CASE WHEN state='MERGED' THEN 1.0 ELSE 0.0 END)
                    AS mr_alltime,
                COUNT(*) AS n_alltime,
                AVG(CASE WHEN created_at >= ?::TIMESTAMP - INTERVAL '2 years'
                    THEN (CASE WHEN state='MERGED' THEN 1.0 ELSE 0.0 END)
                    ELSE NULL END) AS mr_2yr,
                SUM(CASE WHEN created_at >= ?::TIMESTAMP - INTERVAL '2 years'
                    THEN 1 ELSE 0 END) AS n_2yr,
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
            [cutoff_str] * 9,
        ).fetchdf()

        # Exponentially weighted merge rate (half-life 180 days)
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

    if not pr_df.empty:
        cutoff_ts = pd.Timestamp(cutoff_str)
        pr_df["age_days"] = (
            (cutoff_ts - pr_df["created_at"]).dt.total_seconds() / 86400.0
        )
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


def _query_has_prior_merged(
    db_path: Path, cutoff_str: str,
) -> pd.DataFrame:
    """Pre-cutoff: which author-repo pairs have a merged PR?"""
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


def _query_post_cutoff_targets(
    db_path: Path, cutoff_str: str,
) -> pd.DataFrame:
    """Post-cutoff per-author-per-repo merge rates."""
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
    """Total pre-cutoff PR count per repo."""
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
    windowed_df: pd.DataFrame,
    repo_pre: pd.DataFrame,
    repo_post: pd.DataFrame,
    repo_sizes: pd.DataFrame,
    min_repo_prs: int = 100,
    min_pairs_per_repo: int = 10,
) -> pd.DataFrame:
    """Build dataset of unknown-to-repo authors with windowed MR."""
    qualifying_repos = repo_sizes[repo_sizes["repo_prs"] >= min_repo_prs]

    # Known-merged pairs to exclude
    known_merged = repo_pre[repo_pre["has_prior_merged_in_R"] == 1][
        ["login", "repo"]
    ]

    # Post-cutoff targets
    pairs = repo_post[repo_post["post_prs_in_R"] >= 1].copy()

    # Remove known-merged
    pairs = pairs.merge(
        known_merged, on=["login", "repo"], how="left", indicator=True,
    )
    pairs = pairs[pairs["_merge"] == "left_only"].drop(columns=["_merge"])

    # Filter to qualifying repos
    pairs = pairs.merge(qualifying_repos, on="repo", how="inner")

    # Join windowed merge rates
    dataset = pairs.merge(windowed_df, on="login", how="inner")

    # Filter repos with enough pairs
    repo_counts = dataset.groupby("repo")["login"].nunique()
    good_repos = repo_counts[repo_counts >= min_pairs_per_repo].index
    dataset = dataset[dataset["repo"].isin(good_repos)].copy()

    n_repos = dataset["repo"].nunique()
    logger.info(
        "  Unknown dataset: %d pairs across %d repos",
        len(dataset), n_repos,
    )

    return dataset


def _compute_metrics(
    y: np.ndarray, scores: np.ndarray,
) -> dict[str, Any]:
    """Compute AUC-ROC and AUC-PR."""
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
    """Aggregate across cutoffs."""
    if not per_cutoff:
        return {}

    sections = ["all", "medium", "large", "xl"]
    all_variants = MR_VARIANTS + [
        "mr_3mo_fb", "mr_6mo_fb", "mr_1yr_fb", "mr_2yr_fb",
    ]
    aggregated: dict[str, Any] = {}

    for section in sections:
        per_model: dict[str, dict[str, Any]] = {}

        for variant in all_variants:
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

        # DeLong summary
        delong_summary: dict[str, dict[str, int]] = {}
        all_comparisons = list(DELONG_COMPARISONS) + [
            ("mr_3mo_fb", "mr_alltime"),
            ("mr_6mo_fb", "mr_alltime"),
            ("mr_1yr_fb", "mr_alltime"),
            ("mr_2yr_fb", "mr_alltime"),
        ]
        for alt_name, base_name in all_comparisons:
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
    """Print summary tables."""
    sections = ["all", "medium", "large", "xl"]
    display_variants = [
        "mr_alltime", "mr_2yr", "mr_1yr", "mr_6mo", "mr_3mo", "mr_weighted",
        "mr_3mo_fb", "mr_6mo_fb", "mr_1yr_fb", "mr_2yr_fb",
    ]

    for section in sections:
        data = aggregated.get(section, {})
        per_model = data.get("per_model", {})
        if not per_model:
            continue

        label = section.upper() if section != "all" else "ALL (medium+)"
        logger.info("\n=== Recency Window: %s ===", label)
        logger.info("%-20s  %8s  %8s  %s", "Variant", "Mean ROC", "Std ROC", "N")
        logger.info("-" * 50)
        for variant in display_variants:
            agg = per_model.get(variant, {})
            if not agg:
                continue
            logger.info(
                "%-20s  %8.3f  %8.3f  %d",
                variant,
                agg.get("mean_auc_roc", float("nan")),
                agg.get("std_auc_roc", float("nan")),
                agg.get("n_cutoffs", 0),
            )

        delong_summary = data.get("delong_summary", {})
        if delong_summary:
            logger.info("  DeLong:")
            for test_key, s in delong_summary.items():
                if s["n_tested"] > 0:
                    logger.info(
                        "    %-35s  %d/%d significant",
                        test_key, s["n_significant"], s["n_tested"],
                    )
