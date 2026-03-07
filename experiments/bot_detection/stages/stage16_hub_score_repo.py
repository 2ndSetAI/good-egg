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

LOG_TRANSFORM = {"n_prior_prs_in_R", "total_repos"}

MODEL_VARIANTS = [
    "mr_only",
    "mr_repo",
    "mr_plus_hub",
    "mr_repo_plus_hub",
    "full_global",
    "full_repo",
    "full_combined",
]

VARIANT_FEATURES: dict[str, list[str]] = {
    "mr_only": ["merge_rate"],
    "mr_repo": ["merge_rate_in_R"],
    "mr_plus_hub": ["merge_rate", "hub_score"],
    "mr_repo_plus_hub": ["merge_rate_in_R", "hub_score"],
    "full_global": ["merge_rate", "hub_score", "total_repos"],
    "full_repo": ["merge_rate_in_R", "has_prior_merged_in_R", "n_prior_prs_in_R"],
    "full_combined": [
        "merge_rate", "hub_score", "total_repos",
        "merge_rate_in_R", "has_prior_merged_in_R", "n_prior_prs_in_R",
    ],
}

DELONG_COMPARISONS = [
    ("mr_plus_hub", "mr_only"),
    ("mr_repo_plus_hub", "mr_repo"),
    ("full_combined", "full_repo"),
    ("full_repo", "mr_only"),
]


def run_stage16(
    base_dir: Path,
    config: StudyConfig,
    cutoffs: list[str] | None = None,
) -> dict[str, Any]:
    """Run hub_score repo-specific experiment across temporal cutoffs."""
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
                "No parquet at %s, skipping cutoff %s", parquet_path, cutoff_str,
            )
            continue

        logger.info("=== Hub score repo-specific: cutoff %s ===", cutoff_str)
        result = evaluate_hub_score_cutoff(parquet_path, db_path, cutoff_str)
        result["cutoff"] = cutoff_str
        per_cutoff.append(result)

    aggregated = _aggregate_results(per_cutoff)

    output = {
        "per_cutoff": per_cutoff,
        "aggregated": aggregated,
    }

    output_path = (
        base_dir / "data" / "temporal_holdout" / "hub_score_repo_experiment.json"
    )
    write_json(output_path, output)
    logger.info("Results written to %s", output_path)

    _print_summary(aggregated)
    return output


def evaluate_hub_score_cutoff(
    parquet_path: Path,
    db_path: Path,
    cutoff_str: str,
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Run all 7 models on pooled author-repo pairs for one cutoff."""
    repo_features = _query_repo_specific_features(db_path, cutoff_str)
    repo_targets = _query_repo_specific_targets(db_path, cutoff_str)

    if repo_features.empty or repo_targets.empty:
        return {"n_pairs": 0, "note": "no data from DuckDB"}

    dataset = _build_author_repo_dataset(
        parquet_path, repo_features, repo_targets,
    )
    if dataset.empty:
        return {"n_pairs": 0, "note": "no qualifying author-repo pairs"}

    y = (dataset["post_mr_in_R"] >= 0.5).astype(int).values
    n_pos = int(y.sum())
    n_total = len(y)
    use_loo = n_pos < 30

    logger.info(
        "  %d author-repo pairs, %d high-merge (>=0.5), cv=%s",
        n_total, n_pos, "loo" if use_loo else f"{n_folds}-fold",
    )

    n_repos = dataset["repo"].nunique()
    logger.info("  %d qualifying repos", n_repos)

    models_results: dict[str, Any] = {}
    oof_scores: dict[str, np.ndarray] = {}

    for variant in MODEL_VARIANTS:
        features = _prepare_features(dataset, VARIANT_FEATURES[variant])
        if len(VARIANT_FEATURES[variant]) == 1:
            # Univariate: use raw feature as score (no CV needed)
            scores = features.ravel()
            oof_scores[variant] = scores
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

    # Per-repo breakdown for repos with >=50 pairs
    per_repo = _per_repo_analysis(dataset, oof_scores, y)

    return {
        "n_pairs": n_total,
        "n_positive": n_pos,
        "n_repos": n_repos,
        "cv_strategy": "loo" if use_loo else f"{n_folds}-fold",
        "models": models_results,
        "delong_tests": delong_results,
        "per_repo": per_repo,
    }


def _query_repo_specific_features(
    db_path: Path, cutoff_str: str,
) -> pd.DataFrame:
    """Pre-cutoff per-author-per-repo stats from DuckDB."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        result = con.execute(
            """
            SELECT author AS login, repo,
                   COUNT(*) AS n_prior_prs_in_R,
                   AVG(CASE WHEN state='MERGED' THEN 1.0 ELSE 0.0 END)
                       AS merge_rate_in_R,
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


def _query_repo_specific_targets(
    db_path: Path, cutoff_str: str,
) -> pd.DataFrame:
    """Post-cutoff per-author-per-repo merge rates from DuckDB."""
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


def _build_author_repo_dataset(
    parquet_path: Path,
    repo_features: pd.DataFrame,
    repo_targets: pd.DataFrame,
    min_authors: int = 20,
) -> pd.DataFrame:
    """Join global features with repo-specific features and targets.

    Filter to repos with >= min_authors author-repo pairs.
    """
    global_df = pd.read_parquet(parquet_path)
    global_cols = ["login", "merge_rate", "hub_score", "total_repos"]
    available = [c for c in global_cols if c in global_df.columns]
    global_df = global_df[available].copy()

    # Join repo features with repo targets on (login, repo)
    pairs = repo_features.merge(repo_targets, on=["login", "repo"], how="inner")

    # Require at least 1 post-cutoff PR in R
    pairs = pairs[pairs["post_prs_in_R"] >= 1].copy()

    # Join global features
    dataset = pairs.merge(global_df, on="login", how="inner")

    # Filter to repos with enough authors
    repo_counts = dataset.groupby("repo")["login"].nunique()
    qualifying_repos = repo_counts[repo_counts >= min_authors].index
    dataset = dataset[dataset["repo"].isin(qualifying_repos)].copy()

    logger.info(
        "  Built dataset: %d pairs across %d repos (min_authors=%d)",
        len(dataset), dataset["repo"].nunique(), min_authors,
    )

    return dataset


def _prepare_features(
    df: pd.DataFrame, feature_list: list[str],
) -> np.ndarray:
    """Extract columns, log-transform count features, fill NaN with 0."""
    arrays = []
    for col in feature_list:
        vals = df[col].fillna(0).values.astype(float)
        if col in LOG_TRANSFORM:
            vals = np.log1p(np.abs(vals)) * np.sign(vals)
        arrays.append(vals)
    return np.column_stack(arrays)


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
    """Compute AUC-ROC and AUC-PR."""
    metrics: dict[str, Any] = {}
    if y.sum() > 0 and (1 - y).sum() > 0 and np.all(np.isfinite(scores)):
        metrics["auc_roc"] = float(roc_auc_score(y, scores))
        metrics["auc_pr"] = float(average_precision_score(y, scores))
    else:
        metrics["auc_roc"] = float("nan")
        metrics["auc_pr"] = float("nan")
    return metrics


def _per_repo_analysis(
    dataset: pd.DataFrame,
    oof_scores: dict[str, np.ndarray],
    y: np.ndarray,
    min_pairs: int = 50,
) -> list[dict[str, Any]]:
    """Per-repo AUC for repos with >= min_pairs author-repo pairs."""
    results: list[dict[str, Any]] = []
    repo_groups = dataset.groupby("repo")

    for repo_name, group in repo_groups:
        if len(group) < min_pairs:
            continue
        idx = group.index
        mask = dataset.index.isin(idx)
        y_repo = y[mask]
        if y_repo.sum() == 0 or (1 - y_repo).sum() == 0:
            continue

        repo_metrics: dict[str, Any] = {"repo": repo_name, "n_pairs": len(group)}
        for variant in MODEL_VARIANTS:
            scores_repo = oof_scores[variant][mask]
            if np.all(np.isfinite(scores_repo)):
                repo_metrics[f"{variant}_auc"] = float(
                    roc_auc_score(y_repo, scores_repo),
                )
        results.append(repo_metrics)

    return results


def _aggregate_results(
    per_cutoff: list[dict[str, Any]],
) -> dict[str, Any]:
    """Mean/std across cutoffs for each model variant."""
    if not per_cutoff:
        return {}

    per_model: dict[str, dict[str, Any]] = {}

    for variant in MODEL_VARIANTS:
        aucs: list[float] = []
        prs: list[float] = []
        for r in per_cutoff:
            m = r.get("models", {}).get(variant, {})
            val = m.get("auc_roc")
            if val is not None and not np.isnan(val):
                aucs.append(val)
            val = m.get("auc_pr")
            if val is not None and not np.isnan(val):
                prs.append(val)

        agg: dict[str, Any] = {}
        if aucs:
            arr = np.array(aucs)
            agg["mean_auc_roc"] = float(np.mean(arr))
            agg["std_auc_roc"] = float(np.std(arr))
            agg["n_cutoffs"] = len(aucs)
        if prs:
            arr = np.array(prs)
            agg["mean_auc_pr"] = float(np.mean(arr))
            agg["std_auc_pr"] = float(np.std(arr))
        per_model[variant] = agg

    # DeLong summary: count significant results across cutoffs
    delong_summary: dict[str, dict[str, Any]] = {}
    for alt_name, base_name in DELONG_COMPARISONS:
        test_key = f"{alt_name}_vs_{base_name}"
        n_sig = 0
        n_tested = 0
        for r in per_cutoff:
            dl = r.get("delong_tests", {}).get(test_key, {})
            if "p_value" in dl:
                n_tested += 1
                if dl.get("reject_h0", False):
                    n_sig += 1
        delong_summary[test_key] = {
            "n_significant": n_sig,
            "n_tested": n_tested,
        }

    return {
        "per_model": per_model,
        "delong_summary": delong_summary,
    }


def _print_summary(aggregated: dict[str, Any]) -> None:
    """Print summary table via logger."""
    per_model = aggregated.get("per_model", {})
    logger.info("\n=== Hub Score Repo-Specific Experiment ===")
    logger.info(
        "%-20s  %8s  %8s  %8s  %8s",
        "Model", "Mean ROC", "Std ROC", "Mean PR", "Std PR",
    )
    logger.info("-" * 62)
    for variant in MODEL_VARIANTS:
        agg = per_model.get(variant, {})
        logger.info(
            "%-20s  %8.3f  %8.3f  %8.3f  %8.3f",
            variant,
            agg.get("mean_auc_roc", float("nan")),
            agg.get("std_auc_roc", float("nan")),
            agg.get("mean_auc_pr", float("nan")),
            agg.get("std_auc_pr", float("nan")),
        )

    delong_summary = aggregated.get("delong_summary", {})
    if delong_summary:
        logger.info("\nDeLong significance (Holm-Bonferroni corrected):")
        for test_key, summary in delong_summary.items():
            logger.info(
                "  %-40s  %d/%d significant",
                test_key,
                summary.get("n_significant", 0),
                summary.get("n_tested", 0),
            )
