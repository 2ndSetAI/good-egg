from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.bot_detection.cache import BotDetectionDB
from experiments.bot_detection.checkpoint import write_json
from experiments.bot_detection.models import StudyConfig

logger = logging.getLogger(__name__)


def _parse_cutoff(date_str: str) -> datetime:
    """Parse a YYYY-MM-DD string into a naive datetime."""
    return datetime.fromisoformat(date_str)


def _run_single_cutoff(
    base_dir: Path,
    config: StudyConfig,
    cutoff: datetime,
    knn_exclude_features: list[str],
) -> dict[str, Any]:
    """Run the full author pipeline for a single temporal cutoff.

    Returns per-cutoff metadata and evaluation results.
    """
    from experiments.bot_detection.stages.stage5_author_features import run_stage5
    from experiments.bot_detection.stages.stage6_llm_content import run_stage6_llm_content
    from experiments.bot_detection.stages.stage6_semi_supervised import (
        run_stage6_semi_supervised,
    )
    from experiments.bot_detection.stages.stage6_time_series import run_stage6_time_series
    from experiments.bot_detection.stages.stage6_title_analysis import (
        run_stage6_title_analysis,
    )
    from experiments.bot_detection.stages.stage7_author_evaluate import run_stage7

    cutoff_label = cutoff.strftime("%Y-%m-%d")
    output_dir = base_dir / "data" / "temporal_holdout" / f"T_{cutoff_label}"
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / "author_features.parquet"
    eval_output = output_dir / "author_evaluation.json"

    logger.info("=" * 60)
    logger.info("Temporal holdout: cutoff = %s", cutoff_label)
    logger.info("=" * 60)

    # Stage 5: Author features (pre-cutoff)
    logger.info("--- Stage 5: Author features (cutoff=%s) ---", cutoff_label)
    run_stage5(base_dir, config, cutoff=cutoff, output_dir=output_dir)

    if not parquet_path.exists():
        logger.error("Stage 5 did not produce %s, skipping cutoff", parquet_path)
        return {"cutoff": cutoff_label, "error": "stage5 produced no output"}

    # Collect metadata
    df = pd.read_parquet(parquet_path)
    db_path = base_dir / config.paths.get("local_db", "data/bot_detection.duckdb")
    with BotDetectionDB(db_path) as db:
        n_prs = db.con.execute(
            "SELECT COUNT(*) FROM prs WHERE created_at < ?", [cutoff],
        ).fetchone()[0]

    metadata: dict[str, Any] = {
        "cutoff": cutoff_label,
        "n_authors": len(df),
        "n_prs_before_cutoff": int(n_prs),
    }

    # Stage 6a: Time series (pre-cutoff)
    logger.info("--- Stage 6a: Time series (cutoff=%s) ---", cutoff_label)
    run_stage6_time_series(base_dir, config, cutoff=cutoff, parquet_path=parquet_path)

    # Stage 6d: Title analysis (pre-cutoff)
    logger.info("--- Stage 6d: Title analysis (cutoff=%s) ---", cutoff_label)
    run_stage6_title_analysis(base_dir, config, cutoff=cutoff, parquet_path=parquet_path)

    # Stage 6b: LLM content (pre-cutoff)
    logger.info("--- Stage 6b: LLM content (cutoff=%s) ---", cutoff_label)
    run_stage6_llm_content(base_dir, config, cutoff=cutoff, parquet_path=parquet_path)

    # Stage 6c: Semi-supervised -- two variants
    # H13-clean: exclude outcome-derived features
    logger.info(
        "--- Stage 6c: Semi-supervised H13-clean (cutoff=%s) ---", cutoff_label,
    )
    run_stage6_semi_supervised(
        base_dir, config,
        cutoff=cutoff,
        parquet_path=parquet_path,
        exclude_features=knn_exclude_features,
    )

    # H13-temporal: keep pre-cutoff merge_rate (already in the parquet from stage5)
    # Run again with a different column name suffix to distinguish
    logger.info(
        "--- Stage 6c: Semi-supervised H13-temporal (cutoff=%s) ---", cutoff_label,
    )
    _run_knn_temporal_variant(base_dir, config, parquet_path)

    # Prepare parquet for stage7: restore knn_distance_to_seed from clean variant
    df_pre_eval = pd.read_parquet(parquet_path)
    if "knn_distance_to_seed_clean" in df_pre_eval.columns:
        df_pre_eval["knn_distance_to_seed"] = df_pre_eval["knn_distance_to_seed_clean"]
        df_pre_eval.to_parquet(parquet_path, index=False)

    # Stage 7: Evaluate (primary target only)
    logger.info("--- Stage 7: Evaluation (cutoff=%s) ---", cutoff_label)
    eval_results = run_stage7(
        base_dir, config,
        parquet_path=parquet_path,
        output_path=eval_output,
        skip_auxiliary=True,
    )

    # Evaluate H13-temporal variant separately
    _add_temporal_knn_evaluation(df_pre_eval, eval_results, config)

    # Augment metadata
    df_final = pd.read_parquet(parquet_path)
    if "account_status" in df_final.columns:
        has_status = df_final["account_status"].notna()
        n_evaluable = int(has_status.sum())
        n_suspended = int((df_final["account_status"] == "suspended").sum())
    else:
        n_evaluable = 0
        n_suspended = 0

    metadata["n_authors_evaluable"] = n_evaluable
    metadata["n_suspended_evaluable"] = n_suspended
    metadata["n_seeds"] = n_suspended  # seeds = suspended accounts
    metadata["graph_n_edges"] = int(
        df_final["author_projection_degree"].notna().sum()
    ) if "author_projection_degree" in df_final.columns else 0

    return {
        "metadata": metadata,
        "evaluation": eval_results,
    }


def _add_temporal_knn_evaluation(
    df: pd.DataFrame,
    eval_results: dict[str, Any],
    config: StudyConfig,
) -> None:
    """Evaluate the H13-temporal k-NN variant and add to results."""
    if "knn_distance_to_seed_temporal" not in df.columns:
        return
    if "account_status" not in df.columns:
        return

    from experiments.bot_detection.stages.stage7_author_evaluate import (
        _evaluate_single_score,
    )

    k_values = config.author_analysis.get("precision_at_k", [10, 25, 50, 100, 250])

    y = (df["account_status"] == "suspended").astype(int).values
    raw = df["knn_distance_to_seed_temporal"].values.astype(float)
    scores = 1 / (1 + raw)  # Same transform as H13_knn

    result = _evaluate_single_score(y, scores, k_values)
    result["description"] = "1 / (1 + knn_distance) with pre-cutoff merge_rate in features"

    primary = eval_results.get("primary_results", {})
    primary["H13_knn_temporal"] = result


def _run_knn_temporal_variant(
    base_dir: Path,
    config: StudyConfig,
    parquet_path: Path,
) -> None:
    """Run k-NN with all features (including pre-cutoff merge_rate).

    Saves result as knn_distance_to_seed_temporal alongside the existing
    knn_distance_to_seed (which was computed with excluded features).
    """
    from experiments.bot_detection.stages.stage6_semi_supervised import (
        FEATURE_COLS,
        compute_knn_distances,
    )

    knn_k = config.author_analysis.get("knn_k", 5)
    df = pd.read_parquet(parquet_path)

    # Rename the clean variant before computing temporal
    if "knn_distance_to_seed" in df.columns:
        df = df.rename(columns={"knn_distance_to_seed": "knn_distance_to_seed_clean"})

    # Build feature matrix with all available FEATURE_COLS
    available = [c for c in FEATURE_COLS if c in df.columns]
    raw = df[available].copy()
    for col in available:
        median = raw[col].median()
        if pd.isna(median):
            median = 0.0
        raw[col] = raw[col].fillna(median)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    feat_matrix = scaler.fit_transform(raw.values)

    seed_mask = (df["account_status"] == "suspended").values
    n_seeds = int(seed_mask.sum())

    if n_seeds > 0:
        knn_distances = compute_knn_distances(feat_matrix, seed_mask, k=knn_k)
        df["knn_distance_to_seed_temporal"] = knn_distances
    else:
        df["knn_distance_to_seed_temporal"] = float("nan")

    df.to_parquet(parquet_path, index=False)
    logger.info(
        "Wrote knn_distance_to_seed_temporal (%d seeds, %d features)",
        n_seeds, len(available),
    )


def _aggregate_results(
    per_cutoff: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate evaluation results across cutoffs.

    Extracts per-hypothesis AUC and computes mean, std, min, max.
    """
    # Collect per-hypothesis AUC values across cutoffs
    hypothesis_aucs: dict[str, list[float]] = {}
    cutoff_summaries: list[dict[str, Any]] = []

    for result in per_cutoff:
        metadata = result.get("metadata", {})
        evaluation = result.get("evaluation", {})
        primary = evaluation.get("primary_results", {})

        summary: dict[str, Any] = {
            "cutoff": metadata.get("cutoff", "unknown"),
            "n_authors_evaluable": metadata.get("n_authors_evaluable", 0),
            "n_suspended_evaluable": metadata.get("n_suspended_evaluable", 0),
            "n_prs_before_cutoff": metadata.get("n_prs_before_cutoff", 0),
            "n_seeds": metadata.get("n_seeds", 0),
            "aucs": {},
        }

        for h_name, h_result in primary.items():
            if isinstance(h_result, dict) and "auc_roc" in h_result:
                auc_info = h_result["auc_roc"]
                auc_val = auc_info.get("auc") if isinstance(auc_info, dict) else None
                if auc_val is not None and not (isinstance(auc_val, float) and np.isnan(auc_val)):
                    hypothesis_aucs.setdefault(h_name, []).append(float(auc_val))
                    summary["aucs"][h_name] = float(auc_val)

        cutoff_summaries.append(summary)

    # Aggregate
    aggregated: dict[str, Any] = {}
    for h_name, aucs in hypothesis_aucs.items():
        arr = np.array(aucs)
        aggregated[h_name] = {
            "mean_auc": float(np.mean(arr)),
            "std_auc": float(np.std(arr)),
            "min_auc": float(np.min(arr)),
            "max_auc": float(np.max(arr)),
            "n_cutoffs": len(aucs),
            "per_cutoff": aucs,
        }

    return {
        "per_cutoff": cutoff_summaries,
        "aggregated": aggregated,
    }


def run_temporal_holdout(base_dir: Path, config: StudyConfig) -> None:
    """Run the temporal holdout experiment across all configured cutoffs."""
    holdout_config = config.author_analysis.get("temporal_holdout", {})
    cutoff_strs = holdout_config.get("cutoffs", [
        "2020-01-01", "2021-01-01", "2022-01-01",
        "2022-07-01", "2023-01-01", "2024-01-01",
    ])
    knn_exclude_features = holdout_config.get("knn_exclude_features", [
        "merge_rate", "rejection_rate", "pocket_veto_rate",
    ])

    cutoffs = [_parse_cutoff(s) for s in cutoff_strs]
    logger.info("Temporal holdout: %d cutoffs configured", len(cutoffs))

    per_cutoff_results: list[dict[str, Any]] = []

    for cutoff in cutoffs:
        result = _run_single_cutoff(base_dir, config, cutoff, knn_exclude_features)
        per_cutoff_results.append(result)

    # Aggregate across cutoffs
    aggregated = _aggregate_results(per_cutoff_results)

    output_dir = base_dir / "data" / "temporal_holdout"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "aggregated_results.json"
    write_json(output_path, aggregated)
    logger.info("Aggregated temporal holdout results written to %s", output_path)

    # Print summary table
    logger.info("\n=== Temporal Holdout Summary ===")
    for h_name, agg in aggregated.get("aggregated", {}).items():
        logger.info(
            "  %s: AUC %.3f +/- %.3f (min=%.3f, max=%.3f, n=%d cutoffs)",
            h_name,
            agg["mean_auc"],
            agg["std_auc"],
            agg["min_auc"],
            agg["max_auc"],
            agg["n_cutoffs"],
        )
