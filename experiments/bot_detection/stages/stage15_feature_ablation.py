from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from experiments.bot_detection.checkpoint import write_json
from experiments.bot_detection.models import StudyConfig
from experiments.bot_detection.stats import delong_auc_test, holm_bonferroni

logger = logging.getLogger(__name__)

AVAILABLE_FEATURES = [
    "merge_rate", "total_prs", "career_span_days", "mean_title_length",
    "hub_score", "bipartite_clustering", "isolation_score", "total_repos",
    "median_additions", "median_files_changed",
]

LOG_TRANSFORM = {
    "median_additions", "median_files_changed", "career_span_days", "total_prs",
}

STABLE_CUTOFFS = ["2022-07-01", "2023-01-01", "2024-01-01"]


def run_stage15(
    base_dir: Path,
    config: StudyConfig,
    cutoffs: list[str] | None = None,
) -> dict[str, Any]:
    """Loop 3 stable cutoffs, aggregate, write JSON, print summary."""
    if cutoffs is None:
        cutoffs = STABLE_CUTOFFS

    per_cutoff_results: list[dict[str, Any]] = []

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

        logger.info("=== Feature ablation: cutoff %s ===", cutoff_str)
        result = evaluate_ablation_cutoff(parquet_path)
        result["cutoff"] = cutoff_str
        per_cutoff_results.append(result)

    aggregated = _aggregate_ablation(per_cutoff_results)

    output = {
        "per_cutoff": per_cutoff_results,
        "aggregated": aggregated,
    }

    output_path = (
        base_dir / "data" / "temporal_holdout"
        / "feature_ablation_experiment.json"
    )
    write_json(output_path, output)
    logger.info("Results written to %s", output_path)

    _print_summary(aggregated)
    return output


def evaluate_ablation_cutoff(
    parquet_path: Path,
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Run both LOO ablation and forward selection for one cutoff."""
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

    loo_result = _run_loo_ablation(df, AVAILABLE_FEATURES, n_folds, seed, use_loo)
    fwd_result = _run_forward_selection(
        df, AVAILABLE_FEATURES, n_folds, seed, use_loo,
    )

    return {
        "n_labeled": n_labeled,
        "n_suspended": n_suspended,
        "cv_strategy": "loo" if use_loo else f"{n_folds}-fold",
        "loo_ablation": loo_result,
        "forward_selection": fwd_result,
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


def _get_oof_probs(
    features: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    seed: int,
    use_loo: bool,
) -> np.ndarray:
    """Run CV and return out-of-fold predicted probabilities."""
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
        skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=seed,
        )
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


def _run_loo_ablation(
    df: pd.DataFrame,
    feature_list: list[str],
    n_folds: int,
    seed: int,
    use_loo: bool,
) -> dict[str, Any]:
    """For each feature, run CV without it. DeLong vs full model."""
    y = (df["account_status"] == "suspended").astype(int).values

    # Full model OOF probs
    full_features = _prepare_features(df, feature_list)
    full_probs = _get_oof_probs(full_features, y, n_folds, seed, use_loo)
    full_auc = float(roc_auc_score(y, full_probs))
    logger.info("  Full model AUC: %.4f", full_auc)

    per_feature: dict[str, Any] = {}
    p_values_for_correction: dict[str, float] = {}

    for feat in feature_list:
        reduced_list = [f for f in feature_list if f != feat]
        reduced_features = _prepare_features(df, reduced_list)
        reduced_probs = _get_oof_probs(
            reduced_features, y, n_folds, seed, use_loo,
        )
        reduced_auc = float(roc_auc_score(y, reduced_probs))
        delta = reduced_auc - full_auc

        dl = delong_auc_test(y, reduced_probs, full_probs)

        per_feature[feat] = {
            "reduced_auc": reduced_auc,
            "delta_auc": delta,
            "delong_z": dl["z_statistic"],
            "delong_p": dl["p_value"],
        }
        p_values_for_correction[feat] = dl["p_value"]

        logger.info(
            "    drop %-22s  AUC=%.4f  delta=%+.4f  p=%.4f",
            feat, reduced_auc, delta, dl["p_value"],
        )

    # Holm-Bonferroni correction
    corrected = holm_bonferroni(p_values_for_correction)
    dispensable = []
    non_dispensable = []
    for feat in feature_list:
        adj_p = corrected[feat]["adjusted_p"]
        per_feature[feat]["adjusted_p"] = adj_p
        per_feature[feat]["dispensable"] = adj_p > 0.05
        if adj_p > 0.05:
            dispensable.append(feat)
        else:
            non_dispensable.append(feat)

    return {
        "full_auc": full_auc,
        "per_feature": per_feature,
        "dispensable": dispensable,
        "non_dispensable": non_dispensable,
    }


def _run_forward_selection(
    df: pd.DataFrame,
    candidate_features: list[str],
    n_folds: int,
    seed: int,
    use_loo: bool,
) -> dict[str, Any]:
    """Greedy forward selection with DeLong stopping criterion."""
    y = (df["account_status"] == "suspended").astype(int).values
    remaining = list(candidate_features)
    selected: list[str] = []
    steps: list[dict[str, Any]] = []
    prev_probs: np.ndarray | None = None
    stopped_at: int | None = None

    while remaining:
        best_feat: str | None = None
        best_auc = -1.0
        best_probs: np.ndarray | None = None

        for feat in remaining:
            trial = selected + [feat]
            features = _prepare_features(df, trial)
            probs = _get_oof_probs(features, y, n_folds, seed, use_loo)
            auc = float(roc_auc_score(y, probs))
            if auc > best_auc:
                best_auc = auc
                best_feat = feat
                best_probs = probs

        assert best_feat is not None
        assert best_probs is not None

        # DeLong test: k-feature model vs (k-1)-feature model
        step_info: dict[str, Any] = {
            "feature_added": best_feat,
            "k": len(selected) + 1,
            "auc": best_auc,
        }

        if prev_probs is not None:
            dl = delong_auc_test(y, best_probs, prev_probs)
            step_info["delong_z"] = dl["z_statistic"]
            step_info["delong_p"] = dl["p_value"]

            if dl["p_value"] > 0.05 and stopped_at is None:
                stopped_at = len(selected)
                step_info["stop_trigger"] = True
        else:
            step_info["delong_p"] = 0.0  # first feature always added

        steps.append(step_info)
        selected.append(best_feat)
        remaining.remove(best_feat)
        prev_probs = best_probs

        logger.info(
            "    +%-22s  k=%d  AUC=%.4f  p=%.4f",
            best_feat, step_info["k"], best_auc,
            step_info.get("delong_p", 0.0),
        )

    if stopped_at is None:
        stopped_at = len(selected)

    selected_before_stop = selected[:stopped_at]

    return {
        "order": selected,
        "steps": steps,
        "stopped_at_k": stopped_at,
        "selected_features": selected_before_stop,
    }


def _aggregate_ablation(
    per_cutoff: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate ablation results across cutoffs."""
    if not per_cutoff:
        return {}

    # Aggregate LOO ablation: mean delta per feature
    feature_deltas: dict[str, list[float]] = {}
    feature_dispensable_counts: dict[str, int] = {}
    for r in per_cutoff:
        loo = r.get("loo_ablation", {})
        for feat, info in loo.get("per_feature", {}).items():
            feature_deltas.setdefault(feat, []).append(info["delta_auc"])
            if info.get("dispensable", False):
                feature_dispensable_counts[feat] = (
                    feature_dispensable_counts.get(feat, 0) + 1
                )

    n_cutoffs = len(per_cutoff)
    loo_agg: dict[str, Any] = {}
    for feat in AVAILABLE_FEATURES:
        deltas = feature_deltas.get(feat, [])
        if deltas:
            loo_agg[feat] = {
                "mean_delta": float(np.mean(deltas)),
                "std_delta": float(np.std(deltas)),
                "dispensable_count": feature_dispensable_counts.get(feat, 0),
                "dispensable_in_all": (
                    feature_dispensable_counts.get(feat, 0) == n_cutoffs
                ),
            }

    # Aggregate forward selection: mean order rank per feature
    feature_ranks: dict[str, list[int]] = {}
    feature_selected_counts: dict[str, int] = {}
    for r in per_cutoff:
        fwd = r.get("forward_selection", {})
        order = fwd.get("order", [])
        selected_set = set(fwd.get("selected_features", []))
        for rank, feat in enumerate(order, start=1):
            feature_ranks.setdefault(feat, []).append(rank)
            if feat in selected_set:
                feature_selected_counts[feat] = (
                    feature_selected_counts.get(feat, 0) + 1
                )

    fwd_agg: dict[str, Any] = {}
    for feat in AVAILABLE_FEATURES:
        ranks = feature_ranks.get(feat, [])
        if ranks:
            fwd_agg[feat] = {
                "mean_rank": float(np.mean(ranks)),
                "std_rank": float(np.std(ranks)),
                "selected_count": feature_selected_counts.get(feat, 0),
                "selected_in_all": (
                    feature_selected_counts.get(feat, 0) == n_cutoffs
                ),
            }

    # Recommended features: non-dispensable in ALL cutoffs AND selected
    # before stopping in ALL cutoffs
    recommended = []
    for feat in AVAILABLE_FEATURES:
        loo_info = loo_agg.get(feat, {})
        fwd_info = fwd_agg.get(feat, {})
        non_dispensable = not loo_info.get("dispensable_in_all", True)
        selected_in_all = fwd_info.get("selected_in_all", False)
        if non_dispensable and selected_in_all:
            recommended.append(feat)

    return {
        "loo_ablation": loo_agg,
        "forward_selection": fwd_agg,
        "recommended_features": recommended,
        "n_cutoffs": n_cutoffs,
    }


def _print_summary(aggregated: dict[str, Any]) -> None:
    """Print a summary table to the log."""
    loo_agg = aggregated.get("loo_ablation", {})
    fwd_agg = aggregated.get("forward_selection", {})
    recommended = aggregated.get("recommended_features", [])

    logger.info("\n=== Feature Ablation Summary ===")
    logger.info(
        "%-22s  %10s  %10s  %5s  %9s  %5s",
        "Feature", "Mean Delta", "Std Delta", "Disp?",
        "Mean Rank", "Sel?",
    )
    logger.info("-" * 72)

    for feat in AVAILABLE_FEATURES:
        loo = loo_agg.get(feat, {})
        fwd = fwd_agg.get(feat, {})
        disp = "Y" if loo.get("dispensable_in_all", False) else "N"
        sel = "Y" if fwd.get("selected_in_all", False) else "N"
        logger.info(
            "%-22s  %+10.4f  %10.4f  %5s  %9.1f  %5s",
            feat,
            loo.get("mean_delta", float("nan")),
            loo.get("std_delta", float("nan")),
            disp,
            fwd.get("mean_rank", float("nan")),
            sel,
        )

    logger.info("\nRecommended feature subset (%d features):", len(recommended))
    for feat in recommended:
        logger.info("  - %s", feat)
