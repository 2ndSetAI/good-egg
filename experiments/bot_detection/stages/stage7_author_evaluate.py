from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import average_precision_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from experiments.bot_detection.checkpoint import write_json, write_stage_checkpoint
from experiments.bot_detection.models import StudyConfig
from experiments.bot_detection.stats import auc_roc_with_ci

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hypothesis score definitions
# ---------------------------------------------------------------------------

HYPOTHESIS_SCORES: dict[str, dict[str, Any]] = {
    "H8_merge_rate": {
        "column": "merge_rate",
        "transform": lambda x: 1 - x,
        "description": "1 - merge_rate (low merge rate = suspicious)",
    },
    "H9_temporal": {
        "column": "inter_pr_cv",
        "transform": None,
        "description": "Inter-PR coefficient of variation",
    },
    "H10_network": {
        "column": "hub_score",
        "transform": None,
        "description": "HITS hub score from bipartite graph",
    },
    "H11_llm": {
        "column": "llm_spam_score",
        "transform": None,
        "description": "LLM spam classification score",
    },
    "H13_knn": {
        "column": "knn_distance_to_seed",
        "transform": lambda x: 1 / (1 + x),
        "description": "1 / (1 + knn_distance) to seed bots",
    },
    "H13_if": {
        "column": "isolation_forest_score",
        "transform": lambda x: -x,
        "description": "Negated isolation forest score (more negative = more anomalous)",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Fraction of top-k scored items that are positive."""
    order = np.argsort(-scores)
    top_k = y_true[order[:k]]
    return float(top_k.sum() / k)


def _recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Fraction of positives captured in top-k."""
    total_pos = y_true.sum()
    if total_pos == 0:
        return 0.0
    order = np.argsort(-scores)
    top_k = y_true[order[:k]]
    return float(top_k.sum() / total_pos)


def _mann_whitney(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    """Mann-Whitney U comparing score distributions of positives vs negatives."""
    pos_scores = scores[y_true == 1]
    neg_scores = scores[y_true == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return {"u_statistic": float("nan"), "p_value": float("nan")}
    u_stat, p_val = sp_stats.mannwhitneyu(
        pos_scores, neg_scores, alternative="two-sided",
    )
    return {"u_statistic": float(u_stat), "p_value": float(p_val)}


def _evaluate_single_score(
    y_true: np.ndarray,
    scores: np.ndarray,
    k_values: list[int],
) -> dict[str, Any]:
    """Evaluate a single score column against binary labels."""
    valid = ~np.isnan(scores)
    y_v = y_true[valid]
    s_v = scores[valid]
    n_pos = int(y_v.sum())
    n_total = len(y_v)

    result: dict[str, Any] = {
        "n_total": n_total,
        "n_positive": n_pos,
        "n_valid_scores": int(valid.sum()),
    }

    # Precision@k and Recall@k
    for k in k_values:
        if k > n_total:
            continue
        result[f"precision_at_{k}"] = _precision_at_k(y_v, s_v, k)
        result[f"recall_at_{k}"] = _recall_at_k(y_v, s_v, k)

    # AUC-ROC with CI (need >= 1 positive and >= 1 negative)
    if n_pos >= 1 and n_pos < n_total:
        if n_pos >= 30:
            result["auc_roc"] = auc_roc_with_ci(y_v, s_v)
        else:
            # Fewer positives: compute AUC without CI
            from sklearn.metrics import roc_auc_score
            try:
                auc = float(roc_auc_score(y_v, s_v))
                result["auc_roc"] = {"auc": auc, "ci_lower": None, "ci_upper": None}
            except ValueError:
                result["auc_roc"] = {"auc": float("nan")}

        # AUC-PR
        try:
            result["auc_pr"] = float(average_precision_score(y_v, s_v))
        except ValueError:
            result["auc_pr"] = float("nan")
    else:
        result["auc_roc"] = {"auc": float("nan")}
        result["auc_pr"] = float("nan")

    # Mann-Whitney U
    result["mann_whitney"] = _mann_whitney(y_v, s_v)

    return result


# ---------------------------------------------------------------------------
# Combined standardized score
# ---------------------------------------------------------------------------

def _compute_combined_score(
    df: pd.DataFrame,
    hypothesis_scores: dict[str, dict[str, Any]],
) -> np.ndarray:
    """Compute mean of standardized individual hypothesis scores."""
    score_arrays: list[np.ndarray] = []

    for h_def in hypothesis_scores.values():
        col = h_def["column"]
        if col not in df.columns:
            continue
        raw = df[col].values.astype(float)
        if np.isnan(raw).all():
            continue

        transform = h_def["transform"]
        if transform is not None:
            raw = transform(raw)

        # Standardize (ignoring NaN)
        valid = ~np.isnan(raw)
        if valid.sum() < 2:
            continue
        scaler = StandardScaler()
        standardized = np.full_like(raw, np.nan)
        standardized[valid] = scaler.fit_transform(raw[valid].reshape(-1, 1)).ravel()
        score_arrays.append(standardized)

    if not score_arrays:
        return np.full(len(df), np.nan)

    stacked = np.column_stack(score_arrays)
    return np.nanmean(stacked, axis=1)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def _cv_evaluation(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_folds: int,
    seed: int,
) -> dict[str, Any]:
    """Stratified K-fold or LOO-positive evaluation."""
    valid = ~np.isnan(scores)
    y_v = y_true[valid]
    s_v = scores[valid]
    n_pos = int(y_v.sum())

    if n_pos < 2 or n_pos >= len(y_v):
        return {"cv_method": "skipped", "reason": "insufficient class balance"}

    if n_pos >= 30:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        method = f"{n_folds}-fold stratified"
    else:
        cv = LeaveOneOut()
        method = "leave-one-out"

    fold_aucs: list[float] = []
    for _, test_idx in cv.split(s_v.reshape(-1, 1), y_v):
        y_test = y_v[test_idx]
        s_test = s_v[test_idx]
        if len(np.unique(y_test)) < 2:
            continue
        try:
            from sklearn.metrics import roc_auc_score
            fold_aucs.append(float(roc_auc_score(y_test, s_test)))
        except ValueError:
            continue

    return {
        "cv_method": method,
        "n_folds_evaluated": len(fold_aucs),
        "mean_auc": float(np.mean(fold_aucs)) if fold_aucs else float("nan"),
        "std_auc": float(np.std(fold_aucs)) if fold_aucs else float("nan"),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_stage7(base_dir: Path, config: StudyConfig) -> dict[str, Any]:
    """Evaluate author-level hypothesis scores against ground truth."""
    features_dir = base_dir / config.paths.get("features", "data/features")
    results_dir = base_dir / config.paths.get("results", "data/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    author_cfg = config.author_analysis
    k_values = author_cfg.get("precision_at_k", [10, 25, 50, 100, 250])
    n_folds = author_cfg.get("cv_folds", 5)
    seed = author_cfg.get("random_seed", 42)

    # Load author features
    features_path = features_dir / "author_features.parquet"
    logger.info("Loading author features from %s", features_path)
    df = pd.read_parquet(features_path)
    logger.info("Loaded %d authors", len(df))

    results: dict[str, Any] = {"n_authors": len(df)}

    # --------------- Primary target: suspended accounts ---------------
    if "account_status" in df.columns:
        y_suspended = (df["account_status"] == "suspended").astype(int).values
        n_suspended = int(y_suspended.sum())
        logger.info(
            "Primary target (suspended): %d positives / %d total",
            n_suspended, len(df),
        )
        results["primary_target"] = {
            "name": "suspended",
            "n_positive": n_suspended,
            "n_total": len(df),
        }

        primary_results: dict[str, Any] = {}
        for h_name, h_def in HYPOTHESIS_SCORES.items():
            col = h_def["column"]
            if col not in df.columns:
                logger.info("Skipping %s: column %s not found", h_name, col)
                primary_results[h_name] = {"skipped": True, "reason": f"{col} missing"}
                continue

            raw = df[col].values.astype(float)
            transform = h_def["transform"]
            scores = transform(raw) if transform is not None else raw

            logger.info("Evaluating %s (primary target)", h_name)
            eval_result = _evaluate_single_score(y_suspended, scores, k_values)
            eval_result["description"] = h_def["description"]
            eval_result["cv"] = _cv_evaluation(y_suspended, scores, n_folds, seed)
            primary_results[h_name] = eval_result

        # Combined score
        combined = _compute_combined_score(df, HYPOTHESIS_SCORES)
        logger.info("Evaluating Combined (primary target)")
        combined_result = _evaluate_single_score(y_suspended, combined, k_values)
        combined_result["description"] = "Mean of standardized hypothesis scores"
        combined_result["cv"] = _cv_evaluation(y_suspended, combined, n_folds, seed)
        primary_results["Combined"] = combined_result

        results["primary_results"] = primary_results
    else:
        logger.warning("No account_status column -- skipping primary target")
        results["primary_target"] = {"skipped": True}

    # --------------- Auxiliary target: suspicious authors ---------------
    has_repos = "total_repos" in df.columns
    has_merge = "merge_rate" in df.columns
    if has_repos and has_merge:
        y_suspicious = (
            (df["total_repos"] >= 3) & (df["merge_rate"] < 0.30)
        ).astype(int).values
        n_suspicious = int(y_suspicious.sum())
        logger.info(
            "Auxiliary target (suspicious): %d positives / %d total",
            n_suspicious, len(df),
        )
        results["auxiliary_target"] = {
            "name": "suspicious (total_repos>=3, merge_rate<0.30)",
            "n_positive": n_suspicious,
            "n_total": len(df),
        }

        aux_results: dict[str, Any] = {}
        for h_name, h_def in HYPOTHESIS_SCORES.items():
            col = h_def["column"]
            if col not in df.columns:
                aux_results[h_name] = {"skipped": True, "reason": f"{col} missing"}
                continue

            raw = df[col].values.astype(float)
            transform = h_def["transform"]
            scores = transform(raw) if transform is not None else raw

            logger.info("Evaluating %s (auxiliary target)", h_name)
            aux_results[h_name] = _evaluate_single_score(y_suspicious, scores, k_values)

        combined = _compute_combined_score(df, HYPOTHESIS_SCORES)
        aux_results["Combined"] = _evaluate_single_score(
            y_suspicious, combined, k_values,
        )
        results["auxiliary_results"] = aux_results
    else:
        logger.warning("Missing total_repos/merge_rate -- skipping auxiliary target")
        results["auxiliary_target"] = {"skipped": True}

    # Write results
    output_path = results_dir / "author_evaluation.json"
    write_json(output_path, results)
    logger.info("Author evaluation written to %s", output_path)

    # Checkpoint
    write_stage_checkpoint(
        base_dir / "data",
        "stage7",
        {"authors": len(df)},
        details={
            "primary_n_positive": results.get("primary_target", {}).get("n_positive", 0),
            "hypotheses_evaluated": list(HYPOTHESIS_SCORES.keys()) + ["Combined"],
        },
    )

    return results
