from __future__ import annotations

import logging
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from experiments.bot_detection.checkpoint import write_json, write_stage_checkpoint
from experiments.bot_detection.models import StudyConfig
from experiments.bot_detection.stats import (
    auc_roc_with_ci,
    compute_binary_metrics,
    delong_auc_test,
    holm_bonferroni,
    likelihood_ratio_test,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature column groups
# ---------------------------------------------------------------------------
H1_COLS = [
    "burst_count_1h", "burst_repos_1h",
    "burst_count_24h", "burst_repos_24h", "burst_max_rate",
]
H2_COLS = [
    "review_response_rate", "ci_failure_followup_rate",
    "avg_response_latency_hours", "abandoned_pr_rate",
]
H3_COLS = [
    "max_title_similarity", "language_entropy",
    "topic_coherence", "duplicate_title_count",
]
GE_COL = "ge_score"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_binary_outcome(df: pd.DataFrame) -> np.ndarray:
    """Binary label: merged=0, not_merged=1 (positive = non-merge)."""
    return (df["outcome"] != "merged").astype(int).values


def _fill_and_scale(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Impute NaN with column median from train, then standard-scale."""
    scaler = StandardScaler()
    medians = np.nanmedian(x_train, axis=0)
    # Fall back to 0.0 for all-NaN columns (e.g. abandoned_pr_rate, ge_score)
    medians = np.where(np.isnan(medians), 0.0, medians)
    for col_idx in range(x_train.shape[1]):
        mask_train = np.isnan(x_train[:, col_idx])
        x_train[mask_train, col_idx] = medians[col_idx]
        mask_test = np.isnan(x_test[:, col_idx])
        x_test[mask_test, col_idx] = medians[col_idx]
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, scaler


def _fit_lr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> LogisticRegression:
    """Fit LogisticRegression with no penalty."""
    model = LogisticRegression(
        C=np.inf, solver="lbfgs", max_iter=1000, random_state=seed,
    )
    model.fit(x_train, y_train)
    return model


def _cv_predict(
    df: pd.DataFrame,
    feature_cols: list[str],
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
    seed: int,
) -> np.ndarray:
    """5-fold StratifiedGroupKFold CV, returning out-of-fold probabilities."""
    x_all = df[feature_cols].values.astype(float)
    oof_probs = np.full(len(y), np.nan)
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for train_idx, test_idx in cv.split(x_all, y, groups):
        x_tr, x_te = x_all[train_idx].copy(), x_all[test_idx].copy()
        y_tr = y[train_idx]
        x_tr, x_te, _ = _fill_and_scale(x_tr, x_te)
        model = _fit_lr(x_tr, y_tr, seed)
        oof_probs[test_idx] = model.predict_proba(x_te)[:, 1]

    return oof_probs


def _full_model_log_likelihood(
    df: pd.DataFrame,
    feature_cols: list[str],
    y: np.ndarray,
    seed: int,
) -> float:
    """Fit on all data, return log-likelihood."""
    x_all = df[feature_cols].values.astype(float)
    medians = np.nanmedian(x_all, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    for col_idx in range(x_all.shape[1]):
        mask = np.isnan(x_all[:, col_idx])
        x_all[mask, col_idx] = medians[col_idx]
    scaler = StandardScaler()
    x_all = scaler.fit_transform(x_all)
    model = _fit_lr(x_all, y, seed)
    probs = model.predict_proba(x_all)[:, 1]
    probs = np.clip(probs, 1e-15, 1 - 1e-15)
    return float(np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs)))


# ---------------------------------------------------------------------------
# Individual hypothesis evaluations
# ---------------------------------------------------------------------------

def _evaluate_hypothesis(
    df: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    feature_cols: list[str],
    hypothesis_name: str,
    n_folds: int,
    seed: int,
) -> dict[str, Any]:
    """Run CV + metrics + Mann-Whitney for a single hypothesis."""
    oof_probs = _cv_predict(df, feature_cols, y, groups, n_folds, seed)

    # Drop any NaN from folds that didn't converge
    valid = ~np.isnan(oof_probs)
    y_v = y[valid]
    p_v = oof_probs[valid]

    auc_ci = auc_roc_with_ci(y_v, p_v)
    metrics = compute_binary_metrics(y_v, p_v)

    # Mann-Whitney U between predicted scores for positive vs negative
    pos_scores = p_v[y_v == 1]
    neg_scores = p_v[y_v == 0]
    if len(pos_scores) > 0 and len(neg_scores) > 0:
        u_stat, u_p = sp_stats.mannwhitneyu(
            pos_scores, neg_scores, alternative="two-sided",
        )
        mann_whitney = {"u_statistic": float(u_stat), "p_value": float(u_p)}
    else:
        mann_whitney = {"u_statistic": float("nan"), "p_value": float("nan")}

    return {
        "hypothesis": hypothesis_name,
        "features": feature_cols,
        "n_samples": int(valid.sum()),
        "auc_roc": auc_ci["auc"],
        "auc_roc_ci_lower": auc_ci["ci_lower"],
        "auc_roc_ci_upper": auc_ci["ci_upper"],
        "auc_roc_se": auc_ci["se"],
        "auc_pr": metrics["auc_pr"],
        "brier_score": metrics["brier_score"],
        "log_loss": metrics["log_loss"],
        "mann_whitney": mann_whitney,
        "oof_probs": oof_probs.tolist(),
    }


# ---------------------------------------------------------------------------
# H1 parameter sweep
# ---------------------------------------------------------------------------

def _h1_burstiness_sweep(
    df: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    config: StudyConfig,
) -> dict[str, Any]:
    """Threshold sweep over burstiness columns, Holm-Bonferroni corrected."""
    seed = config.analysis.get("random_seed", 42)
    n_folds = config.analysis.get("cv_folds", 5)

    burst_cols = [
        "burst_count_1h", "burst_repos_1h",
        "burst_count_24h", "burst_repos_24h",
    ]
    thresholds = [1, 2, 3, 5, 7, 10]

    sweep_results: list[dict[str, Any]] = []
    p_values: dict[str, float] = {}

    for col, thresh in product(burst_cols, thresholds):
        name = f"{col}_ge_{thresh}"
        binary_feature = (df[col] >= thresh).astype(int).values
        # Use as single binary predictor
        df_tmp = df.copy()
        df_tmp["_sweep_feature"] = binary_feature

        oof = _cv_predict(df_tmp, ["_sweep_feature"], y, groups, n_folds, seed)
        valid = ~np.isnan(oof)
        y_v = y[valid]
        p_v = oof[valid]

        if len(np.unique(y_v)) < 2 or len(np.unique(binary_feature)) < 2:
            continue

        auc_ci = auc_roc_with_ci(y_v, p_v)
        # Mann-Whitney on the raw binary feature vs outcome
        pos_vals = binary_feature[y == 1]
        neg_vals = binary_feature[y == 0]
        _, mw_p = sp_stats.mannwhitneyu(
            pos_vals, neg_vals, alternative="two-sided",
        )

        entry = {
            "config": name,
            "column": col,
            "threshold": thresh,
            "auc_roc": auc_ci["auc"],
            "auc_roc_ci": [auc_ci["ci_lower"], auc_ci["ci_upper"]],
            "mann_whitney_p": float(mw_p),
        }
        sweep_results.append(entry)
        p_values[name] = float(mw_p)

    corrected = holm_bonferroni(p_values, config.analysis.get("alpha", 0.05))

    return {
        "sweep_configs": sweep_results,
        "holm_bonferroni": {
            k: {kk: vv for kk, vv in v.items()}
            for k, v in corrected.items()
        },
        "n_configs": len(sweep_results),
    }


# ---------------------------------------------------------------------------
# H3a vs H3b DeLong comparison
# ---------------------------------------------------------------------------

def _h3_comparison(
    df: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    config: StudyConfig,
) -> dict[str, Any]:
    """DeLong test comparing H3a (text stats) vs H3b (all H3 features)."""
    seed = config.analysis.get("random_seed", 42)
    n_folds = config.analysis.get("cv_folds", 5)

    h3a_cols = ["max_title_similarity", "duplicate_title_count"]
    h3b_cols = H3_COLS  # includes language_entropy, topic_coherence

    oof_a = _cv_predict(df, h3a_cols, y, groups, n_folds, seed)
    oof_b = _cv_predict(df, h3b_cols, y, groups, n_folds, seed)

    valid = ~(np.isnan(oof_a) | np.isnan(oof_b))
    y_v = y[valid]

    delong = delong_auc_test(y_v, oof_a[valid], oof_b[valid])

    return {
        "h3a_features": h3a_cols,
        "h3b_features": h3b_cols,
        "delong": delong,
    }


# ---------------------------------------------------------------------------
# H4 combined model with nested LRT
# ---------------------------------------------------------------------------

def _h4_combined(
    df: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    config: StudyConfig,
) -> dict[str, Any]:
    """Combined model (H1+H2+H3) with nested LRT against single-signal."""
    seed = config.analysis.get("random_seed", 42)
    n_folds = config.analysis.get("cv_folds", 5)
    all_cols = H1_COLS + H2_COLS + H3_COLS

    oof_combined = _cv_predict(df, all_cols, y, groups, n_folds, seed)
    valid = ~np.isnan(oof_combined)
    auc_ci = auc_roc_with_ci(y[valid], oof_combined[valid])
    metrics = compute_binary_metrics(y[valid], oof_combined[valid])

    # Nested LRT: full model vs each single-signal model
    ll_full = _full_model_log_likelihood(df, all_cols, y, seed)
    lrt_results: dict[str, Any] = {}

    for name, cols in [("H1", H1_COLS), ("H2", H2_COLS), ("H3", H3_COLS)]:
        ll_restricted = _full_model_log_likelihood(df, cols, y, seed)
        df_diff = len(all_cols) - len(cols)
        lrt = likelihood_ratio_test(ll_restricted, ll_full, df_diff)
        lrt_results[name] = lrt

    return {
        "features": all_cols,
        "auc_roc": auc_ci["auc"],
        "auc_roc_ci": [auc_ci["ci_lower"], auc_ci["ci_upper"]],
        "auc_pr": metrics["auc_pr"],
        "brier_score": metrics["brier_score"],
        "log_loss": metrics["log_loss"],
        "nested_lrt": lrt_results,
        "oof_probs": oof_combined.tolist(),
    }


# ---------------------------------------------------------------------------
# H5 GE complement
# ---------------------------------------------------------------------------

def _h5_ge_complement(
    df: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    config: StudyConfig,
) -> dict[str, Any]:
    """LRT of (GE + bot signals) vs GE-only."""
    seed = config.analysis.get("random_seed", 42)
    n_folds = config.analysis.get("cv_folds", 5)
    bot_cols = H1_COLS + H2_COLS + H3_COLS
    ge_only = [GE_COL]
    ge_plus = [GE_COL] + bot_cols

    oof_ge = _cv_predict(df, ge_only, y, groups, n_folds, seed)
    oof_ge_plus = _cv_predict(df, ge_plus, y, groups, n_folds, seed)

    valid = ~(np.isnan(oof_ge) | np.isnan(oof_ge_plus))
    y_v = y[valid]

    auc_ge = auc_roc_with_ci(y_v, oof_ge[valid])
    auc_ge_plus = auc_roc_with_ci(y_v, oof_ge_plus[valid])
    delong = delong_auc_test(y_v, oof_ge_plus[valid], oof_ge[valid])

    # LRT on full data
    ll_ge = _full_model_log_likelihood(df, ge_only, y, seed)
    ll_ge_plus = _full_model_log_likelihood(df, ge_plus, y, seed)
    df_diff = len(ge_plus) - len(ge_only)
    lrt = likelihood_ratio_test(ll_ge, ll_ge_plus, df_diff)

    return {
        "ge_only_auc": auc_ge,
        "ge_plus_bot_auc": auc_ge_plus,
        "delong": delong,
        "lrt": lrt,
    }


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------

def _write_summary_md(
    results: dict[str, Any],
    output_path: Path,
) -> None:
    """Write a human-readable summary to SUMMARY.md."""
    lines: list[str] = []
    lines.append("# Stage 3: Evaluation Results")
    lines.append("")

    # Per-hypothesis results
    for h_key in ["H1", "H2", "H3"]:
        h = results.get(h_key)
        if not h:
            continue
        lines.append(f"## {h_key}: {h['hypothesis']}")
        lines.append(
            f"- AUC-ROC: {h['auc_roc']:.3f} "
            f"[{h['auc_roc_ci_lower']:.3f}, {h['auc_roc_ci_upper']:.3f}]"
        )
        lines.append(f"- AUC-PR: {h['auc_pr']:.3f}")
        mw = h["mann_whitney"]
        lines.append(
            f"- Mann-Whitney U: {mw['u_statistic']:.1f}, p={mw['p_value']:.4g}"
        )
        lines.append("")

    # H1 sweep
    sweep = results.get("h1_sweep")
    if sweep:
        lines.append("## H1 Burstiness Sweep")
        lines.append(f"- Configs tested: {sweep['n_configs']}")
        hb = sweep.get("holm_bonferroni", {})
        sig = [k for k, v in hb.items() if v.get("reject")]
        lines.append(f"- Significant after Holm-Bonferroni: {len(sig)}")
        if sig:
            for s in sig[:5]:
                lines.append(f"  - {s}: adj_p={hb[s]['adjusted_p']:.4g}")
        lines.append("")

    # H3 comparison
    h3c = results.get("h3_comparison")
    if h3c:
        dl = h3c["delong"]
        lines.append("## H3a vs H3b Comparison (DeLong)")
        lines.append(f"- H3a AUC: {dl['auc_a']:.3f}, H3b AUC: {dl['auc_b']:.3f}")
        lines.append(f"- z={dl['z_statistic']:.3f}, p={dl['p_value']:.4g}")
        lines.append("")

    # H4
    h4 = results.get("H4_combined")
    if h4:
        lines.append("## H4 Combined Model")
        lines.append(
            f"- AUC-ROC: {h4['auc_roc']:.3f} "
            f"[{h4['auc_roc_ci'][0]:.3f}, {h4['auc_roc_ci'][1]:.3f}]"
        )
        for name, lrt in h4["nested_lrt"].items():
            lines.append(
                f"- LRT vs {name}: chi2={lrt['lr_statistic']:.2f}, "
                f"df={lrt['df']}, p={lrt['p_value']:.4g}"
            )
        lines.append("")

    # H5
    h5 = results.get("H5_ge_complement")
    if h5 and not h5.get("skipped"):
        lines.append("## H5 GE Complement")
        lines.append(f"- GE-only AUC: {h5['ge_only_auc']['auc']:.3f}")
        lines.append(f"- GE+bot AUC: {h5['ge_plus_bot_auc']['auc']:.3f}")
        lrt = h5["lrt"]
        lines.append(
            f"- LRT: chi2={lrt['lr_statistic']:.2f}, "
            f"df={lrt['df']}, p={lrt['p_value']:.4g}"
        )
        dl = h5["delong"]
        lines.append(f"- DeLong: z={dl['z_statistic']:.3f}, p={dl['p_value']:.4g}")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    logger.info("Summary written to %s", output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_stage3(base_dir: Path, config: StudyConfig) -> dict[str, Any]:
    """Run all stage-3 evaluations.

    Reads features from Parquet, runs H1-H5 analyses, writes
    results/statistical_tests.json and results/SUMMARY.md.
    """
    seed = config.analysis.get("random_seed", 42)
    n_folds = config.analysis.get("cv_folds", 5)
    features_dir = base_dir / config.paths.get("features", "data/features")
    results_dir = base_dir / config.paths.get("results", "data/results")

    # Load features
    features_path = features_dir / "features.parquet"
    logger.info("Loading features from %s", features_path)
    df = pd.read_parquet(features_path)
    logger.info("Loaded %d rows", len(df))

    y = _make_binary_outcome(df)
    groups = df["repo"].values

    results: dict[str, Any] = {"n_samples": len(df)}

    # H1: Burstiness
    logger.info("Evaluating H1 (burstiness)")
    results["H1"] = _evaluate_hypothesis(
        df, y, groups, H1_COLS, "Burstiness", n_folds, seed,
    )

    # H2: Engagement
    logger.info("Evaluating H2 (engagement)")
    results["H2"] = _evaluate_hypothesis(
        df, y, groups, H2_COLS, "Engagement", n_folds, seed,
    )

    # H3: Cross-repo
    logger.info("Evaluating H3 (cross-repo)")
    results["H3"] = _evaluate_hypothesis(
        df, y, groups, H3_COLS, "Cross-repo fingerprinting", n_folds, seed,
    )

    # H1 parameter sweep
    logger.info("Running H1 burstiness sweep")
    results["h1_sweep"] = _h1_burstiness_sweep(df, y, groups, config)

    # H3a vs H3b
    logger.info("Running H3a vs H3b comparison")
    results["h3_comparison"] = _h3_comparison(df, y, groups, config)

    # H4 combined
    logger.info("Evaluating H4 (combined model)")
    results["H4_combined"] = _h4_combined(df, y, groups, config)

    # H5 GE complement (skip if ge_score is entirely NaN)
    if df[GE_COL].notna().any():
        logger.info("Evaluating H5 (GE complement)")
        results["H5_ge_complement"] = _h5_ge_complement(df, y, groups, config)
    else:
        logger.warning("Skipping H5: ge_score is entirely NaN (no author GE scores loaded)")
        results["H5_ge_complement"] = {"skipped": True, "reason": "ge_score all NaN"}

    # Strip oof_probs from JSON output (keep for downstream but don't persist)
    json_results = _strip_oof_probs(results)

    # Write outputs
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "statistical_tests.json"
    write_json(json_path, json_results)
    logger.info("Statistical tests written to %s", json_path)

    summary_path = results_dir / "SUMMARY.md"
    _write_summary_md(results, summary_path)

    # Checkpoint
    write_stage_checkpoint(
        base_dir / "data",
        "stage3",
        row_counts={"features": len(df)},
        details={"hypotheses_evaluated": ["H1", "H2", "H3", "H4", "H5"]},
    )

    return results


def _strip_oof_probs(obj: Any) -> Any:
    """Recursively remove oof_probs keys for JSON serialization."""
    if isinstance(obj, dict):
        return {
            k: _strip_oof_probs(v)
            for k, v in obj.items()
            if k != "oof_probs"
        }
    if isinstance(obj, list):
        return [_strip_oof_probs(item) for item in obj]
    return obj
