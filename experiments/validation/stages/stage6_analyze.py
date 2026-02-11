from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from experiments.validation.models import PROutcome, StudyConfig
from experiments.validation.plots import (
    plot_ablation_forest,
    plot_calibration,
    plot_feature_importance,
    plot_pocket_veto_by_trust,
    plot_roc_curve,
    plot_score_distributions,
)
from experiments.validation.stats import (
    auc_roc_with_ci,
    chi_squared_test,
    compute_binary_metrics,
    delong_auc_test,
    holm_bonferroni,
    kruskal_wallis_with_dunn,
    likelihood_ratio_test,
)

logger = logging.getLogger(__name__)


def _binary_target(outcome: str) -> int:
    """Convert outcome to binary: merged=1, else=0."""
    return 1 if outcome == PROutcome.MERGED.value else 0


def _trust_level_bin(
    score: float, thresholds: dict[str, float],
) -> str:
    """Bin a normalized score into trust levels."""
    if score >= thresholds.get("HIGH", 0.7):
        return "HIGH"
    if score >= thresholds.get("MEDIUM", 0.3):
        return "MEDIUM"
    return "LOW"


def run_stage6(base_dir: Path, config: StudyConfig) -> None:
    """Stage 6: Statistical analysis and report generation."""
    features_dir = base_dir / config.paths.get(
        "features", "data/features",
    )
    results_dir = base_dir / config.paths.get("results", "results")
    figures_dir = base_dir / config.paths.get(
        "figures", "results/figures",
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load feature data
    features_path = features_dir / "features.parquet"
    if not features_path.exists():
        logger.error("Features not found. Run Stage 5 first.")
        return

    df = pd.read_parquet(features_path)
    logger.info("Loaded %d feature rows", len(df))

    analysis_cfg = config.analysis
    alpha = analysis_cfg.get("alpha", 0.05)
    cv_folds = analysis_cfg.get("cv_folds", 5)
    seed = analysis_cfg.get("random_seed", 42)
    trust_thresholds = analysis_cfg.get(
        "trust_level_bins", {"HIGH": 0.7, "MEDIUM": 0.3},
    )

    all_results: dict[str, Any] = {}

    # === H1: Binary discrimination (merged vs. not merged) ===
    y_binary = df["outcome"].apply(_binary_target).values
    y_scores = df["normalized_score"].values

    h1_metrics = compute_binary_metrics(y_binary, y_scores)
    h1_auc_ci = auc_roc_with_ci(y_binary, y_scores, alpha=alpha)

    all_results["H1_binary_metrics"] = h1_metrics
    all_results["H1_auc_ci"] = h1_auc_ci

    plot_roc_curve(
        y_binary, y_scores,
        auc_ci=(h1_auc_ci["ci_lower"], h1_auc_ci["ci_upper"]),
        title="H1: GE Score as Merge Predictor",
        output_path=figures_dir / "h1_roc_curve.png",
    )
    plot_calibration(
        y_binary, y_scores,
        title="GE Score Calibration",
        output_path=figures_dir / "calibration.png",
    )

    # === H1a: Three-class analysis ===
    outcome_groups: dict[str, np.ndarray] = {}
    for outcome in PROutcome:
        mask = df["outcome"] == outcome.value
        if mask.any():
            outcome_groups[outcome.value] = (
                df.loc[mask, "normalized_score"].values
            )

    # Score distributions plot
    plot_score_distributions(
        outcome_groups,
        title="GE Score Distribution by Outcome (3-class)",
        output_path=figures_dir / "score_distributions_3class.png",
    )

    # Kruskal-Wallis across three classes
    if len(outcome_groups) >= 2:
        kw_result = kruskal_wallis_with_dunn(outcome_groups)
        all_results["H1a_kruskal_wallis"] = kw_result

    # Multinomial logistic regression
    outcome_map = {
        PROutcome.MERGED.value: 0,
        PROutcome.REJECTED.value: 1,
        PROutcome.POCKET_VETO.value: 2,
    }
    y_multi = df["outcome"].map(outcome_map).values
    x_score = y_scores.reshape(-1, 1)

    try:
        multi_lr = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=1000,
            random_state=seed,
        )
        multi_lr.fit(x_score, y_multi)
        all_results["H1a_multinomial_coefs"] = {
            "classes": list(outcome_map.keys()),
            "coefficients": multi_lr.coef_.tolist(),
            "intercepts": multi_lr.intercept_.tolist(),
        }
    except Exception:
        logger.exception("Multinomial logistic regression failed")

    # === Pocket veto deep-dive ===
    non_merged = df[df["outcome"] != PROutcome.MERGED.value].copy()
    if len(non_merged) > 0:
        non_merged["trust_bin"] = non_merged[
            "normalized_score"
        ].apply(lambda s: _trust_level_bin(s, trust_thresholds))

        # Chi-squared: trust level x outcome (rejected vs pocket veto)
        ct = pd.crosstab(
            non_merged["trust_bin"], non_merged["outcome"],
        )
        if ct.shape[0] >= 2 and ct.shape[1] >= 2:
            chi2_result = chi_squared_test(ct.values)
            all_results["pocket_veto_chi2"] = chi2_result

        # Pocket veto rate by trust level
        trust_order = ["LOW", "MEDIUM", "HIGH"]
        pv_rates = []
        rej_rates = []
        valid_levels = []
        for level in trust_order:
            subset = non_merged[non_merged["trust_bin"] == level]
            if len(subset) > 0:
                pv = (
                    subset["outcome"] == PROutcome.POCKET_VETO.value
                ).mean()
                rej = (
                    subset["outcome"] == PROutcome.REJECTED.value
                ).mean()
                pv_rates.append(pv)
                rej_rates.append(rej)
                valid_levels.append(level)

        if valid_levels:
            plot_pocket_veto_by_trust(
                valid_levels, pv_rates, rej_rates,
                title="Outcome Type by Trust Level (Non-Merged)",
                output_path=figures_dir / "pocket_veto_by_trust.png",
            )

    # === H2: Ablation study ===
    # Extract ablation scores from the dict column
    ablation_results_h2: dict[str, dict[str, float]] = {}
    ablation_p_values: dict[str, float] = {}

    if "ablation_scores" in df.columns:
        sample_val = df["ablation_scores"].iloc[0]
        if isinstance(sample_val, dict):
            ablation_names = list(sample_val.keys())
        elif isinstance(sample_val, str):
            ablation_names = list(json.loads(sample_val).keys())
        else:
            ablation_names = []

        for abl_name in ablation_names:
            def extract_score(
                row: Any, name: str = abl_name,
            ) -> float:
                scores = row["ablation_scores"]
                if isinstance(scores, str):
                    scores = json.loads(scores)
                return scores.get(name, 0.0)

            abl_scores = df.apply(extract_score, axis=1).values

            delong = delong_auc_test(y_binary, y_scores, abl_scores)
            ablation_results_h2[abl_name] = {
                "auc": delong["auc_b"],
                "auc_diff": delong["auc_b"] - delong["auc_a"],
                "p_value": delong["p_value"],
            }
            ablation_p_values[abl_name] = delong["p_value"]

        # Holm-Bonferroni correction
        if ablation_p_values:
            corrected = holm_bonferroni(
                ablation_p_values, alpha=alpha,
            )
            all_results["H2_ablation_corrected"] = corrected

        all_results["H2_ablation_raw"] = ablation_results_h2

        plot_ablation_forest(
            ablation_results_h2,
            title="H2: Ablation AUC-ROC Impact",
            output_path=figures_dir / "ablation_forest.png",
        )

    # === H3: Account age ===
    if "log_account_age_days" in df.columns:
        x_base = y_scores.reshape(-1, 1)
        x_age = np.column_stack(
            [y_scores, df["log_account_age_days"]],
        )

        try:
            lr_base = LogisticRegression(
                max_iter=1000, random_state=seed,
            )
            lr_base.fit(x_base, y_binary)
            ll_base = -log_loss_manual(
                y_binary,
                lr_base.predict_proba(x_base)[:, 1],
            ) * len(y_binary)

            lr_age = LogisticRegression(
                max_iter=1000, random_state=seed,
            )
            lr_age.fit(x_age, y_binary)
            ll_age = -log_loss_manual(
                y_binary,
                lr_age.predict_proba(x_age)[:, 1],
            ) * len(y_binary)

            lrt = likelihood_ratio_test(ll_base, ll_age, df_diff=1)
            all_results["H3_account_age_lrt"] = lrt
        except Exception:
            logger.exception("H3 analysis failed")

    # === H4: Semantic similarity ===
    if "embedding_similarity" in df.columns:
        valid_mask = df["embedding_similarity"].notna()
        if valid_mask.sum() > 50:
            x_emb = np.column_stack([
                y_scores[valid_mask],
                df.loc[valid_mask, "embedding_similarity"],
            ])
            y_emb = y_binary[valid_mask]

            try:
                lr_base = LogisticRegression(
                    max_iter=1000, random_state=seed,
                )
                lr_base.fit(
                    y_scores[valid_mask].reshape(-1, 1), y_emb,
                )
                ll_base = -log_loss_manual(
                    y_emb,
                    lr_base.predict_proba(
                        y_scores[valid_mask].reshape(-1, 1),
                    )[:, 1],
                ) * len(y_emb)

                lr_full = LogisticRegression(
                    max_iter=1000, random_state=seed,
                )
                lr_full.fit(x_emb, y_emb)
                ll_full = -log_loss_manual(
                    y_emb,
                    lr_full.predict_proba(x_emb)[:, 1],
                ) * len(y_emb)

                lrt = likelihood_ratio_test(
                    ll_base, ll_full, df_diff=1,
                )
                all_results["H4_embedding_lrt"] = lrt
            except Exception:
                logger.exception("H4 analysis failed")

    # === H5: Author merge rate ===
    if "author_merge_rate" in df.columns:
        valid_mask = df["author_merge_rate"].notna()
        if valid_mask.sum() > 50:
            x_mr = np.column_stack([
                y_scores[valid_mask],
                df.loc[valid_mask, "author_merge_rate"],
            ])
            y_mr = y_binary[valid_mask]

            try:
                lr_base = LogisticRegression(
                    max_iter=1000, random_state=seed,
                )
                lr_base.fit(
                    y_scores[valid_mask].reshape(-1, 1), y_mr,
                )
                ll_base = -log_loss_manual(
                    y_mr,
                    lr_base.predict_proba(
                        y_scores[valid_mask].reshape(-1, 1),
                    )[:, 1],
                ) * len(y_mr)

                lr_full = LogisticRegression(
                    max_iter=1000, random_state=seed,
                )
                lr_full.fit(x_mr, y_mr)
                ll_full = -log_loss_manual(
                    y_mr,
                    lr_full.predict_proba(x_mr)[:, 1],
                ) * len(y_mr)

                lrt = likelihood_ratio_test(
                    ll_base, ll_full, df_diff=1,
                )
                all_results["H5_merge_rate_lrt"] = lrt
            except Exception:
                logger.exception("H5 analysis failed")

    # === Newcomer Cohort Analysis (Rec 3) ===
    if "is_newcomer" in df.columns:
        newcomer_mask = df["is_newcomer"] == 1
        established_mask = df["is_newcomer"] == 0

        cohort_results: dict[str, Any] = {}

        for cohort_name, mask in [
            ("newcomer", newcomer_mask),
            ("established", established_mask),
        ]:
            cohort_df = df[mask]
            if len(cohort_df) < 20:
                cohort_results[cohort_name] = {
                    "n": len(cohort_df),
                    "note": "Too few samples for analysis",
                }
                continue

            y_cohort = cohort_df["outcome"].apply(
                _binary_target,
            ).values
            s_cohort = cohort_df["normalized_score"].values

            # Only compute if there are both classes
            if len(set(y_cohort)) < 2:
                cohort_results[cohort_name] = {
                    "n": len(cohort_df),
                    "note": "Only one class present",
                }
                continue

            cohort_metrics = compute_binary_metrics(
                y_cohort, s_cohort,
            )
            cohort_auc_ci = auc_roc_with_ci(
                y_cohort, s_cohort, alpha=alpha,
            )

            cohort_results[cohort_name] = {
                "n": len(cohort_df),
                "n_merged": int(y_cohort.sum()),
                "n_not_merged": int(
                    len(y_cohort) - y_cohort.sum(),
                ),
                "auc_roc": cohort_metrics["auc_roc"],
                "auc_ci": cohort_auc_ci,
                "mean_score": float(s_cohort.mean()),
                "median_score": float(np.median(s_cohort)),
                "merge_rate": float(y_cohort.mean()),
            }

        all_results["newcomer_cohort"] = cohort_results

        # Log summary
        for name, res in cohort_results.items():
            if "auc_roc" in res:
                logger.info(
                    "Cohort %s: n=%d, AUC=%.3f, merge_rate=%.3f",
                    name,
                    res["n"],
                    res["auc_roc"],
                    res["merge_rate"],
                )

    # === Cross-validation ===
    groups = df["repo"].values
    try:
        cv = StratifiedGroupKFold(
            n_splits=cv_folds, shuffle=True, random_state=seed,
        )
        cv_aucs = []
        for train_idx, test_idx in cv.split(
            x_score, y_binary, groups,
        ):
            lr = LogisticRegression(
                max_iter=1000, random_state=seed,
            )
            lr.fit(x_score[train_idx], y_binary[train_idx])
            proba = lr.predict_proba(x_score[test_idx])[:, 1]
            fold_auc = auc_roc_with_ci(y_binary[test_idx], proba)
            cv_aucs.append(fold_auc["auc"])

        all_results["cross_validation"] = {
            "fold_aucs": cv_aucs,
            "mean_auc": float(np.mean(cv_aucs)),
            "std_auc": float(np.std(cv_aucs)),
        }
    except Exception:
        logger.exception("Cross-validation failed")

    # === Feature importance ===
    feature_cols = [
        "normalized_score",
        "log_account_age_days",
        "log_followers",
        "log_public_repos",
    ]
    available = [c for c in feature_cols if c in df.columns]
    if available:
        x_all = df[available].fillna(0).values
        try:
            lr = LogisticRegression(
                max_iter=1000, random_state=seed,
            )
            lr.fit(x_all, y_binary)
            plot_feature_importance(
                available,
                lr.coef_[0].tolist(),
                title="Feature Coefficients (Logistic Regression)",
                output_path=(
                    figures_dir / "feature_importance.png"
                ),
            )
        except Exception:
            logger.exception("Feature importance plot failed")

    # === Save results ===
    results_path = results_dir / "statistical_tests.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Saved statistical results to %s", results_path)

    # === Generate pilot report ===
    _generate_report(results_dir, all_results, len(df))

    logger.info("Stage 6 complete")


def log_loss_manual(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> float:
    """Compute mean log loss."""
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -float(np.mean(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    ))


def _generate_report(
    results_dir: Path,
    results: dict[str, Any],
    n_samples: int,
) -> None:
    """Generate a markdown pilot report."""
    report_path = results_dir / "pilot_report.md"

    lines = [
        "# GE Validation Pilot Report",
        "",
        f"**Sample size**: {n_samples} PRs",
        "",
        "## H1: Binary Discrimination (Merged vs. Not Merged)",
        "",
    ]

    h1 = results.get("H1_binary_metrics", {})
    h1_ci = results.get("H1_auc_ci", {})
    lines.extend([
        f"- **AUC-ROC**: {h1.get('auc_roc', 'N/A'):.4f}"
        f" (95% CI: [{h1_ci.get('ci_lower', 'N/A'):.4f},"
        f" {h1_ci.get('ci_upper', 'N/A'):.4f}])",
        f"- **AUC-PR**: {h1.get('auc_pr', 'N/A'):.4f}",
        f"- **Brier Score**: {h1.get('brier_score', 'N/A'):.4f}",
        f"- **Log Loss**: {h1.get('log_loss', 'N/A'):.4f}",
        "",
        "## H1a: Three-Class Analysis",
        "",
    ])

    kw = results.get("H1a_kruskal_wallis", {})
    if kw:
        lines.extend([
            f"- **Kruskal-Wallis H**:"
            f" {kw.get('h_statistic', 'N/A'):.4f}"
            f" (p = {kw.get('p_value', 'N/A'):.4e})",
            "",
        ])

    lines.extend(["## H2: Ablation Study", ""])
    h2 = results.get("H2_ablation_raw", {})
    for name, vals in h2.items():
        lines.append(
            f"- **{name}**: AUC = {vals.get('auc', 0):.4f}"
            f" (diff = {vals.get('auc_diff', 0):+.4f},"
            f" p = {vals.get('p_value', 1):.4e})"
        )

    lines.extend(["", "## Cross-Validation", ""])
    cv = results.get("cross_validation", {})
    if cv:
        lines.append(
            f"- **Mean AUC**: {cv.get('mean_auc', 0):.4f}"
            f" +/- {cv.get('std_auc', 0):.4f}"
        )

    # Newcomer cohort
    cohort = results.get("newcomer_cohort", {})
    if cohort:
        lines.extend(["", "## Newcomer Cohort Analysis", ""])
        for name, vals in cohort.items():
            if "auc_roc" in vals:
                ci = vals.get("auc_ci", {})
                lines.append(
                    f"- **{name.title()}** (n={vals['n']}): "
                    f"AUC = {vals['auc_roc']:.4f} "
                    f"(CI: [{ci.get('ci_lower', 0):.4f}, "
                    f"{ci.get('ci_upper', 0):.4f}]), "
                    f"merge rate = {vals['merge_rate']:.3f}"
                )
            else:
                lines.append(
                    f"- **{name.title()}** (n={vals['n']}): "
                    f"{vals.get('note', 'N/A')}"
                )

    lines.append("")
    lines.append("---")
    lines.append("*Generated by GE Validation Pipeline*")

    report_path.write_text("\n".join(lines))
    logger.info("Saved pilot report to %s", report_path)
