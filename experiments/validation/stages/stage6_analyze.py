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
    plot_baseline_comparison,
    plot_calibration,
    plot_feature_importance,
    plot_pocket_veto_by_trust,
    plot_roc_curve,
    plot_score_distributions,
)
from experiments.validation.stats import (
    auc_roc_with_ci,
    chi_squared_test,
    cochran_armitage_trend,
    compute_binary_metrics,
    delong_auc_test,
    holm_bonferroni,
    kruskal_wallis_with_dunn,
    likelihood_ratio_test,
    odds_ratio,
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

    # Note: brier_score and log_loss are computed on raw (uncalibrated)
    # scores. They are not interpretable as calibration metrics without
    # Platt scaling. AUC-ROC and AUC-PR are rank-based and remain valid.
    h1_metrics["_note"] = (
        "brier_score and log_loss use uncalibrated scores; "
        "interpret with caution"
    )
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
            penalty=None,
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

        # Cochran-Armitage trend test (DOE Section 6.3)
        # Build 2xK table: rows = outcome (pocket_veto=0, rejected=1),
        # columns = trust level (LOW, MEDIUM, HIGH)
        try:
            ca_rows = []
            for level in trust_order:
                subset = non_merged[non_merged["trust_bin"] == level]
                if len(subset) > 0:
                    n_pv = (
                        subset["outcome"] == PROutcome.POCKET_VETO.value
                    ).sum()
                    n_rej = (
                        subset["outcome"] == PROutcome.REJECTED.value
                    ).sum()
                    ca_rows.append((n_pv, n_rej))

            if len(ca_rows) >= 2:
                ca_table = np.array(ca_rows).T  # 2 x K
                ca_result = cochran_armitage_trend(ca_table)
                all_results["pocket_veto_cochran_armitage"] = ca_result
        except Exception:
            logger.exception("Cochran-Armitage trend test failed")

        # Odds ratios for trust level pairs (DOE Section 6.4)
        try:
            trust_odds: dict[str, Any] = {}
            # Binary outcome for odds: merged=1, not-merged=0
            all_with_trust = df.copy()
            all_with_trust["trust_bin"] = all_with_trust[
                "normalized_score"
            ].apply(lambda s: _trust_level_bin(s, trust_thresholds))
            all_with_trust["is_merged"] = (
                all_with_trust["outcome"] == PROutcome.MERGED.value
            ).astype(int)

            pairs = [
                ("HIGH", "LOW"),
                ("HIGH", "MEDIUM"),
                ("MEDIUM", "LOW"),
            ]
            for level_a, level_b in pairs:
                subset = all_with_trust[
                    all_with_trust["trust_bin"].isin([level_a, level_b])
                ]
                if len(subset) < 10:
                    continue
                ct_2x2 = pd.crosstab(
                    subset["trust_bin"] == level_a,
                    subset["is_merged"],
                ).values
                if ct_2x2.shape == (2, 2):
                    or_result = odds_ratio(ct_2x2)
                    trust_odds[f"{level_a}_vs_{level_b}"] = or_result

            if trust_odds:
                all_results["trust_level_odds_ratios"] = trust_odds
        except Exception:
            logger.exception("Odds ratio computation failed")

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

        # Holm-Bonferroni correction on the 6 primary single-dimension
        # ablations only (DOE Section 7.4). Two-way interactions and
        # recursive_quality are exploratory.
        primary_ablations = {
            "no_recency", "no_repo_quality", "no_self_penalty",
            "no_language_match", "no_diversity_volume",
            "no_language_norm",
        }
        primary_p_values = {
            k: v for k, v in ablation_p_values.items()
            if k in primary_ablations
        }
        exploratory_p_values = {
            k: v for k, v in ablation_p_values.items()
            if k not in primary_ablations
        }
        if primary_p_values:
            corrected = holm_bonferroni(
                primary_p_values, alpha=alpha,
            )
            all_results["H2_ablation_corrected"] = corrected
        if exploratory_p_values:
            corrected_exploratory = holm_bonferroni(
                exploratory_p_values, alpha=alpha,
            )
            all_results["H2_ablation_exploratory"] = (
                corrected_exploratory
            )

        all_results["H2_ablation_raw"] = ablation_results_h2

        plot_ablation_forest(
            ablation_results_h2,
            title="H2: Ablation AUC-ROC Impact",
            output_path=figures_dir / "ablation_forest.png",
        )

    # === H3: Account age ===
    # LRTs require unregularized logistic regression (penalty=None)
    # so the chi-squared distributional assumption holds.
    if "log_account_age_days" in df.columns:
        # Exclude rows with missing/imputed-zero account age, matching
        # the H4/H5 pattern of filtering to valid data only.
        h3_valid = (
            df["log_account_age_days"].notna()
            & (df["log_account_age_days"] > 0)
        )
        if h3_valid.sum() > 50:
            x_base_h3 = y_scores[h3_valid].reshape(-1, 1)
            x_age = np.column_stack(
                [y_scores[h3_valid], df.loc[h3_valid, "log_account_age_days"]],
            )
            y_h3 = y_binary[h3_valid]

            try:
                lr_base = LogisticRegression(
                    penalty=None, max_iter=1000, random_state=seed,
                )
                lr_base.fit(x_base_h3, y_h3)
                ll_base = -log_loss_manual(
                    y_h3,
                    lr_base.predict_proba(x_base_h3)[:, 1],
                ) * len(y_h3)

                lr_age = LogisticRegression(
                    penalty=None, max_iter=1000, random_state=seed,
                )
                lr_age.fit(x_age, y_h3)
                ll_age = -log_loss_manual(
                    y_h3,
                    lr_age.predict_proba(x_age)[:, 1],
                ) * len(y_h3)

                lrt = likelihood_ratio_test(ll_base, ll_age, df_diff=1)
                lrt["n_valid"] = int(h3_valid.sum())
                all_results["H3_account_age_lrt"] = lrt
            except Exception:
                logger.exception("H3 analysis failed")

    # === H4: Semantic similarity ===
    # Note: embedding similarity feature has known limitations (see
    # RED_TEAM_AUDIT.md C2). Results should be interpreted as
    # inconclusive rather than definitive.
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
                    penalty=None,
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
                    penalty=None,
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
                lrt["n_valid"] = int(valid_mask.sum())
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
                    penalty=None,
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
                    penalty=None,
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
                lrt["n_valid"] = int(valid_mask.sum())
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

    # === One-vs-rest AUC (DOE Section 6.2) ===
    try:
        ovr_aucs: dict[str, float] = {}
        for outcome_name, outcome_code in outcome_map.items():
            y_ovr = (y_multi == outcome_code).astype(int)
            if y_ovr.sum() > 0 and y_ovr.sum() < len(y_ovr):
                ovr_auc = auc_roc_with_ci(y_ovr, y_scores, alpha=alpha)
                ovr_aucs[outcome_name] = ovr_auc
        all_results["H1a_one_vs_rest_auc"] = ovr_aucs
    except Exception:
        logger.exception("One-vs-rest AUC failed")

    # === Confusion matrix at Youden's J (DOE Section 6.2) ===
    try:
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import roc_curve as sk_roc

        fpr, tpr, thresholds = sk_roc(y_binary, y_scores)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = float(thresholds[best_idx])

        y_pred_binary = (y_scores >= best_threshold).astype(int)
        cm = confusion_matrix(y_binary, y_pred_binary)
        all_results["confusion_matrix_binary"] = {
            "threshold": best_threshold,
            "youdens_j": float(j_scores[best_idx]),
            "matrix": cm.tolist(),
            "labels": ["not_merged", "merged"],
        }
    except Exception:
        logger.exception("Confusion matrix computation failed")

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
                penalty=None, max_iter=1000, random_state=seed,
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

    # === Baseline comparisons ===
    # Compare GE score against simple features and "dumb" models to
    # assess whether the graph machinery adds value beyond arithmetic.
    try:
        baseline_results: dict[str, Any] = {}
        ge_auc_val = h1_auc_ci["auc"]

        # --- Single-feature AUC baselines ---
        single_feature_aucs: dict[str, Any] = {}
        single_feature_cols = {
            "author_merge_rate": "Author merge rate",
            "log_account_age_days": "Account age (log)",
            "log_followers": "Followers (log)",
            "log_public_repos": "Public repos (log)",
            "embedding_similarity": "Embedding similarity",
        }
        # total_prs_at_time is prior merge count, already in df
        if "total_prs_at_time" in df.columns:
            single_feature_cols["total_prs_at_time"] = (
                "Prior merge count"
            )

        for col, label in single_feature_cols.items():
            if col not in df.columns:
                continue
            feat_valid = df[col].notna()
            if feat_valid.sum() < 50:
                continue
            feat_scores = df.loc[feat_valid, col].values
            feat_y = y_binary[feat_valid]
            feat_auc_ci = auc_roc_with_ci(feat_y, feat_scores)
            # Paired DeLong test vs GE score
            feat_delong = delong_auc_test(
                feat_y,
                y_scores[feat_valid],
                feat_scores,
            )
            single_feature_aucs[col] = {
                "label": label,
                "auc": feat_auc_ci["auc"],
                "ci_lower": feat_auc_ci["ci_lower"],
                "ci_upper": feat_auc_ci["ci_upper"],
                "n_valid": int(feat_valid.sum()),
                "vs_ge_delong_z": feat_delong["z_statistic"],
                "vs_ge_delong_p": feat_delong["p_value"],
            }
        baseline_results["single_feature_aucs"] = single_feature_aucs

        # --- "Dumb baseline" models ---
        # Model A: merge_rate + account_age
        dumb_feature_sets = {
            "model_A_merge_rate_age": [
                "author_merge_rate", "log_account_age_days",
            ],
            "model_B_merge_rate_age_emb": [
                "author_merge_rate", "log_account_age_days",
                "embedding_similarity",
            ],
        }
        for model_name, feat_cols in dumb_feature_sets.items():
            avail_cols = [c for c in feat_cols if c in df.columns]
            if len(avail_cols) != len(feat_cols):
                continue
            valid = df[avail_cols].notna().all(axis=1)
            if valid.sum() < 50:
                continue
            x_dumb = df.loc[valid, avail_cols].values
            y_dumb = y_binary[valid]
            groups_dumb = groups[valid]

            # Full-sample AUC via LR
            lr_dumb = LogisticRegression(
                penalty=None, max_iter=1000, random_state=seed,
            )
            lr_dumb.fit(x_dumb, y_dumb)
            dumb_proba = lr_dumb.predict_proba(x_dumb)[:, 1]
            dumb_auc_ci = auc_roc_with_ci(y_dumb, dumb_proba)

            # 5-fold grouped CV
            try:
                cv_dumb = StratifiedGroupKFold(
                    n_splits=cv_folds, shuffle=True,
                    random_state=seed,
                )
                cv_dumb_aucs = []
                for tr_idx, te_idx in cv_dumb.split(
                    x_dumb, y_dumb, groups_dumb,
                ):
                    lr_cv = LogisticRegression(
                        penalty=None, max_iter=1000,
                        random_state=seed,
                    )
                    lr_cv.fit(x_dumb[tr_idx], y_dumb[tr_idx])
                    p = lr_cv.predict_proba(x_dumb[te_idx])[:, 1]
                    cv_a = auc_roc_with_ci(y_dumb[te_idx], p)
                    cv_dumb_aucs.append(cv_a["auc"])
                cv_mean = float(np.mean(cv_dumb_aucs))
                cv_std = float(np.std(cv_dumb_aucs))
            except Exception:
                cv_mean = float("nan")
                cv_std = float("nan")
                cv_dumb_aucs = []

            # DeLong vs GE (on common valid subset)
            delong_vs_ge = delong_auc_test(
                y_dumb, y_scores[valid], dumb_proba,
            )

            baseline_results[model_name] = {
                "features": avail_cols,
                "auc": dumb_auc_ci["auc"],
                "ci_lower": dumb_auc_ci["ci_lower"],
                "ci_upper": dumb_auc_ci["ci_upper"],
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "cv_fold_aucs": cv_dumb_aucs,
                "n_valid": int(valid.sum()),
                "vs_ge_delong_z": delong_vs_ge["z_statistic"],
                "vs_ge_delong_p": delong_vs_ge["p_value"],
            }

        # --- Combined model: GE + significant external features ---
        combined_cols = ["normalized_score"]
        for c in [
            "author_merge_rate", "log_account_age_days",
            "embedding_similarity",
        ]:
            if c in df.columns:
                combined_cols.append(c)
        if len(combined_cols) > 1:
            valid_comb = df[combined_cols].notna().all(axis=1)
            if valid_comb.sum() > 50:
                x_comb = df.loc[valid_comb, combined_cols].values
                y_comb = y_binary[valid_comb]
                groups_comb = groups[valid_comb]

                lr_comb = LogisticRegression(
                    penalty=None, max_iter=1000, random_state=seed,
                )
                lr_comb.fit(x_comb, y_comb)
                comb_proba = lr_comb.predict_proba(x_comb)[:, 1]
                comb_auc_ci = auc_roc_with_ci(y_comb, comb_proba)

                try:
                    cv_comb = StratifiedGroupKFold(
                        n_splits=cv_folds, shuffle=True,
                        random_state=seed,
                    )
                    cv_comb_aucs = []
                    for tr_idx, te_idx in cv_comb.split(
                        x_comb, y_comb, groups_comb,
                    ):
                        lr_cv = LogisticRegression(
                            penalty=None, max_iter=1000,
                            random_state=seed,
                        )
                        lr_cv.fit(x_comb[tr_idx], y_comb[tr_idx])
                        p = lr_cv.predict_proba(
                            x_comb[te_idx],
                        )[:, 1]
                        cv_a = auc_roc_with_ci(
                            y_comb[te_idx], p,
                        )
                        cv_comb_aucs.append(cv_a["auc"])
                    cv_comb_mean = float(np.mean(cv_comb_aucs))
                    cv_comb_std = float(np.std(cv_comb_aucs))
                except Exception:
                    cv_comb_mean = float("nan")
                    cv_comb_std = float("nan")
                    cv_comb_aucs = []

                delong_comb = delong_auc_test(
                    y_comb,
                    y_scores[valid_comb],
                    comb_proba,
                )
                baseline_results["combined_model"] = {
                    "features": combined_cols,
                    "auc": comb_auc_ci["auc"],
                    "ci_lower": comb_auc_ci["ci_lower"],
                    "ci_upper": comb_auc_ci["ci_upper"],
                    "cv_mean": cv_comb_mean,
                    "cv_std": cv_comb_std,
                    "cv_fold_aucs": cv_comb_aucs,
                    "n_valid": int(valid_comb.sum()),
                    "vs_ge_delong_z": delong_comb["z_statistic"],
                    "vs_ge_delong_p": delong_comb["p_value"],
                }

        # --- GE score baseline entry (for comparison table) ---
        baseline_results["ge_score"] = {
            "auc": ge_auc_val,
            "ci_lower": h1_auc_ci["ci_lower"],
            "ci_upper": h1_auc_ci["ci_upper"],
            "cv_mean": all_results.get(
                "cross_validation", {},
            ).get("mean_auc"),
            "cv_std": all_results.get(
                "cross_validation", {},
            ).get("std_auc"),
        }

        all_results["baseline_comparisons"] = baseline_results

        # --- Generate baseline comparison figure ---
        plot_labels = []
        plot_aucs = []
        plot_ci_lo = []
        plot_ci_hi = []

        # GE score first
        plot_labels.append("GE Score (graph)")
        plot_aucs.append(ge_auc_val)
        plot_ci_lo.append(h1_auc_ci["ci_lower"])
        plot_ci_hi.append(h1_auc_ci["ci_upper"])

        # Single features
        for _col, info in single_feature_aucs.items():
            plot_labels.append(info["label"])
            plot_aucs.append(info["auc"])
            plot_ci_lo.append(info["ci_lower"])
            plot_ci_hi.append(info["ci_upper"])

        # Dumb baselines
        for model_name in [
            "model_A_merge_rate_age",
            "model_B_merge_rate_age_emb",
        ]:
            if model_name in baseline_results:
                info = baseline_results[model_name]
                short = model_name.replace("model_", "Model ")
                short = short.replace("_merge_rate_age_emb", "")
                short = short.replace("_merge_rate_age", "")
                feat_str = " + ".join(
                    f.replace("log_", "").replace("_", " ")
                    for f in info["features"]
                )
                plot_labels.append(f"LR({feat_str})")
                plot_aucs.append(info["auc"])
                plot_ci_lo.append(info["ci_lower"])
                plot_ci_hi.append(info["ci_upper"])

        # Combined model
        if "combined_model" in baseline_results:
            info = baseline_results["combined_model"]
            plot_labels.append("GE + external features")
            plot_aucs.append(info["auc"])
            plot_ci_lo.append(info["ci_lower"])
            plot_ci_hi.append(info["ci_upper"])

        if len(plot_labels) > 1:
            plot_baseline_comparison(
                plot_labels, plot_aucs, plot_ci_lo, plot_ci_hi,
                ge_auc=ge_auc_val,
                title="AUC-ROC: GE Score vs. Baselines",
                output_path=(
                    figures_dir / "baseline_comparison.png"
                ),
            )

        logger.info(
            "Baseline comparisons: %d single features, %d models",
            len(single_feature_aucs),
            sum(
                1 for k in baseline_results
                if k.startswith("model_")
                or k == "combined_model"
            ),
        )
    except Exception:
        logger.exception("Baseline comparison failed")

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
                penalty=None, max_iter=1000, random_state=seed,
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
