from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import stats as sp_stats
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def delong_auc_test(
    y_true: np.ndarray,
    y_scores_a: np.ndarray,
    y_scores_b: np.ndarray,
) -> dict[str, Any]:
    """Paired DeLong test for comparing two AUC-ROC values.

    Uses the method from DeLong et al. (1988) to test whether two
    correlated ROC curves have significantly different AUCs.

    Returns dict with keys: auc_a, auc_b, z_statistic, p_value.
    """
    n1 = np.sum(y_true == 1)
    n0 = np.sum(y_true == 0)

    if n1 == 0 or n0 == 0:
        return {
            "auc_a": float("nan"),
            "auc_b": float("nan"),
            "z_statistic": float("nan"),
            "p_value": float("nan"),
        }

    # Compute AUCs
    auc_a = roc_auc_score(y_true, y_scores_a)
    auc_b = roc_auc_score(y_true, y_scores_b)

    # Placement values for each model
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    def placement_values(
        scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute placement values for positive and negative samples."""
        pos_scores = scores[pos_idx]
        neg_scores = scores[neg_idx]
        # For each positive, fraction of negatives it exceeds
        v10 = np.array([
            np.mean(ps > neg_scores) + 0.5 * np.mean(ps == neg_scores)
            for ps in pos_scores
        ])
        # For each negative, fraction of positives that exceed it
        v01 = np.array([
            np.mean(pos_scores > ns) + 0.5 * np.mean(pos_scores == ns)
            for ns in neg_scores
        ])
        return v10, v01

    v10_a, v01_a = placement_values(y_scores_a)
    v10_b, v01_b = placement_values(y_scores_b)

    # Covariance matrix of the two AUCs
    s10 = np.cov(np.stack([v10_a, v10_b]))
    s01 = np.cov(np.stack([v01_a, v01_b]))

    # Handle scalar case (single observation)
    if s10.ndim == 0:
        s10 = np.array([[s10]])
    if s01.ndim == 0:
        s01 = np.array([[s01]])

    s = s10 / n1 + s01 / n0

    # Variance of AUC difference
    contrast = np.array([1, -1])
    var_diff = contrast @ s @ contrast

    if var_diff <= 0:
        return {
            "auc_a": auc_a,
            "auc_b": auc_b,
            "z_statistic": 0.0,
            "p_value": 1.0,
        }

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p_value = 2.0 * sp_stats.norm.sf(abs(z))

    return {
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "z_statistic": float(z),
        "p_value": float(p_value),
    }


def auc_roc_with_ci(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Compute AUC-ROC with DeLong confidence interval.

    Returns dict with keys: auc, ci_lower, ci_upper, se.
    """
    n1 = np.sum(y_true == 1)
    n0 = np.sum(y_true == 0)

    if n1 == 0 or n0 == 0:
        return {
            "auc": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "se": float("nan"),
        }

    auc = roc_auc_score(y_true, y_scores)

    # DeLong variance estimate
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    pos_scores = y_scores[pos_idx]
    neg_scores = y_scores[neg_idx]

    v10 = np.array([
        np.mean(ps > neg_scores) + 0.5 * np.mean(ps == neg_scores)
        for ps in pos_scores
    ])
    v01 = np.array([
        np.mean(pos_scores > ns) + 0.5 * np.mean(pos_scores == ns)
        for ns in neg_scores
    ])

    var_auc = np.var(v10) / n1 + np.var(v01) / n0
    se = np.sqrt(var_auc)

    z_crit = sp_stats.norm.ppf(1 - alpha / 2)
    ci_lower = max(0.0, auc - z_crit * se)
    ci_upper = min(1.0, auc + z_crit * se)

    return {
        "auc": float(auc),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "se": float(se),
    }


def holm_bonferroni(
    p_values: dict[str, float],
    alpha: float = 0.05,
) -> dict[str, dict[str, Any]]:
    """Apply Holm-Bonferroni correction to a set of p-values.

    Returns dict mapping test name to {p_value, adjusted_p, reject}.
    """
    sorted_tests = sorted(p_values.items(), key=lambda x: x[1])
    m = len(sorted_tests)
    results: dict[str, dict[str, Any]] = {}

    for rank, (name, p) in enumerate(sorted_tests, start=1):
        adjusted_alpha = alpha / (m - rank + 1)
        adjusted_p = min(1.0, p * (m - rank + 1))
        results[name] = {
            "p_value": p,
            "adjusted_p": adjusted_p,
            "reject": p <= adjusted_alpha,
            "rank": rank,
        }

    return results


def likelihood_ratio_test(
    ll_null: float,
    ll_alt: float,
    df_diff: int,
) -> dict[str, float]:
    """Likelihood ratio test comparing nested models.

    Parameters
    ----------
    ll_null : Log-likelihood of the null (restricted) model.
    ll_alt : Log-likelihood of the alternative (full) model.
    df_diff : Difference in degrees of freedom.

    Returns dict with keys: lr_statistic, p_value, df.
    """
    lr_stat = -2 * (ll_null - ll_alt)
    lr_stat = max(0.0, lr_stat)  # Numerical safety
    p_value = float(sp_stats.chi2.sf(lr_stat, df_diff))
    return {
        "lr_statistic": float(lr_stat),
        "p_value": p_value,
        "df": df_diff,
    }


def kruskal_wallis_with_dunn(
    groups: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Kruskal-Wallis test with Dunn's post-hoc pairwise comparisons.

    Parameters
    ----------
    groups : dict mapping group name to array of values.

    Returns dict with keys: h_statistic, p_value, posthoc (dict of
    pairwise comparisons).
    """
    group_names = list(groups.keys())
    group_arrays = [groups[k] for k in group_names]

    if len(group_arrays) < 2:
        return {
            "h_statistic": float("nan"),
            "p_value": float("nan"),
            "posthoc": {},
        }

    h_stat, p_value = sp_stats.kruskal(*group_arrays)

    # Dunn's post-hoc pairwise comparisons (Bonferroni corrected)
    posthoc: dict[str, dict[str, float]] = {}
    n_comparisons = len(group_names) * (len(group_names) - 1) // 2

    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            name = f"{group_names[i]}_vs_{group_names[j]}"
            u_stat, u_p = sp_stats.mannwhitneyu(
                group_arrays[i], group_arrays[j],
                alternative="two-sided",
            )
            adjusted_p = min(1.0, u_p * n_comparisons)
            posthoc[name] = {
                "u_statistic": float(u_stat),
                "p_value": float(u_p),
                "adjusted_p": float(adjusted_p),
            }

    return {
        "h_statistic": float(h_stat),
        "p_value": float(p_value),
        "posthoc": posthoc,
    }


def chi_squared_test(
    contingency_table: np.ndarray,
) -> dict[str, float]:
    """Chi-squared test of independence on a contingency table.

    Returns dict with keys: chi2, p_value, dof, cramers_v.
    """
    chi2, p_value, dof, _ = sp_stats.chi2_contingency(contingency_table)
    n = contingency_table.sum()
    k = min(contingency_table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k)) if n * k > 0 else 0.0

    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "dof": int(dof),
        "cramers_v": float(cramers_v),
    }


def cochran_armitage_trend(
    table: np.ndarray,
    scores: np.ndarray | None = None,
) -> dict[str, float]:
    """Cochran-Armitage test for trend in a 2xK table.

    Parameters
    ----------
    table : 2xK array where rows are outcome (0/1) and columns are
        ordered categories.
    scores : Optional scores for each column. Defaults to 0, 1, ..., K-1.

    Returns dict with keys: z_statistic, p_value.
    """
    k = table.shape[1]
    if scores is None:
        scores = np.arange(k, dtype=float)

    n = table.sum()
    col_totals = table.sum(axis=0)
    row1_totals = table[0]

    p_hat = table[0].sum() / n
    t_num = np.sum(scores * (row1_totals - col_totals * p_hat))
    t_den_sq = p_hat * (1 - p_hat) * (
        n * np.sum(scores**2 * col_totals)
        - np.sum(scores * col_totals) ** 2
    )

    if t_den_sq <= 0:
        return {"z_statistic": 0.0, "p_value": 1.0}

    z = t_num / np.sqrt(t_den_sq)
    p_value = 2.0 * sp_stats.norm.sf(abs(z))

    return {"z_statistic": float(z), "p_value": float(p_value)}


def compute_binary_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> dict[str, float]:
    """Compute all binary classification metrics.

    Returns dict with keys: auc_roc, auc_pr, brier_score, log_loss_val.
    """
    from sklearn.metrics import average_precision_score

    auc_roc = float(roc_auc_score(y_true, y_scores))
    auc_pr = float(average_precision_score(y_true, y_scores))
    brier = float(brier_score_loss(y_true, y_scores))

    # Clip scores for log_loss stability
    clipped = np.clip(y_scores, 1e-15, 1 - 1e-15)
    ll = float(log_loss(y_true, clipped))

    return {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "brier_score": brier,
        "log_loss": ll,
    }


def odds_ratio(
    contingency_2x2: np.ndarray,
) -> dict[str, float]:
    """Compute odds ratio and CI from a 2x2 contingency table.

    Returns dict with keys: odds_ratio, ci_lower, ci_upper, p_value.
    """
    a, b = contingency_2x2[0]
    c, d = contingency_2x2[1]

    if b == 0 or c == 0 or d == 0 or a == 0:
        # Apply Haldane correction
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5

    or_val = (a * d) / (b * c)
    log_or = np.log(or_val)
    se_log_or = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    z_crit = sp_stats.norm.ppf(0.975)

    ci_lower = np.exp(log_or - z_crit * se_log_or)
    ci_upper = np.exp(log_or + z_crit * se_log_or)

    z = log_or / se_log_or
    p_value = 2.0 * sp_stats.norm.sf(abs(z))

    return {
        "odds_ratio": float(or_val),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_value": float(p_value),
    }
