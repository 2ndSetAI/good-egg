from __future__ import annotations

import numpy as np
import pytest

from experiments.validation.models import PROutcome
from experiments.validation.stages.stage6_analyze import (
    _binary_target,
    _trust_level_bin,
    log_loss_manual,
)
from experiments.validation.stats import (
    auc_roc_with_ci,
    chi_squared_test,
    compute_binary_metrics,
    holm_bonferroni,
    likelihood_ratio_test,
    odds_ratio,
)


def test_binary_metrics_perfect() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    metrics = compute_binary_metrics(y_true, y_scores)
    assert metrics["auc_roc"] == 1.0
    assert metrics["auc_pr"] == 1.0


def test_binary_metrics_random() -> None:
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, 100)
    y_scores = rng.rand(100)
    metrics = compute_binary_metrics(y_true, y_scores)
    assert 0.0 <= metrics["auc_roc"] <= 1.0
    assert metrics["brier_score"] >= 0.0


def test_auc_roc_with_ci() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_scores = np.array(
        [0.1, 0.3, 0.6, 0.8, 0.2, 0.7, 0.4, 0.9],
    )
    result = auc_roc_with_ci(y_true, y_scores)
    assert 0.0 <= result["ci_lower"] <= result["auc"]
    assert result["auc"] <= result["ci_upper"] <= 1.0


def test_holm_bonferroni_correction() -> None:
    p_values = {
        "test_a": 0.01, "test_b": 0.04, "test_c": 0.06,
    }
    results = holm_bonferroni(p_values, alpha=0.05)
    # test_a: p=0.01, adjusted=0.03, reject
    assert results["test_a"]["reject"]
    # test_c: p=0.06, adjusted=0.06, no reject
    assert not results["test_c"]["reject"]


def test_likelihood_ratio_test() -> None:
    result = likelihood_ratio_test(
        ll_null=-100.0, ll_alt=-90.0, df_diff=1,
    )
    assert result["lr_statistic"] == pytest.approx(20.0)
    assert result["p_value"] < 0.001


def test_chi_squared_test() -> None:
    table = np.array([[30, 10], [20, 40]])
    result = chi_squared_test(table)
    assert result["chi2"] > 0
    assert result["p_value"] < 0.05
    assert result["dof"] == 1


def test_odds_ratio_basic() -> None:
    table = np.array([[50, 10], [20, 40]])
    result = odds_ratio(table)
    assert result["odds_ratio"] > 1.0
    assert result["ci_lower"] > 0
    assert result["p_value"] < 0.05


def test_odds_ratio_with_zero_cell() -> None:
    table = np.array([[10, 0], [5, 10]])
    result = odds_ratio(table)
    # Should apply Haldane correction
    assert result["odds_ratio"] > 0


# === Stage 6 helper function tests ===


def test_binary_target_merged() -> None:
    assert _binary_target(PROutcome.MERGED.value) == 1


def test_binary_target_rejected() -> None:
    assert _binary_target(PROutcome.REJECTED.value) == 0


def test_binary_target_pocket_veto() -> None:
    assert _binary_target(PROutcome.POCKET_VETO.value) == 0


def test_trust_level_bin_high() -> None:
    thresholds = {"HIGH": 0.7, "MEDIUM": 0.3}
    assert _trust_level_bin(0.8, thresholds) == "HIGH"
    assert _trust_level_bin(0.7, thresholds) == "HIGH"


def test_trust_level_bin_medium() -> None:
    thresholds = {"HIGH": 0.7, "MEDIUM": 0.3}
    assert _trust_level_bin(0.5, thresholds) == "MEDIUM"
    assert _trust_level_bin(0.3, thresholds) == "MEDIUM"


def test_trust_level_bin_low() -> None:
    thresholds = {"HIGH": 0.7, "MEDIUM": 0.3}
    assert _trust_level_bin(0.1, thresholds) == "LOW"
    assert _trust_level_bin(0.0, thresholds) == "LOW"
    assert _trust_level_bin(0.29, thresholds) == "LOW"


def test_trust_level_bin_default_thresholds() -> None:
    """Empty thresholds fall back to defaults (HIGH=0.7, MEDIUM=0.3)."""
    assert _trust_level_bin(0.8, {}) == "HIGH"
    assert _trust_level_bin(0.5, {}) == "MEDIUM"
    assert _trust_level_bin(0.1, {}) == "LOW"


def test_log_loss_manual_perfect() -> None:
    y_true = np.array([0.0, 0.0, 1.0, 1.0])
    y_pred = np.array([0.01, 0.01, 0.99, 0.99])
    loss = log_loss_manual(y_true, y_pred)
    assert loss < 0.05


def test_log_loss_manual_random() -> None:
    y_true = np.array([0.0, 1.0, 0.0, 1.0])
    y_pred = np.array([0.5, 0.5, 0.5, 0.5])
    loss = log_loss_manual(y_true, y_pred)
    assert loss == pytest.approx(np.log(2), rel=1e-5)


# === Newcomer cohort analysis tests (Rec 3) ===


def test_newcomer_cohort_metrics_with_perfect_separation() -> None:
    """Newcomer cohort with perfect score separation has AUC=1."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    metrics = compute_binary_metrics(y_true, y_scores)
    assert metrics["auc_roc"] == 1.0


def test_newcomer_cohort_subset_auc() -> None:
    """Can compute AUC on a newcomer subset of the data."""
    # Simulate: 10 established, 10 newcomers
    is_newcomer = np.array([0] * 10 + [1] * 10)
    y_true = np.array(
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # established
        + [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # newcomers
    )
    y_scores = np.array(
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
        + [0.6, 0.55, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.05]
    )

    # Newcomer subset
    newcomer_mask = is_newcomer == 1
    newcomer_y = y_true[newcomer_mask]
    newcomer_scores = y_scores[newcomer_mask]

    assert len(newcomer_y) == 10
    assert newcomer_y.sum() == 2  # 2 merged

    metrics = compute_binary_metrics(newcomer_y, newcomer_scores)
    assert 0.0 <= metrics["auc_roc"] <= 1.0

    # Established subset
    established_mask = is_newcomer == 0
    est_y = y_true[established_mask]
    est_scores = y_scores[established_mask]

    est_metrics = compute_binary_metrics(est_y, est_scores)
    assert 0.0 <= est_metrics["auc_roc"] <= 1.0
