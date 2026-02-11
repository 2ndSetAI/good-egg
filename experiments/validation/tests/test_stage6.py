from __future__ import annotations

import numpy as np
import pytest

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
