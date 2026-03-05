from __future__ import annotations

import numpy as np

from experiments.bot_detection.stats import (
    auc_roc_with_ci,
    delong_auc_test,
    holm_bonferroni,
    likelihood_ratio_test,
)


class TestHolmBonferroni:
    def test_basic_correction(self) -> None:
        p_values = {"test_a": 0.01, "test_b": 0.04, "test_c": 0.06}
        results = holm_bonferroni(p_values, alpha=0.05)
        # test_a: 0.01 * 3 = 0.03 -> reject
        assert results["test_a"]["reject"] is True
        # test_b: 0.04 * 2 = 0.08 -> do not reject
        assert results["test_b"]["reject"] is False
        # test_c: 0.06 * 1 = 0.06 -> do not reject
        assert results["test_c"]["reject"] is False

    def test_all_significant(self) -> None:
        p_values = {"a": 0.001, "b": 0.005, "c": 0.01}
        results = holm_bonferroni(p_values, alpha=0.05)
        assert all(r["reject"] for r in results.values())

    def test_monotonicity(self) -> None:
        p_values = {"a": 0.01, "b": 0.03, "c": 0.02}
        results = holm_bonferroni(p_values, alpha=0.05)
        sorted_adj = sorted(results.values(), key=lambda x: x["rank"])
        for i in range(1, len(sorted_adj)):
            assert sorted_adj[i]["adjusted_p"] >= sorted_adj[i - 1]["adjusted_p"]


class TestDeLongAUC:
    def test_identical_models(self) -> None:
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=100)
        scores = rng.random(100)
        result = delong_auc_test(y_true, scores, scores)
        assert abs(result["z_statistic"]) < 0.01
        assert result["p_value"] > 0.99

    def test_different_models(self) -> None:
        rng = np.random.RandomState(42)
        y_true = np.array([0] * 50 + [1] * 50)
        good_scores = np.concatenate([rng.uniform(0, 0.4, 50), rng.uniform(0.6, 1, 50)])
        bad_scores = rng.random(100)
        result = delong_auc_test(y_true, good_scores, bad_scores)
        assert result["auc_a"] > result["auc_b"]


class TestAUCWithCI:
    def test_perfect_classifier(self) -> None:
        y_true = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        result = auc_roc_with_ci(y_true, scores)
        assert result["auc"] == 1.0
        assert result["ci_lower"] > 0.5

    def test_random_classifier(self) -> None:
        rng = np.random.RandomState(42)
        y_true = np.array([0] * 500 + [1] * 500)
        scores = rng.random(1000)
        result = auc_roc_with_ci(y_true, scores)
        assert 0.4 < result["auc"] < 0.6


class TestLRT:
    def test_nested_models(self) -> None:
        # Alt model should have better (less negative) log-likelihood
        result = likelihood_ratio_test(ll_null=-100, ll_alt=-90, df_diff=2)
        assert result["lr_statistic"] == 20.0
        assert result["p_value"] < 0.001

    def test_equal_models(self) -> None:
        result = likelihood_ratio_test(ll_null=-100, ll_alt=-100, df_diff=1)
        assert result["lr_statistic"] == 0.0
        assert result["p_value"] == 1.0
