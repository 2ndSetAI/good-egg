from __future__ import annotations

import numpy as np
import pytest

from experiments.bot_detection.stages.stage7_author_evaluate import (
    _precision_at_k as precision_at_k,
)


class TestPrecisionAtK:
    def test_perfect_scores(self) -> None:
        """Perfect scoring -> precision@k = 1.0 for k <= num positives."""
        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0])
        assert precision_at_k(y_true, scores, k=3) == 1.0
        assert precision_at_k(y_true, scores, k=1) == 1.0

    def test_random_scores_approximate_base_rate(self) -> None:
        """Random scoring -> precision@k roughly equals base rate for large N."""
        rng = np.random.RandomState(42)
        n = 10000
        base_rate = 0.1
        y_true = (rng.random(n) < base_rate).astype(int)
        scores = rng.random(n)

        k = 500
        p = precision_at_k(y_true, scores, k)
        # Should be close to base_rate with some tolerance
        assert abs(p - base_rate) < 0.05

    def test_inverse_scores(self) -> None:
        """Worst-case scoring: positives get lowest scores -> precision@k = 0."""
        y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0])
        scores = np.array([0.0, 0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        assert precision_at_k(y_true, scores, k=4) == 0.0

    def test_k_equals_n(self) -> None:
        """precision@N equals the base rate."""
        y_true = np.array([1, 0, 1, 0, 0])
        scores = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
        assert precision_at_k(y_true, scores, k=5) == pytest.approx(0.4)


class TestLiftOverRandom:
    def test_good_model_has_positive_lift(self) -> None:
        """A model better than random should have lift > 1."""
        rng = np.random.RandomState(123)
        n = 5000
        base_rate = 0.05
        y_true = (rng.random(n) < base_rate).astype(int)
        # Give positives higher scores
        scores = y_true * 0.5 + rng.random(n) * 0.3

        k = 250
        p = precision_at_k(y_true, scores, k)
        lift = p / base_rate
        assert lift > 1.5

    def test_random_model_lift_near_one(self) -> None:
        """Random scores -> lift approximately 1."""
        rng = np.random.RandomState(42)
        n = 10000
        base_rate = 0.1
        y_true = (rng.random(n) < base_rate).astype(int)
        scores = rng.random(n)

        k = 500
        p = precision_at_k(y_true, scores, k)
        lift = p / base_rate
        assert 0.7 < lift < 1.4
