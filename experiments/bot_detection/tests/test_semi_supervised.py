from __future__ import annotations

import numpy as np

from experiments.bot_detection.stages.stage6_semi_supervised import (
    compute_isolation_forest_scores,
    compute_knn_distances,
)


class TestKnnDistances:
    def test_seeds_get_zero_distance(self) -> None:
        feats = np.array([[0, 0], [1, 1], [10, 10], [11, 11]], dtype=float)
        seed_mask = np.array([True, True, False, False])
        distances = compute_knn_distances(feats, seed_mask, k=2)
        assert distances[0] == 0.0
        assert distances[1] == 0.0

    def test_far_points_have_larger_distance(self) -> None:
        feats = np.array([[0, 0], [1, 0], [100, 100]], dtype=float)
        seed_mask = np.array([True, False, False])
        distances = compute_knn_distances(feats, seed_mask, k=1)
        # Point at [1,0] is closer to seed than [100,100]
        assert distances[1] < distances[2]

    def test_no_seeds_returns_nan(self) -> None:
        feats = np.array([[0, 0], [1, 1]], dtype=float)
        seed_mask = np.array([False, False])
        distances = compute_knn_distances(feats, seed_mask, k=1)
        assert np.all(np.isnan(distances))

    def test_k_larger_than_seeds(self) -> None:
        # k=5 but only 2 seeds -- should still work with effective_k=2
        feats = np.array([[0, 0], [1, 0], [5, 5], [10, 10]], dtype=float)
        seed_mask = np.array([True, True, False, False])
        distances = compute_knn_distances(feats, seed_mask, k=5)
        assert distances[0] == 0.0
        assert distances[1] == 0.0
        assert distances[2] > 0.0
        assert distances[3] > 0.0

    def test_single_seed(self) -> None:
        feats = np.array([[0, 0], [3, 4]], dtype=float)
        seed_mask = np.array([True, False])
        distances = compute_knn_distances(feats, seed_mask, k=1)
        assert distances[0] == 0.0
        assert abs(distances[1] - 5.0) < 1e-10  # 3-4-5 triangle


class TestIsolationForest:
    def test_returns_scores_for_all_rows(self) -> None:
        rng = np.random.default_rng(42)
        feats = rng.standard_normal((100, 5))
        scores = compute_isolation_forest_scores(feats, contamination=0.05, random_state=42)
        assert len(scores) == 100

    def test_outlier_has_lower_score(self) -> None:
        # Cluster of normal points + one extreme outlier
        rng = np.random.default_rng(42)
        normal = rng.standard_normal((50, 2))
        outlier = np.array([[100, 100]])
        feats = np.vstack([normal, outlier])
        scores = compute_isolation_forest_scores(feats, contamination=0.05, random_state=42)
        # The outlier (last row) should have the lowest score
        assert scores[-1] < np.median(scores)

    def test_scores_are_finite(self) -> None:
        rng = np.random.default_rng(42)
        feats = rng.standard_normal((30, 3))
        scores = compute_isolation_forest_scores(feats, contamination=0.1, random_state=42)
        assert np.all(np.isfinite(scores))
