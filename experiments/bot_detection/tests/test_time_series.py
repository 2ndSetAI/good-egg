from __future__ import annotations

from datetime import datetime, timedelta

from experiments.bot_detection.stages.stage6_time_series import (
    compute_time_series_features,
)


class TestComputeTimeSeriesFeatures:
    def test_single_pr_returns_none_for_intervals(self) -> None:
        ts = [datetime(2024, 6, 1, 14, 0)]
        result = compute_time_series_features(ts)
        assert result["inter_pr_cv"] is None
        assert result["inter_pr_median_hours"] is None
        assert result["max_dormancy_days"] is None
        assert result["burst_episode_count"] is None
        assert result["dormancy_burst_ratio"] is None
        assert result["regularity_score"] is None
        # weekend_ratio and hour_entropy still computed
        assert result["weekend_ratio"] is not None
        assert result["hour_entropy"] is not None

    def test_single_pr_weekend(self) -> None:
        # Saturday
        ts = [datetime(2024, 6, 1, 10, 0)]  # 2024-06-01 is a Saturday
        result = compute_time_series_features(ts)
        assert result["weekend_ratio"] == 1.0

    def test_single_pr_weekday(self) -> None:
        # Monday
        ts = [datetime(2024, 6, 3, 10, 0)]  # 2024-06-03 is a Monday
        result = compute_time_series_features(ts)
        assert result["weekend_ratio"] == 0.0

    def test_inter_pr_cv_uniform_spacing(self) -> None:
        # Uniformly spaced: CV should be 0 (no variance)
        base = datetime(2024, 1, 1)
        ts = [base + timedelta(hours=i * 24) for i in range(5)]
        result = compute_time_series_features(ts)
        assert result["inter_pr_cv"] is not None
        assert abs(result["inter_pr_cv"]) < 1e-10

    def test_inter_pr_median_hours(self) -> None:
        base = datetime(2024, 1, 1)
        # Intervals: 2h, 4h, 6h -> median = 4h
        ts = [base, base + timedelta(hours=2), base + timedelta(hours=6),
              base + timedelta(hours=12)]
        result = compute_time_series_features(ts)
        assert result["inter_pr_median_hours"] == 4.0

    def test_max_dormancy_days(self) -> None:
        base = datetime(2024, 1, 1)
        # Intervals: 1h, 48h (2 days), 1h -> max dormancy = 2 days
        ts = [
            base,
            base + timedelta(hours=1),
            base + timedelta(hours=49),
            base + timedelta(hours=50),
        ]
        result = compute_time_series_features(ts)
        assert result["max_dormancy_days"] == 2.0

    def test_burst_episode_count(self) -> None:
        base = datetime(2024, 1, 1)
        # Burst 1: 3 PRs within 24h, then 10-day gap, then burst 2: 2 PRs within 24h
        ts = [
            base,
            base + timedelta(hours=1),
            base + timedelta(hours=2),
            base + timedelta(days=10),
            base + timedelta(days=10, hours=1),
        ]
        result = compute_time_series_features(ts, burst_gap_days=7)
        assert result["burst_episode_count"] == 2

    def test_burst_episode_no_bursts(self) -> None:
        base = datetime(2024, 1, 1)
        # All PRs spaced > 24h apart
        ts = [base + timedelta(days=i * 3) for i in range(4)]
        result = compute_time_series_features(ts, burst_gap_days=7)
        assert result["burst_episode_count"] == 0

    def test_weekend_ratio_mixed(self) -> None:
        # 2 weekend + 2 weekday = 0.5
        ts = [
            datetime(2024, 6, 1, 10, 0),   # Saturday
            datetime(2024, 6, 2, 10, 0),   # Sunday
            datetime(2024, 6, 3, 10, 0),   # Monday
            datetime(2024, 6, 4, 10, 0),   # Tuesday
        ]
        result = compute_time_series_features(ts)
        assert result["weekend_ratio"] == 0.5

    def test_hour_entropy_non_negative(self) -> None:
        base = datetime(2024, 1, 1)
        ts = [base + timedelta(hours=i) for i in range(10)]
        result = compute_time_series_features(ts)
        assert result["hour_entropy"] is not None
        assert result["hour_entropy"] >= 0

    def test_hour_entropy_single_hour(self) -> None:
        # All same hour -> entropy = 0
        ts = [datetime(2024, 1, d, 14, 0) for d in range(1, 6)]
        result = compute_time_series_features(ts)
        assert result["hour_entropy"] == 0.0

    def test_dormancy_burst_ratio(self) -> None:
        base = datetime(2024, 1, 1)
        # Intervals: 24h, 24h, 24h -> median=24h, max_dormancy=1 day
        ts = [base + timedelta(hours=i * 24) for i in range(4)]
        result = compute_time_series_features(ts)
        # max_dormancy_days = 1.0, inter_pr_median_hours = 24.0
        # ratio = 1.0 / (24.0 / 24.0) = 1.0
        assert result["dormancy_burst_ratio"] == 1.0

    def test_regularity_score_returns_float_or_none(self) -> None:
        base = datetime(2024, 1, 1)
        ts = [base + timedelta(hours=i * 12) for i in range(6)]
        result = compute_time_series_features(ts)
        # With uniform intervals, regularity is undefined (0 variance) -> could be None or NaN
        # Just check it's handled
        score = result["regularity_score"]
        assert score is None or isinstance(score, float)

    def test_empty_timestamps(self) -> None:
        result = compute_time_series_features([])
        assert result["weekend_ratio"] is None
        assert result["hour_entropy"] is None
        assert result["inter_pr_cv"] is None
