from __future__ import annotations

from datetime import datetime

from experiments.bot_detection.models import (
    BotDetectionPR,
    BurstinessFeatures,
    CrossRepoFeatures,
    EngagementFeatures,
    FeatureRow,
    PROutcome,
    StageCheckpoint,
)


class TestPROutcome:
    def test_values(self) -> None:
        assert PROutcome.MERGED == "merged"
        assert PROutcome.REJECTED == "rejected"
        assert PROutcome.POCKET_VETO == "pocket_veto"

    def test_from_string(self) -> None:
        assert PROutcome("merged") == PROutcome.MERGED
        assert PROutcome("rejected") == PROutcome.REJECTED


class TestBotDetectionPR:
    def test_minimal(self) -> None:
        pr = BotDetectionPR(
            repo="owner/repo",
            number=1,
            author="user",
            title="Fix bug",
            created_at=datetime(2024, 1, 1),
            state="MERGED",
        )
        assert pr.repo == "owner/repo"
        assert pr.number == 1
        assert pr.labels == []

    def test_with_outcome(self) -> None:
        pr = BotDetectionPR(
            repo="owner/repo",
            number=1,
            author="user",
            title="Fix bug",
            created_at=datetime(2024, 1, 1),
            state="CLOSED",
            outcome=PROutcome.REJECTED,
            stale_threshold_days=45.0,
        )
        assert pr.outcome == PROutcome.REJECTED


class TestFeatureRow:
    def test_all_features(self) -> None:
        row = FeatureRow(
            repo="owner/repo",
            number=1,
            author="user",
            outcome=PROutcome.MERGED,
            created_at=datetime(2024, 6, 1),
            burst_count_1h=5,
            burst_repos_1h=3,
            burst_count_24h=20,
            burst_repos_24h=10,
            burst_max_rate=8.0,
            review_response_rate=0.75,
            abandoned_pr_rate=0.1,
            max_title_similarity=0.85,
            language_entropy=2.3,
        )
        assert row.burst_count_1h == 5
        assert row.review_response_rate == 0.75
        assert row.max_title_similarity == 0.85

    def test_defaults(self) -> None:
        row = FeatureRow(
            repo="a/b",
            number=1,
            author="u",
            outcome=PROutcome.MERGED,
            created_at=datetime(2024, 1, 1),
        )
        assert row.burst_count_1h == 0
        assert row.review_response_rate is None
        assert row.ge_score is None

    def test_serialization(self) -> None:
        row = FeatureRow(
            repo="a/b",
            number=1,
            author="u",
            outcome=PROutcome.MERGED,
            created_at=datetime(2024, 1, 1),
        )
        d = row.model_dump()
        assert d["repo"] == "a/b"
        assert d["outcome"] == "merged"


class TestBurstinessFeatures:
    def test_defaults(self) -> None:
        bf = BurstinessFeatures()
        assert bf.burst_count_1h == 0
        assert bf.burst_max_rate == 0.0


class TestEngagementFeatures:
    def test_defaults(self) -> None:
        ef = EngagementFeatures()
        assert ef.review_response_rate is None
        assert ef.abandoned_pr_rate is None


class TestCrossRepoFeatures:
    def test_defaults(self) -> None:
        cf = CrossRepoFeatures()
        assert cf.max_title_similarity == 0.0
        assert cf.duplicate_title_count == 0


class TestStageCheckpoint:
    def test_creation(self) -> None:
        cp = StageCheckpoint(
            stage="stage1",
            timestamp=datetime(2024, 1, 1),
            row_counts={"prs": 100, "reviews": 500},
        )
        assert cp.stage == "stage1"
        assert cp.row_counts["prs"] == 100
