from __future__ import annotations

from datetime import UTC, datetime

from experiments.validation.models import (
    AuthorRecord,
    ClassifiedPR,
    CollectedPR,
    FeatureRow,
    PROutcome,
    ScoredPR,
    StatisticalResult,
    StudyConfig,
    TemporalBin,
)


def test_collected_pr_round_trip() -> None:
    pr = CollectedPR(
        repo="owner/repo",
        number=42,
        author_login="alice",
        title="Fix bug",
        state="MERGED",
        created_at=datetime(2024, 6, 1, tzinfo=UTC),
        merged_at=datetime(2024, 6, 2, tzinfo=UTC),
        temporal_bin="2024H1",
    )
    data = pr.model_dump(mode="json")
    restored = CollectedPR(**data)
    assert restored.repo == pr.repo
    assert restored.number == pr.number
    assert restored.author_login == pr.author_login


def test_classified_pr_round_trip() -> None:
    pr = ClassifiedPR(
        repo="owner/repo",
        number=42,
        author_login="alice",
        title="Fix bug",
        state="CLOSED",
        created_at=datetime(2024, 6, 1, tzinfo=UTC),
        closed_at=datetime(2024, 7, 1, tzinfo=UTC),
        temporal_bin="2024H1",
        outcome=PROutcome.REJECTED,
        stale_threshold_days=60.0,
    )
    data = pr.model_dump(mode="json")
    restored = ClassifiedPR(**data)
    assert restored.outcome == PROutcome.REJECTED
    assert restored.stale_threshold_days == 60.0


def test_author_record_defaults() -> None:
    rec = AuthorRecord(login="bob")
    assert rec.merged_count is None
    assert rec.tier2_sampled is False
    data = rec.model_dump(mode="json")
    restored = AuthorRecord(**data)
    assert restored.login == "bob"


def test_scored_pr_with_ablations() -> None:
    pr = ScoredPR(
        repo="owner/repo",
        pr_number=1,
        author_login="alice",
        outcome=PROutcome.MERGED,
        temporal_bin="2024H1",
        created_at=datetime(2024, 6, 1, tzinfo=UTC),
        normalized_score=0.75,
        ablation_scores={"no_recency": 0.70, "no_quality": 0.65},
    )
    data = pr.model_dump(mode="json")
    restored = ScoredPR(**data)
    assert restored.ablation_scores["no_recency"] == 0.70


def test_feature_row_optional_fields() -> None:
    row = FeatureRow(
        repo="owner/repo",
        pr_number=1,
        author_login="alice",
        outcome=PROutcome.MERGED,
        temporal_bin="2024H1",
        created_at=datetime(2024, 6, 1, tzinfo=UTC),
    )
    assert row.embedding_similarity is None
    assert row.author_merge_rate is None


def test_statistical_result_round_trip() -> None:
    result = StatisticalResult(
        test_name="delong",
        statistic=2.5,
        p_value=0.01,
        details={"auc_a": 0.75, "auc_b": 0.70},
    )
    data = result.model_dump(mode="json")
    restored = StatisticalResult(**data)
    assert restored.test_name == "delong"
    assert restored.details["auc_a"] == 0.75


def test_study_config_defaults() -> None:
    cfg = StudyConfig()
    assert cfg.stale_threshold_bin == "2024H1"
    assert cfg.temporal_bins == []


def test_temporal_bin_round_trip() -> None:
    tb = TemporalBin(
        label="2024H1", start="2024-01-01", end="2024-06-30",
    )
    data = tb.model_dump(mode="json")
    restored = TemporalBin(**data)
    assert restored.label == "2024H1"


def test_collected_pr_labels_default() -> None:
    """Labels field defaults to empty list."""
    pr = CollectedPR(
        repo="owner/repo",
        number=1,
        author_login="alice",
        title="Test",
        state="OPEN",
        created_at=datetime(2024, 6, 1, tzinfo=UTC),
        temporal_bin="2024H1",
    )
    assert pr.labels == []


def test_collected_pr_labels_round_trip() -> None:
    """Labels survive serialization round trip."""
    pr = CollectedPR(
        repo="owner/repo",
        number=1,
        author_login="alice",
        title="Test",
        state="CLOSED",
        created_at=datetime(2024, 6, 1, tzinfo=UTC),
        closed_at=datetime(2024, 6, 2, tzinfo=UTC),
        labels=["auto-merge", "bug"],
        temporal_bin="2024H1",
    )
    data = pr.model_dump(mode="json")
    restored = CollectedPR(**data)
    assert restored.labels == ["auto-merge", "bug"]


def test_classified_pr_labels_default() -> None:
    """ClassifiedPR labels field defaults to empty list."""
    pr = ClassifiedPR(
        repo="owner/repo",
        number=1,
        author_login="alice",
        title="Test",
        state="CLOSED",
        created_at=datetime(2024, 6, 1, tzinfo=UTC),
        temporal_bin="2024H1",
        outcome=PROutcome.REJECTED,
        stale_threshold_days=60.0,
    )
    assert pr.labels == []


def test_classified_pr_labels_from_collected() -> None:
    """ClassifiedPR preserves labels when built from CollectedPR dump."""
    collected = CollectedPR(
        repo="owner/repo",
        number=1,
        author_login="alice",
        title="Test",
        state="CLOSED",
        created_at=datetime(2024, 6, 1, tzinfo=UTC),
        closed_at=datetime(2024, 6, 2, tzinfo=UTC),
        labels=["merged", "bors"],
        temporal_bin="2024H1",
    )
    classified = ClassifiedPR(
        **collected.model_dump(),
        outcome=PROutcome.MERGED,
        stale_threshold_days=60.0,
    )
    assert classified.labels == ["merged", "bors"]
