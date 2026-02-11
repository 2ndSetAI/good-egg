from __future__ import annotations

from datetime import UTC, datetime, timedelta

from experiments.validation.models import CollectedPR, PROutcome
from experiments.validation.stages.stage2_discover_authors import (
    _classify_pr,
    _compute_stale_threshold,
    _is_bot,
)


def _make_pr(
    merged_at: datetime | None = None,
    closed_at: datetime | None = None,
    created_at: datetime | None = None,
) -> CollectedPR:
    if created_at is None:
        created_at = datetime(2024, 6, 1, tzinfo=UTC)
    return CollectedPR(
        repo="owner/repo",
        number=1,
        author_login="alice",
        title="Test PR",
        state=(
            "MERGED" if merged_at
            else "CLOSED" if closed_at
            else "OPEN"
        ),
        created_at=created_at,
        merged_at=merged_at,
        closed_at=closed_at,
        temporal_bin="2024H1",
    )


def test_is_bot_suffix() -> None:
    assert _is_bot("dependabot[bot]", [])
    assert _is_bot("renovate-bot", [])
    assert _is_bot("my_bot", [])


def test_is_bot_prefix() -> None:
    assert _is_bot("dependabot", [])
    assert _is_bot("renovate", [])
    assert _is_bot("github-actions", [])


def test_is_not_bot() -> None:
    assert not _is_bot("alice", [])
    assert not _is_bot("bob-smith", [])


def test_is_bot_extra_patterns() -> None:
    assert _is_bot("custom-ci-bot", ["custom-ci"])
    assert not _is_bot("alice", ["custom-ci"])


def test_stale_threshold_basic() -> None:
    prs = [
        _make_pr(
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
            merged_at=(
                datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=d)
            ),
        )
        for d in [2, 3, 5, 7, 10]
    ]
    # median TTM = 5 days, threshold = max(30, min(5*5, 180)) = 30
    threshold = _compute_stale_threshold(prs)
    assert threshold == 30.0


def test_stale_threshold_long_ttm() -> None:
    prs = [
        _make_pr(
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
            merged_at=(
                datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=d)
            ),
        )
        for d in [20, 25, 30, 35, 40]
    ]
    # median TTM = 30 days, threshold = max(30, min(150, 180)) = 150
    threshold = _compute_stale_threshold(prs)
    assert threshold == 150.0


def test_stale_threshold_capped() -> None:
    prs = [
        _make_pr(
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
            merged_at=(
                datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=d)
            ),
        )
        for d in [50, 60, 70, 80, 90]
    ]
    # median TTM = 70 days, threshold = max(30, min(350, 180)) = 180
    threshold = _compute_stale_threshold(prs)
    assert threshold == 180.0


def test_stale_threshold_no_prs() -> None:
    threshold = _compute_stale_threshold([])
    assert threshold == 30.0


def test_classify_merged_pr() -> None:
    pr = _make_pr(merged_at=datetime(2024, 6, 2, tzinfo=UTC))
    result = _classify_pr(pr, stale_threshold_days=60.0)
    assert result is not None
    assert result.outcome == PROutcome.MERGED


def test_classify_rejected_pr() -> None:
    # Closed within stale threshold (10 days < 60 days)
    pr = _make_pr(
        closed_at=datetime(2024, 6, 11, tzinfo=UTC),
    )
    result = _classify_pr(pr, stale_threshold_days=60.0)
    assert result is not None
    assert result.outcome == PROutcome.REJECTED


def test_classify_pocket_veto_closed() -> None:
    # Closed after stale threshold (100 days > 60 days)
    pr = _make_pr(
        closed_at=(
            datetime(2024, 6, 1, tzinfo=UTC) + timedelta(days=100)
        ),
    )
    result = _classify_pr(pr, stale_threshold_days=60.0)
    assert result is not None
    assert result.outcome == PROutcome.POCKET_VETO


def test_classify_pocket_veto_open() -> None:
    # Open past threshold + buffer
    pr = _make_pr()  # No merged_at or closed_at -> OPEN
    now = datetime(2024, 6, 1, tzinfo=UTC) + timedelta(days=100)
    result = _classify_pr(
        pr, stale_threshold_days=60.0, buffer_days=30, now=now,
    )
    assert result is not None
    assert result.outcome == PROutcome.POCKET_VETO


def test_classify_indeterminate() -> None:
    # Open within threshold -> None
    pr = _make_pr()  # OPEN
    now = datetime(2024, 6, 1, tzinfo=UTC) + timedelta(days=20)
    result = _classify_pr(
        pr, stale_threshold_days=60.0, buffer_days=30, now=now,
    )
    assert result is None
