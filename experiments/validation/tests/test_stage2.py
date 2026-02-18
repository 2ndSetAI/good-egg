from __future__ import annotations

from datetime import UTC, datetime, timedelta

from experiments.validation.models import CollectedPR, PROutcome
from experiments.validation.stages.stage2_discover_authors import (
    _classify_pr,
    _compute_stale_threshold,
    _is_bot,
    _is_bursty,
    _is_merge_bot_close,
)


def _make_pr(
    merged_at: datetime | None = None,
    closed_at: datetime | None = None,
    created_at: datetime | None = None,
    labels: list[str] | None = None,
    repo: str = "owner/repo",
    author_login: str = "alice",
    number: int = 1,
) -> CollectedPR:
    if created_at is None:
        created_at = datetime(2024, 6, 1, tzinfo=UTC)
    return CollectedPR(
        repo=repo,
        number=number,
        author_login=author_login,
        title="Test PR",
        state=(
            "MERGED" if merged_at
            else "CLOSED" if closed_at
            else "OPEN"
        ),
        created_at=created_at,
        merged_at=merged_at,
        closed_at=closed_at,
        labels=labels or [],
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
    # p90 TTM = 8.8 days, threshold = max(30, min(8.8, 180)) = 30
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
    # p90 TTM = 38 days, threshold = max(30, min(38, 180)) = 38
    threshold = _compute_stale_threshold(prs)
    assert threshold == 38.0


def test_stale_threshold_capped() -> None:
    prs = [
        _make_pr(
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
            merged_at=(
                datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=d)
            ),
        )
        for d in [100, 150, 200, 250, 300]
    ]
    # p90 TTM = 280 days, threshold = max(30, min(280, 180)) = 180
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
    # Open past threshold (buffer=0 default)
    pr = _make_pr()  # No merged_at or closed_at -> OPEN
    now = datetime(2024, 6, 1, tzinfo=UTC) + timedelta(days=61)
    result = _classify_pr(
        pr, stale_threshold_days=60.0, buffer_days=0, now=now,
    )
    assert result is not None
    assert result.outcome == PROutcome.POCKET_VETO


def test_classify_pocket_veto_open_with_buffer() -> None:
    # Open past threshold + buffer (explicit buffer_days)
    pr = _make_pr()  # No merged_at or closed_at -> OPEN
    now = datetime(2024, 6, 1, tzinfo=UTC) + timedelta(days=100)
    result = _classify_pr(
        pr, stale_threshold_days=60.0, buffer_days=30, now=now,
    )
    assert result is not None
    assert result.outcome == PROutcome.POCKET_VETO


def test_classify_indeterminate() -> None:
    # Open within threshold (buffer=0) -> None
    pr = _make_pr()  # OPEN
    now = datetime(2024, 6, 1, tzinfo=UTC) + timedelta(days=20)
    result = _classify_pr(
        pr, stale_threshold_days=60.0, buffer_days=0, now=now,
    )
    assert result is None


# === Burstiness detection (Rec 2) ===


def test_is_bursty_below_minimum_prs() -> None:
    """Fewer than min_prs PRs should never be bursty."""
    prs = [
        _make_pr(
            created_at=datetime(2024, 6, 1, 0, i, tzinfo=UTC),
            repo=f"org/repo-{i}",
            number=i,
        )
        for i in range(3)
    ]
    assert not _is_bursty(prs, min_prs=5)


def test_is_bursty_single_repo_not_bursty() -> None:
    """Many PRs in one repo within a window should not be bursty."""
    prs = [
        _make_pr(
            created_at=datetime(2024, 6, 1, 0, i, tzinfo=UTC),
            repo="owner/repo",
            number=i,
        )
        for i in range(10)
    ]
    assert not _is_bursty(prs, min_repos=3, min_prs=5)


def test_is_bursty_multi_repo_within_window() -> None:
    """PRs across many repos within the window should be bursty."""
    base = datetime(2024, 6, 1, 0, 0, tzinfo=UTC)
    prs = [
        _make_pr(
            created_at=base + timedelta(minutes=i * 5),
            repo=f"org/repo-{i}",
            number=i,
        )
        for i in range(6)
    ]
    assert _is_bursty(prs, window_hours=1.0, min_repos=3, min_prs=5)


def test_is_bursty_spread_across_time() -> None:
    """PRs spread far apart in time should not be bursty."""
    prs = [
        _make_pr(
            created_at=(
                datetime(2024, 6, 1, tzinfo=UTC) + timedelta(days=i)
            ),
            repo=f"org/repo-{i}",
            number=i,
        )
        for i in range(6)
    ]
    assert not _is_bursty(prs, window_hours=1.0, min_repos=3, min_prs=5)


def test_is_bursty_exact_threshold() -> None:
    """Exactly min_prs PRs across exactly min_repos repos is bursty."""
    base = datetime(2024, 6, 1, 0, 0, tzinfo=UTC)
    prs = [
        _make_pr(
            created_at=base + timedelta(minutes=i),
            repo=f"org/repo-{i % 3}",
            number=i,
        )
        for i in range(5)
    ]
    assert _is_bursty(prs, window_hours=1.0, min_repos=3, min_prs=5)


# === Merge bot detection (Rec 4) ===


def test_merge_bot_close_merged_label() -> None:
    pr = _make_pr(
        closed_at=datetime(2024, 6, 2, tzinfo=UTC),
        labels=["merged"],
    )
    assert _is_merge_bot_close(pr)


def test_merge_bot_close_auto_merge_label() -> None:
    pr = _make_pr(
        closed_at=datetime(2024, 6, 2, tzinfo=UTC),
        labels=["auto-merge"],
    )
    assert _is_merge_bot_close(pr)


def test_merge_bot_close_automerge_label() -> None:
    pr = _make_pr(
        closed_at=datetime(2024, 6, 2, tzinfo=UTC),
        labels=["S-waiting-on-automerge"],
    )
    assert _is_merge_bot_close(pr)


def test_merge_bot_close_bors_label() -> None:
    pr = _make_pr(
        closed_at=datetime(2024, 6, 2, tzinfo=UTC),
        labels=["bors-merge"],
    )
    assert _is_merge_bot_close(pr)


def test_merge_bot_close_mergify_label() -> None:
    pr = _make_pr(
        closed_at=datetime(2024, 6, 2, tzinfo=UTC),
        labels=["mergify/merge"],
    )
    assert _is_merge_bot_close(pr)


def test_merge_bot_close_no_matching_labels() -> None:
    pr = _make_pr(
        closed_at=datetime(2024, 6, 2, tzinfo=UTC),
        labels=["bug", "enhancement"],
    )
    assert not _is_merge_bot_close(pr)


def test_merge_bot_close_empty_labels() -> None:
    pr = _make_pr(
        closed_at=datetime(2024, 6, 2, tzinfo=UTC),
        labels=[],
    )
    assert not _is_merge_bot_close(pr)


def test_merge_bot_close_case_insensitive() -> None:
    pr = _make_pr(
        closed_at=datetime(2024, 6, 2, tzinfo=UTC),
        labels=["MERGED"],
    )
    assert _is_merge_bot_close(pr)


def test_merge_bot_reclassification_in_classify() -> None:
    """A closed PR with merge-bot labels, classified as REJECTED,
    should be detectable for reclassification."""
    pr = _make_pr(
        closed_at=datetime(2024, 6, 11, tzinfo=UTC),
        labels=["auto-merge"],
    )
    result = _classify_pr(pr, stale_threshold_days=60.0)
    assert result is not None
    # Without reclassification, this is REJECTED
    assert result.outcome == PROutcome.REJECTED
    # But merge bot detection flags it
    assert _is_merge_bot_close(pr)


def test_merge_bot_pocket_veto_reclassifiable() -> None:
    """A closed PR past threshold with merge-bot labels should
    also be detectable."""
    pr = _make_pr(
        closed_at=(
            datetime(2024, 6, 1, tzinfo=UTC) + timedelta(days=100)
        ),
        labels=["bors"],
    )
    result = _classify_pr(pr, stale_threshold_days=60.0)
    assert result is not None
    assert result.outcome == PROutcome.POCKET_VETO
    assert _is_merge_bot_close(pr)
