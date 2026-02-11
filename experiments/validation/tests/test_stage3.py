from __future__ import annotations

from pathlib import Path

from experiments.validation.checkpoint import author_already_fetched
from experiments.validation.stages.stage3_fetch_authors import (
    _classify_tier2_prs,
)


def test_author_already_fetched(tmp_path: Path) -> None:
    (tmp_path / "alice.json").write_text('{"login": "alice"}')
    assert author_already_fetched(tmp_path, "alice")
    assert not author_already_fetched(tmp_path, "bob")


def test_classify_tier2_prs_mixed() -> None:
    prs = [
        {
            "createdAt": "2024-01-01T00:00:00Z",
            "closedAt": "2024-01-10T00:00:00Z",  # 9 days < 90
        },
        {
            "createdAt": "2024-01-01T00:00:00Z",
            "closedAt": "2024-06-01T00:00:00Z",  # 152 days > 90
        },
        {
            "createdAt": "2024-01-01T00:00:00Z",
            "closedAt": "2024-01-15T00:00:00Z",  # 14 days < 90
        },
    ]
    result = _classify_tier2_prs(prs, stale_threshold_days=90)
    assert result["timeout_count"] == 1
    assert result["explicit_rejection_count"] == 2


def test_classify_tier2_empty() -> None:
    result = _classify_tier2_prs([])
    assert result["timeout_count"] == 0
    assert result["explicit_rejection_count"] == 0


def test_classify_tier2_missing_dates() -> None:
    prs = [
        {"createdAt": "2024-01-01T00:00:00Z", "closedAt": None},
    ]
    result = _classify_tier2_prs(prs, stale_threshold_days=90)
    assert result["timeout_count"] == 0
    assert result["explicit_rejection_count"] == 0
