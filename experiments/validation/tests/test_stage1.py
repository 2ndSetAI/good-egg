from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from experiments.validation.models import RepoEntry, StudyConfig, TemporalBin
from experiments.validation.stages.stage1_collect_prs import (
    _gh_search_prs,
    _parse_pr,
    run_stage1,
)


def _make_gh_result(
    number: int = 1,
    state: str = "MERGED",
    login: str = "alice",
) -> dict:
    return {
        "number": number,
        "title": f"PR #{number}",
        "author": {"login": login},
        "state": state,
        "createdAt": "2024-06-01T00:00:00Z",
        "mergedAt": (
            "2024-06-02T00:00:00Z" if state == "MERGED" else None
        ),
        "closedAt": (
            "2024-06-02T00:00:00Z" if state != "OPEN" else None
        ),
        "additions": 10,
        "deletions": 5,
        "changedFiles": 2,
    }


def test_parse_pr_merged() -> None:
    raw = _make_gh_result(number=42, state="MERGED")
    pr = _parse_pr(raw, "owner/repo", "2024H1")
    assert pr.number == 42
    assert pr.repo == "owner/repo"
    assert pr.author_login == "alice"
    assert pr.merged_at is not None
    assert pr.temporal_bin == "2024H1"


def test_parse_pr_closed() -> None:
    raw = _make_gh_result(number=10, state="CLOSED")
    pr = _parse_pr(raw, "owner/repo", "2024H2")
    assert pr.merged_at is None
    assert pr.closed_at is not None


def test_parse_pr_string_author() -> None:
    raw = _make_gh_result()
    raw["author"] = "bob"
    pr = _parse_pr(raw, "owner/repo", "2024H1")
    assert pr.author_login == "bob"


@patch(
    "experiments.validation.stages.stage1_collect_prs"
    ".subprocess.run",
)
def test_gh_search_prs_success(mock_run: object) -> None:
    mock_run.return_value.returncode = 0  # type: ignore[attr-defined]
    mock_run.return_value.stdout = json.dumps(  # type: ignore[attr-defined]
        [_make_gh_result()],
    )
    results = _gh_search_prs(
        "owner/repo", "--merged", "2024-01-01..2024-06-30", 25,
    )
    assert len(results) == 1


@patch(
    "experiments.validation.stages.stage1_collect_prs"
    ".subprocess.run",
)
def test_gh_search_prs_failure(mock_run: object) -> None:
    mock_run.return_value.returncode = 1  # type: ignore[attr-defined]
    mock_run.return_value.stderr = "error"  # type: ignore[attr-defined]
    results = _gh_search_prs(
        "owner/repo", "--merged", "2024-01-01..2024-06-30", 25,
    )
    assert results == []


def test_run_stage1_skips_collected(tmp_path: Path) -> None:
    prs_dir = tmp_path / "data" / "raw" / "prs"
    prs_dir.mkdir(parents=True)
    # Create existing output
    (prs_dir / "owner__repo.jsonl").write_text('{"number": 1}\n')

    config = StudyConfig(
        temporal_bins=[
            TemporalBin(
                label="2024H1",
                start="2024-01-01",
                end="2024-06-30",
            ),
        ],
        paths={"raw_prs": "data/raw/prs"},
    )
    repos = [RepoEntry(name="owner/repo")]

    with patch(
        "experiments.validation.stages.stage1_collect_prs"
        "._collect_repo_prs",
    ) as mock_collect:
        run_stage1(tmp_path, config, repos)
        mock_collect.assert_not_called()
