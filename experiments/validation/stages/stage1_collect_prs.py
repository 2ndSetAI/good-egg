from __future__ import annotations

import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path

from experiments.validation.checkpoint import (
    append_jsonl,
    repo_already_collected,
)
from experiments.validation.models import (
    CollectedPR,
    RepoEntry,
    StudyConfig,
    TemporalBin,
)

logger = logging.getLogger(__name__)

_GH_JSON_FIELDS = (
    "number,title,author,state,createdAt,closedAt,"
    "mergedAt,additions,deletions,changedFiles"
)


def _gh_search_prs(
    repo: str,
    state_flag: str,
    date_range: str,
    limit: int,
) -> list[dict]:
    """Run gh pr list and return parsed JSON results.

    state_flag should be one of: 'merged', 'closed', 'open'
    date_range is a GitHub search date qualifier (e.g. '2024-01-01..2024-07-01'
    or '<2024-01-01').
    """
    cmd = [
        "gh", "pr", "list",
        "--repo", repo,
        "--state", state_flag,
        "--search", f"created:{date_range}",
        "--limit", str(limit),
        "--json", _GH_JSON_FIELDS,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            logger.warning(
                "gh pr list failed for %s: %s",
                repo, result.stderr.strip(),
            )
            return []
        return json.loads(result.stdout) if result.stdout.strip() else []
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        logger.warning("gh pr list error for %s: %s", repo, exc)
        return []


def _parse_pr(
    raw: dict, repo: str, temporal_bin: str,
) -> CollectedPR:
    """Parse a raw gh search result into a CollectedPR."""
    author = raw.get("author", {})
    author_login = ""
    if isinstance(author, dict):
        author_login = author.get("login", "")
    elif isinstance(author, str):
        author_login = author

    merged_at = raw.get("mergedAt")
    closed_at = raw.get("closedAt")

    return CollectedPR(
        repo=repo,
        number=raw["number"],
        author_login=author_login,
        title=raw.get("title", ""),
        state=raw.get("state", ""),
        created_at=datetime.fromisoformat(raw["createdAt"]),
        merged_at=(
            datetime.fromisoformat(merged_at) if merged_at else None
        ),
        closed_at=(
            datetime.fromisoformat(closed_at) if closed_at else None
        ),
        additions=raw.get("additions", 0),
        deletions=raw.get("deletions", 0),
        changed_files=raw.get("changedFiles", 0),
        temporal_bin=temporal_bin,
    )


def _collect_repo_prs(
    repo_entry: RepoEntry,
    bins: list[TemporalBin],
    data_dir: Path,
    delay: float,
    per_bin_merged: int,
    per_bin_closed: int,
) -> None:
    """Collect PRs for a single repo across all temporal bins."""
    repo = repo_entry.name
    owner, name = repo.split("/", 1)
    output_path = data_dir / f"{owner}__{name}.jsonl"

    all_prs: list[dict] = []
    seen_numbers: set[int] = set()

    for tbin in bins:
        date_range = f"{tbin.start}..{tbin.end}"

        # Fetch merged PRs
        merged_raw = _gh_search_prs(
            repo, "merged", date_range, per_bin_merged,
        )
        time.sleep(delay)

        for raw in merged_raw:
            pr = _parse_pr(raw, repo, tbin.label)
            if pr.number not in seen_numbers:
                seen_numbers.add(pr.number)
                all_prs.append(pr.model_dump(mode="json"))

        # Fetch closed (non-merged) PRs
        closed_raw = _gh_search_prs(
            repo, "closed", date_range, per_bin_closed,
        )
        time.sleep(delay)

        for raw in closed_raw:
            pr = _parse_pr(raw, repo, tbin.label)
            if pr.number not in seen_numbers:
                seen_numbers.add(pr.number)
                all_prs.append(pr.model_dump(mode="json"))

    # Fetch long-open PRs (created > 210 days ago = max stale + buffer)
    open_raw = _gh_search_prs(
        repo,
        "open",
        f"<{bins[0].start}",
        per_bin_closed,
    )
    time.sleep(delay)

    for raw in open_raw:
        pr = _parse_pr(raw, repo, bins[0].label)
        if pr.number not in seen_numbers:
            seen_numbers.add(pr.number)
            all_prs.append(pr.model_dump(mode="json"))

    if all_prs:
        append_jsonl(output_path, all_prs)
        logger.info(
            "Collected %d PRs for %s", len(all_prs), repo,
        )
    else:
        logger.warning("No PRs collected for %s", repo)


def run_stage1(
    base_dir: Path,
    config: StudyConfig,
    repos: list[RepoEntry],
    limit: int = 0,
) -> None:
    """Stage 1: Collect PRs from GitHub via gh CLI."""
    data_dir = base_dir / config.paths.get("raw_prs", "data/raw/prs")
    data_dir.mkdir(parents=True, exist_ok=True)

    delay = config.collection.get("gh_search_delay_seconds", 2.5)
    per_bin_merged = config.collection.get("merged_per_bin", 25)
    per_bin_closed = config.collection.get("closed_per_bin", 25)

    if limit > 0:
        per_bin_merged = min(per_bin_merged, limit)
        per_bin_closed = min(per_bin_closed, limit)

    for repo_entry in repos:
        owner, name = repo_entry.name.split("/", 1)
        if repo_already_collected(data_dir, owner, name):
            logger.info("Skipping %s (already collected)", repo_entry.name)
            continue

        logger.info("Collecting PRs for %s", repo_entry.name)
        _collect_repo_prs(
            repo_entry,
            config.temporal_bins,
            data_dir,
            delay,
            per_bin_merged,
            per_bin_closed,
        )
