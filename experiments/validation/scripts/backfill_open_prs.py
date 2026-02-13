"""Backfill open PRs for all repos across temporal bins.

Fetches currently-open PRs created within each temporal bin for all repos
in repo_list_full.yaml. Appends results to existing JSONL files.
Idempotent: skips repos that already have open-state PRs.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml

from experiments.validation.checkpoint import append_jsonl, read_jsonl
from experiments.validation.models import (
    CollectedPR,
    RepoEntry,
    StudyConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path("experiments/validation")

_GH_JSON_FIELDS = (
    "number,title,body,author,state,createdAt,closedAt,"
    "mergedAt,additions,deletions,changedFiles,labels"
)


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

    labels = raw.get("labels", [])
    label_names = [
        lb.get("name", "") if isinstance(lb, dict) else str(lb)
        for lb in labels
    ]

    return CollectedPR(
        repo=repo,
        number=raw["number"],
        author_login=author_login,
        title=raw.get("title", ""),
        body=raw.get("body", ""),
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
        labels=label_names,
        temporal_bin=temporal_bin,
    )


def _has_open_prs(records: list[dict]) -> bool:
    """Check if any existing records have OPEN state."""
    return any(r.get("state") == "OPEN" for r in records)


def _fetch_open_prs(
    repo: str, date_range: str, limit: int,
) -> list[dict]:
    """Fetch open PRs for a repo in a date range via gh CLI."""
    cmd = [
        "gh", "pr", "list",
        "--repo", repo,
        "--state", "open",
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
        return (
            json.loads(result.stdout)
            if result.stdout.strip() else []
        )
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        logger.warning("gh pr list error for %s: %s", repo, exc)
        return []


def main() -> None:
    """Backfill open PRs for all repos."""
    repo_list_path = BASE_DIR / "repo_list_full.yaml"
    config_path = BASE_DIR / "study_config.yaml"

    with open(repo_list_path) as f:
        repo_data = yaml.safe_load(f)
    repos = [RepoEntry(**r) for r in repo_data["repos"]]

    with open(config_path) as f:
        config = StudyConfig(**yaml.safe_load(f))

    data_dir = BASE_DIR / config.paths.get(
        "raw_prs", "data/raw/prs",
    )
    data_dir.mkdir(parents=True, exist_ok=True)

    for repo_entry in repos:
        repo = repo_entry.name
        owner, name = repo.split("/", 1)
        output_path = data_dir / f"{owner}__{name}.jsonl"

        # Load existing records and check for open PRs
        existing = read_jsonl(output_path)
        if _has_open_prs(existing):
            logger.info(
                "Skipping %s (already has open PRs)", repo,
            )
            continue

        existing_numbers = {r["number"] for r in existing}
        new_prs: list[dict] = []

        for tbin in config.temporal_bins:
            date_range = f"{tbin.start}..{tbin.end}"
            raw_list = _fetch_open_prs(repo, date_range, 25)
            time.sleep(2.5)

            for raw in raw_list:
                pr = _parse_pr(raw, repo, tbin.label)
                if pr.number not in existing_numbers:
                    existing_numbers.add(pr.number)
                    new_prs.append(pr.model_dump(mode="json"))

        if new_prs:
            append_jsonl(output_path, new_prs)
            logger.info(
                "Appended %d open PRs for %s",
                len(new_prs), repo,
            )
        else:
            logger.info("No new open PRs for %s", repo)


if __name__ == "__main__":
    main()
