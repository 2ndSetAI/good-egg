"""Backfill PR body text for all existing PR records.

Re-fetches PR data with body field included, then updates existing JSONL
records in-place by matching on PR number.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path

import yaml

from experiments.validation.checkpoint import read_jsonl
from experiments.validation.models import StudyConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path("experiments/validation")


def _fetch_bodies(
    repo: str, state: str, limit: int,
) -> dict[int, str]:
    """Fetch PR bodies for a repo via gh CLI.

    Returns a dict mapping PR number to body text.
    """
    cmd = [
        "gh", "pr", "list",
        "--repo", repo,
        "--state", state,
        "--limit", str(limit),
        "--json", "number,body",
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
                "gh pr list failed for %s (%s): %s",
                repo, state, result.stderr.strip(),
            )
            return {}
        items = (
            json.loads(result.stdout)
            if result.stdout.strip() else []
        )
        return {
            item["number"]: item.get("body", "")
            for item in items
        }
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        logger.warning(
            "gh pr list error for %s (%s): %s", repo, state, exc,
        )
        return {}


def main() -> None:
    """Backfill PR bodies for all repos."""
    config_path = BASE_DIR / "study_config.yaml"
    with open(config_path) as f:
        config = StudyConfig(**yaml.safe_load(f))

    data_dir = BASE_DIR / config.paths.get(
        "raw_prs", "data/raw/prs",
    )
    if not data_dir.exists():
        logger.error("Data directory %s does not exist", data_dir)
        return

    for jsonl_path in sorted(data_dir.glob("*.jsonl")):
        # Extract repo name from filename: owner__repo.jsonl
        stem = jsonl_path.stem
        if "__" not in stem:
            continue
        owner, name = stem.split("__", 1)
        repo = f"{owner}/{name}"

        # Fetch bodies for all states
        bodies: dict[int, str] = {}
        for state in ("all", "merged", "closed"):
            fetched = _fetch_bodies(repo, state, 200)
            bodies.update(fetched)
            time.sleep(2.5)

        if not bodies:
            logger.info("No bodies fetched for %s", repo)
            continue

        # Read existing records, update bodies, write back
        records = read_jsonl(jsonl_path)
        updated = 0
        for record in records:
            pr_num = record.get("number")
            if pr_num in bodies:
                old_body = record.get("body", "")
                new_body = bodies[pr_num] or ""
                if old_body != new_body:
                    record["body"] = new_body
                    updated += 1

        # Write back all records
        with open(jsonl_path, "w") as f:
            for record in records:
                f.write(json.dumps(record, default=str) + "\n")

        logger.info(
            "Updated %d/%d PR bodies for %s",
            updated, len(records), repo,
        )


if __name__ == "__main__":
    main()
