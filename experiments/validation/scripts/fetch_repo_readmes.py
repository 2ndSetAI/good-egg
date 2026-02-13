"""Fetch README content for all target repositories.

For each repo in repo_list_full.yaml, fetches the README via GitHub API
and saves it to data/raw/repos/{owner}__{repo}_readme.md.
Idempotent: skips repos that already have a README file.
"""
from __future__ import annotations

import base64
import logging
import subprocess
import time
from pathlib import Path

import yaml

from experiments.validation.models import RepoEntry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path("experiments/validation")


def _fetch_readme(repo: str) -> str | None:
    """Fetch README content for a repo via gh API.

    Returns decoded README text, or None on failure.
    """
    cmd = [
        "gh", "api",
        f"repos/{repo}/readme",
        "--jq", ".content",
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
                "gh api failed for %s: %s",
                repo, result.stderr.strip(),
            )
            return None

        content_b64 = result.stdout.strip()
        if not content_b64:
            logger.warning("Empty README for %s", repo)
            return None

        # GitHub returns base64-encoded content (may have newlines)
        cleaned = content_b64.replace("\n", "")
        return base64.b64decode(cleaned).decode("utf-8")

    except (subprocess.TimeoutExpired, Exception) as exc:
        logger.warning(
            "README fetch error for %s: %s", repo, exc,
        )
        return None


def main() -> None:
    """Fetch READMEs for all repos."""
    repo_list_path = BASE_DIR / "repo_list_full.yaml"
    with open(repo_list_path) as f:
        repo_data = yaml.safe_load(f)
    repos = [RepoEntry(**r) for r in repo_data["repos"]]

    output_dir = BASE_DIR / "data" / "raw" / "repos"
    output_dir.mkdir(parents=True, exist_ok=True)

    for repo_entry in repos:
        repo = repo_entry.name
        owner, name = repo.split("/", 1)
        output_path = output_dir / f"{owner}__{name}_readme.md"

        if output_path.exists():
            logger.info(
                "Skipping %s (README already exists)", repo,
            )
            continue

        readme = _fetch_readme(repo)
        if readme is not None:
            output_path.write_text(readme)
            logger.info("Saved README for %s", repo)
        else:
            logger.warning("No README found for %s", repo)

        time.sleep(1)


if __name__ == "__main__":
    main()
