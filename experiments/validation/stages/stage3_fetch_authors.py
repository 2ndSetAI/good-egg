from __future__ import annotations

import asyncio
import logging
import os
import random
from datetime import datetime, timedelta
from pathlib import Path

import httpx

from experiments.validation.checkpoint import (
    author_already_fetched,
    read_json,
    write_json,
)
from experiments.validation.models import StudyConfig
from good_egg.config import GoodEggConfig
from good_egg.github_client import GitHubClient

logger = logging.getLogger(__name__)

_TIER1_QUERY = """
query($login: String!) {
  user(login: $login) {
    mergedPrs: pullRequests(states: MERGED) { totalCount }
    closedPrs: pullRequests(states: CLOSED) { totalCount }
    openPrs: pullRequests(states: OPEN) { totalCount }
  }
}
""".strip()

_TIER2_QUERY = """
query($login: String!, $limit: Int!) {
  user(login: $login) {
    pullRequests(
      first: $limit,
      states: CLOSED,
      orderBy: {field: CREATED_AT, direction: DESC}
    ) {
      nodes {
        createdAt
        closedAt
        repository { nameWithOwner }
      }
    }
  }
}
""".strip()


async def _fetch_tier1_counts(
    client: httpx.AsyncClient,
    token: str,
    login: str,
) -> dict[str, int]:
    """Fetch Tier 1 PR counts by state via GraphQL."""
    response = await client.post(
        "https://api.github.com/graphql",
        json={"query": _TIER1_QUERY, "variables": {"login": login}},
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    if response.status_code != 200:
        logger.warning(
            "Tier 1 query failed for %s: %d", login, response.status_code,
        )
        return {}

    data = response.json()
    user = data.get("data", {}).get("user")
    if not user:
        return {}

    return {
        "merged_count": user["mergedPrs"]["totalCount"],
        "closed_count": user["closedPrs"]["totalCount"],
        "open_count": user["openPrs"]["totalCount"],
    }


async def _fetch_tier2_closed_prs(
    client: httpx.AsyncClient,
    token: str,
    login: str,
    limit: int = 100,
) -> list[dict]:
    """Fetch Tier 2 closed PR details via GraphQL."""
    response = await client.post(
        "https://api.github.com/graphql",
        json={
            "query": _TIER2_QUERY,
            "variables": {"login": login, "limit": limit},
        },
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    if response.status_code != 200:
        logger.warning(
            "Tier 2 query failed for %s: %d", login, response.status_code,
        )
        return []

    data = response.json()
    user = data.get("data", {}).get("user")
    if not user:
        return []

    return user["pullRequests"]["nodes"]


def _classify_tier2_prs(
    closed_prs: list[dict],
    stale_threshold_days: int = 90,
) -> dict[str, int]:
    """Classify closed PRs as timeout vs explicit rejection."""
    timeout_count = 0
    explicit_count = 0
    threshold = timedelta(days=stale_threshold_days)

    for pr in closed_prs:
        created = pr.get("createdAt")
        closed = pr.get("closedAt")
        if not created or not closed:
            continue
        created_dt = datetime.fromisoformat(created)
        closed_dt = datetime.fromisoformat(closed)
        duration = closed_dt - created_dt
        if duration > threshold:
            timeout_count += 1
        else:
            explicit_count += 1

    return {
        "timeout_count": timeout_count,
        "explicit_rejection_count": explicit_count,
    }


async def _fetch_one_author(
    login: str,
    index: int,
    total: int,
    ge_client: GitHubClient,
    http_client: httpx.AsyncClient,
    token: str,
    authors_dir: Path,
    tier2_set: set[str],
    tier2_stale_days: int,
    tier2_max_prs: int,
    semaphore: asyncio.Semaphore,
) -> bool:
    """Fetch data for a single author, guarded by semaphore."""
    async with semaphore:
        logger.info("Fetching author %d/%d: %s", index, total, login)
        try:
            user_data = await ge_client.get_user_contribution_data(
                login,
            )

            record: dict = {
                "login": login,
                "contribution_data": user_data.model_dump(
                    mode="json",
                ),
            }

            tier1 = await _fetch_tier1_counts(
                http_client, token, login,
            )
            record.update(tier1)

            if login in tier2_set:
                closed_prs = await _fetch_tier2_closed_prs(
                    http_client, token, login, tier2_max_prs,
                )
                tier2_results = _classify_tier2_prs(
                    closed_prs, tier2_stale_days,
                )
                record["tier2_sampled"] = True
                record["tier2_timeout_count"] = (
                    tier2_results["timeout_count"]
                )
                record["tier2_explicit_rejection_count"] = (
                    tier2_results["explicit_rejection_count"]
                )
            else:
                record["tier2_sampled"] = False

            write_json(authors_dir / f"{login}.json", record)
            return True

        except Exception:
            logger.exception("Failed to fetch data for %s", login)
            return False


async def run_stage3(base_dir: Path, config: StudyConfig) -> None:
    """Stage 3: Fetch author contribution data using GE library."""
    authors_dir = base_dir / config.paths.get(
        "raw_authors", "data/raw/authors",
    )
    authors_dir.mkdir(parents=True, exist_ok=True)

    # Load unique authors from Stage 2
    authors_file = authors_dir / "unique_authors.json"
    if not authors_file.exists():
        logger.error("unique_authors.json not found. Run Stage 2 first.")
        return

    authors_data = read_json(authors_file)
    author_logins: list[str] = authors_data["authors"]
    logger.info("Stage 3: Processing %d authors", len(author_logins))

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        logger.error("GITHUB_TOKEN not set")
        return

    tier2_fraction = config.fetch.get("tier2_sample_fraction", 0.20)
    tier2_stale_days = config.fetch.get("tier2_stale_threshold_days", 90)
    tier2_max_prs = config.fetch.get("tier2_max_closed_prs", 100)
    concurrency = config.fetch.get("concurrency", 10)

    # Determine Tier 2 sample
    rng = random.Random(42)
    tier2_set = set(rng.sample(
        author_logins,
        k=min(
            int(len(author_logins) * tier2_fraction),
            len(author_logins),
        ),
    ))

    ge_config = GoodEggConfig()
    semaphore = asyncio.Semaphore(concurrency)

    # Filter to only authors not yet fetched
    pending = [
        (i, login) for i, login in enumerate(author_logins, 1)
        if not author_already_fetched(authors_dir, login)
    ]
    logger.info(
        "Stage 3: %d already fetched, %d remaining",
        len(author_logins) - len(pending), len(pending),
    )

    async with (
        GitHubClient(token=token, config=ge_config) as ge_client,
        httpx.AsyncClient(timeout=30.0) as http_client,
    ):
        tasks = [
            _fetch_one_author(
                login, index, len(author_logins),
                ge_client, http_client, token,
                authors_dir, tier2_set,
                tier2_stale_days, tier2_max_prs,
                semaphore,
            )
            for index, login in pending
        ]
        results = await asyncio.gather(*tasks)
        succeeded = sum(1 for r in results if r)
        failed = sum(1 for r in results if not r)
        logger.info(
            "Stage 3 complete: %d succeeded, %d failed",
            succeeded, failed,
        )
    logger.info("Stage 3 complete")
