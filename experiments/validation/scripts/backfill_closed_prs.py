"""Backfill closed PR timestamps for all authors.

Fetches individual closed PR records (createdAt, closedAt) from GitHub
GraphQL API and stores them in each author's JSON file. This enables
exact temporal scoping of the H5 merge rate feature.

Usage:
    uv run python -m experiments.validation.scripts.backfill_closed_prs
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_CLOSED_PRS_QUERY = """
query($login: String!, $limit: Int!, $cursor: String) {
  user(login: $login) {
    pullRequests(
      first: $limit,
      states: CLOSED,
      orderBy: {field: CREATED_AT, direction: DESC},
      after: $cursor
    ) {
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        createdAt
        closedAt
        repository { nameWithOwner }
      }
    }
  }
}
""".strip()


async def _fetch_closed_prs(
    client: httpx.AsyncClient,
    token: str,
    login: str,
    max_prs: int = 500,
) -> list[dict]:
    """Fetch closed PR records with pagination."""
    all_prs: list[dict] = []
    cursor: str | None = None
    page_size = 100

    while len(all_prs) < max_prs:
        variables: dict = {
            "login": login,
            "limit": min(page_size, max_prs - len(all_prs)),
        }
        if cursor:
            variables["cursor"] = cursor

        response = await client.post(
            "https://api.github.com/graphql",
            json={"query": _CLOSED_PRS_QUERY, "variables": variables},
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )

        if response.status_code == 403:
            # Rate limited; wait and retry
            retry_after = int(
                response.headers.get("Retry-After", "60"),
            )
            logger.warning(
                "Rate limited for %s, waiting %ds", login, retry_after,
            )
            await asyncio.sleep(retry_after)
            continue

        if response.status_code != 200:
            logger.warning(
                "Query failed for %s: %d", login, response.status_code,
            )
            break

        data = response.json()
        if "errors" in data:
            logger.warning("GraphQL errors for %s: %s", login, data["errors"])
            break

        user = data.get("data", {}).get("user")
        if not user:
            break

        prs_data = user["pullRequests"]
        nodes = prs_data["nodes"]
        for node in nodes:
            all_prs.append({
                "created_at": node.get("createdAt"),
                "closed_at": node.get("closedAt"),
                "repo": (node.get("repository") or {}).get(
                    "nameWithOwner", "",
                ),
            })

        page_info = prs_data["pageInfo"]
        if not page_info["hasNextPage"] or not nodes:
            break
        cursor = page_info["endCursor"]

    return all_prs


async def backfill(
    authors_dir: Path,
    concurrency: int = 10,
    max_prs_per_author: int = 500,
) -> None:
    """Backfill closed PR timestamps for all authors."""
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        logger.error("GITHUB_TOKEN not set")
        sys.exit(1)

    # Find authors needing backfill
    author_files = sorted(authors_dir.glob("*.json"))
    to_fetch: list[tuple[Path, dict]] = []

    for fpath in author_files:
        if fpath.name == "unique_authors.json":
            continue
        with open(fpath) as f:
            data = json.load(f)
        closed_count = data.get("closed_count", 0)
        # Skip if already has closed_prs list or no closed PRs
        if "closed_prs" in data or not closed_count or closed_count <= 0:
            continue
        to_fetch.append((fpath, data))

    logger.info(
        "Backfill: %d authors need closed PR timestamps "
        "(%d already done or no closed PRs)",
        len(to_fetch),
        len(author_files) - 1 - len(to_fetch),
    )

    if not to_fetch:
        logger.info("Nothing to backfill")
        return

    semaphore = asyncio.Semaphore(concurrency)
    done = 0
    failed = 0

    async def fetch_one(
        fpath: Path, data: dict, index: int,
    ) -> None:
        nonlocal done, failed
        login = data["login"]
        async with semaphore:
            try:
                closed_prs = await _fetch_closed_prs(
                    client, token, login, max_prs_per_author,
                )
                data["closed_prs"] = closed_prs
                with open(fpath, "w") as f:
                    json.dump(data, f, indent=2)
                done += 1
                if done % 100 == 0:
                    logger.info(
                        "Progress: %d/%d done (%d failed)",
                        done, len(to_fetch), failed,
                    )
            except Exception:
                logger.exception("Failed for %s", login)
                failed += 1

    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [
            fetch_one(fpath, data, i)
            for i, (fpath, data) in enumerate(to_fetch)
        ]
        await asyncio.gather(*tasks)

    logger.info(
        "Backfill complete: %d succeeded, %d failed",
        done, failed,
    )


def main() -> None:
    authors_dir = Path(
        "experiments/validation/data/raw/authors",
    )
    if not authors_dir.exists():
        logger.error("Authors directory not found: %s", authors_dir)
        sys.exit(1)

    asyncio.run(backfill(authors_dir))


if __name__ == "__main__":
    main()
