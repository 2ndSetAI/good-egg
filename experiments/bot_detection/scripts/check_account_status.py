"""Check GitHub account status for all PR authors in the bot detection DB.

For each author, queries GET /users/{login}:
  200 -> "active"
  404 -> "suspended" (suspended or deleted)
  403 -> rate limited, retry after wait

Idempotent: skips authors that already have a status recorded.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import httpx

from experiments.bot_detection.cache import BotDetectionDB

logger = logging.getLogger(__name__)

DEFAULT_DB = "experiments/bot_detection/data/bot_detection.duckdb"


def _upsert_profile_fields(
    db: BotDetectionDB,
    login: str,
    user_data: dict[str, Any],
) -> None:
    """Store profile fields from a GitHub API /users response."""
    created_at = None
    raw_created = user_data.get("created_at")
    if raw_created:
        try:
            raw_created = raw_created.rstrip("Z")
            if "+" in raw_created:
                raw_created = raw_created[: raw_created.index("+")]
            created_at = datetime.fromisoformat(raw_created)
        except (ValueError, TypeError):
            pass

    followers = user_data.get("followers", 0) or 0
    public_repos = user_data.get("public_repos", 0) or 0

    db.con.execute(
        """INSERT INTO authors (login, account_created_at, followers, public_repos)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (login) DO UPDATE SET
            account_created_at = COALESCE(?, authors.account_created_at),
            followers = ?,
            public_repos = ?""",
        [login, created_at, followers, public_repos,
         created_at, followers, public_repos],
    )


def _check_accounts(
    db: BotDetectionDB,
    token: str,
    limit: int | None = None,
    min_repos: int = 1,
) -> None:
    """Query GitHub API for each author without a status.

    If limit is set and min_repos > 1, check only the most suspicious
    authors (multi-repo, ordered by merge rate ascending).
    """
    if min_repos > 1:
        # Targeted check: most suspicious authors first
        target_authors = db.get_suspicious_authors(
            limit=limit or 999999, min_repos=min_repos,
        )
    else:
        # All authors ordered by merge rate ascending (most suspicious first)
        target_authors_rows = db.con.execute(
            """SELECT author
            FROM prs
            GROUP BY author
            ORDER BY SUM(CASE WHEN state = 'MERGED' THEN 1 ELSE 0 END)::DOUBLE
                / COUNT(*) ASC
            LIMIT ?""",
            [limit or 999999],
        ).fetchall()
        target_authors = [r[0] for r in target_authors_rows]

    # Find which already have a status
    existing = db.con.execute(
        "SELECT login, account_status FROM authors WHERE account_status IS NOT NULL"
    ).fetchall()
    done = {r[0] for r in existing}
    to_check = [a for a in target_authors if a not in done]
    logger.info(
        "Targeted check: %d target authors, already checked: %d, remaining: %d",
        len(target_authors), len(done), len(to_check),
    )

    if not to_check:
        logger.info("All authors already checked.")
        return

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }

    # Very conservative: 2s between requests, check remaining before every call,
    # stop using more than 50% of the budget per reset window
    request_delay = 2.0
    max_usage_fraction = 0.5  # only use half the rate limit budget

    with httpx.Client(headers=headers, timeout=30.0) as client:
        # Check current rate limit before starting
        rl_resp = client.get("https://api.github.com/rate_limit")
        if rl_resp.status_code == 200:
            rl = rl_resp.json()["resources"]["core"]
            logger.info(
                "Rate limit: %d/%d remaining, resets at %d",
                rl["remaining"], rl["limit"], rl["reset"],
            )
            budget = int(rl["limit"] * max_usage_fraction)
            if rl["remaining"] < budget:
                wait = max(1, rl["reset"] - int(time.time()) + 5)
                logger.info(
                    "Waiting %ds for rate limit to reset (have %d, need %d)...",
                    wait, rl["remaining"], budget,
                )
                time.sleep(wait)

        for i, login in enumerate(to_check):
            if i > 0 and i % 100 == 0:
                logger.info("Progress: %d / %d checked", i, len(to_check))

            time.sleep(request_delay)

            while True:
                resp = client.get(f"https://api.github.com/users/{login}")

                if resp.status_code == 200:
                    status = "active"
                    # Store profile fields from the response
                    user_data = resp.json()
                    _upsert_profile_fields(db, login, user_data)
                    break
                elif resp.status_code == 404:
                    status = "suspended"
                    logger.info("SUSPENDED: %s", login)
                    break
                elif resp.status_code in (403, 429):
                    reset_at = resp.headers.get("X-RateLimit-Reset")
                    wait = (
                        max(60, int(reset_at) - int(time.time()) + 5)
                        if reset_at else 300
                    )
                    logger.warning("Rate limited. Sleeping %ds...", wait)
                    time.sleep(wait)
                    continue
                else:
                    logger.warning(
                        "Unexpected status %d for %s, skipping", resp.status_code, login,
                    )
                    status = f"error_{resp.status_code}"
                    break

            # Upsert into authors table
            db.con.execute(
                """INSERT INTO authors (login, account_status)
                VALUES (?, ?)
                ON CONFLICT (login) DO UPDATE SET account_status = ?""",
                [login, status, status],
            )

            # Check remaining after every request — pause at half budget
            remaining = resp.headers.get("X-RateLimit-Remaining")
            limit = resp.headers.get("X-RateLimit-Limit")
            if remaining is not None and limit is not None:
                threshold = int(int(limit) * max_usage_fraction)
                if int(remaining) < threshold:
                    reset_at = resp.headers.get("X-RateLimit-Reset")
                    if reset_at:
                        wait = max(60, int(reset_at) - int(time.time()) + 5)
                        logger.info(
                            "Used half budget (%s/%s remaining). Sleeping %ds...",
                            remaining, limit, wait,
                        )
                        time.sleep(wait)

    logger.info("Done. Checked %d authors.", len(to_check))


@click.command()
@click.option(
    "--db-path",
    default=DEFAULT_DB,
    type=click.Path(exists=False),
    help="Path to bot_detection DuckDB.",
)
@click.option(
    "--limit",
    default=None,
    type=int,
    help="Max authors to check (suspicious-first when combined with --min-repos).",
)
@click.option(
    "--min-repos",
    default=1,
    type=int,
    help="Min distinct repos for targeted check (default: 1 = all authors).",
)
def main(db_path: str, limit: int | None, min_repos: int) -> None:
    """Check GitHub account status for all PR authors."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        logger.error("GITHUB_TOKEN environment variable is required.")
        sys.exit(1)

    path = Path(db_path)
    if not path.exists():
        logger.error("Database not found: %s", path)
        sys.exit(1)

    with BotDetectionDB(path) as db:
        _check_accounts(db, token, limit=limit, min_repos=min_repos)


if __name__ == "__main__":
    main()
