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
from pathlib import Path

import click
import httpx

from experiments.bot_detection.cache import BotDetectionDB

logger = logging.getLogger(__name__)

DEFAULT_DB = "experiments/bot_detection/data/bot_detection.duckdb"


def _check_accounts(db: BotDetectionDB, token: str) -> None:
    """Query GitHub API for each author without a status."""
    # Get all distinct authors from prs table (superset of authors table)
    all_authors_rows = db.con.execute(
        "SELECT DISTINCT author FROM prs ORDER BY author"
    ).fetchall()
    all_authors = [r[0] for r in all_authors_rows]

    # Find which already have a status
    existing = db.con.execute(
        "SELECT login, account_status FROM authors WHERE account_status IS NOT NULL"
    ).fetchall()
    done = {r[0] for r in existing}

    to_check = [a for a in all_authors if a not in done]
    logger.info(
        "Total authors: %d, already checked: %d, remaining: %d",
        len(all_authors), len(done), len(to_check),
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
    REQUEST_DELAY = 2.0
    MAX_USAGE_FRACTION = 0.5  # only use half the rate limit budget

    with httpx.Client(headers=headers, timeout=30.0) as client:
        # Check current rate limit before starting
        rl_resp = client.get("https://api.github.com/rate_limit")
        if rl_resp.status_code == 200:
            rl = rl_resp.json()["resources"]["core"]
            logger.info(
                "Rate limit: %d/%d remaining, resets at %d",
                rl["remaining"], rl["limit"], rl["reset"],
            )
            budget = int(rl["limit"] * MAX_USAGE_FRACTION)
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

            time.sleep(REQUEST_DELAY)

            while True:
                resp = client.get(f"https://api.github.com/users/{login}")

                if resp.status_code == 200:
                    status = "active"
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
                threshold = int(int(limit) * MAX_USAGE_FRACTION)
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
def main(db_path: str) -> None:
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
        _check_accounts(db, token)


if __name__ == "__main__":
    main()
