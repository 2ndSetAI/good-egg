"""Fetch updatedAt for every OPEN PR in the bot_detection DuckDB.

Used by pocket_veto_analysis.py to compute idle-time-based staleness (a
better proxy than age-since-created, which the DuckDB schema forces).

For each repo that has OPEN PRs in the DB, paginate
repository.pullRequests(states: OPEN) and collect (number, updatedAt). PRs
that were OPEN in the DB snapshot but have since been closed or merged are
by definition non-stale (the close/merge event itself is activity), so we
don't need to look them up — they just won't appear in the fetched set and
the analysis treats them as non-stale.

Output: experiments/bot_detection/data/open_pr_activity.parquet
  columns: repo, number, updated_at, fetch_now
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import httpx
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DB_PATH = BASE / "data" / "bot_detection.duckdb"
OUT_PATH = BASE / "data" / "open_pr_activity.parquet"

GRAPHQL_URL = "https://api.github.com/graphql"
PAGE_SIZE = 100

QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(states: OPEN, first: 100, after: $cursor,
                 orderBy: {field: CREATED_AT, direction: ASC}) {
      pageInfo { hasNextPage endCursor }
      nodes { number updatedAt }
    }
  }
  rateLimit { remaining resetAt }
}
"""


def get_token() -> str:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token
    result = subprocess.run(
        ["gh", "auth", "token"], check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def fetch_repo(
    client: httpx.Client, owner: str, name: str,
) -> list[tuple[int, str]]:
    results: list[tuple[int, str]] = []
    cursor: str | None = None
    while True:
        resp = client.post(
            GRAPHQL_URL,
            json={
                "query": QUERY,
                "variables": {"owner": owner, "name": name, "cursor": cursor},
            },
        )
        resp.raise_for_status()
        payload = resp.json()
        if "errors" in payload:
            print(f"    GraphQL errors: {payload['errors']}", file=sys.stderr)
            break
        repo_data = payload["data"]["repository"]
        if repo_data is None:
            print(f"    repo not found: {owner}/{name}", file=sys.stderr)
            break
        prs = repo_data["pullRequests"]
        for node in prs["nodes"]:
            results.append((node["number"], node["updatedAt"]))
        remaining = payload["data"]["rateLimit"]["remaining"]
        if remaining < 100:
            reset_at = payload["data"]["rateLimit"]["resetAt"]
            print(
                f"    rate limit low ({remaining}), sleeping until {reset_at}",
                file=sys.stderr,
            )
            time.sleep(60)
        if not prs["pageInfo"]["hasNextPage"]:
            break
        cursor = prs["pageInfo"]["endCursor"]
    return results


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    repo_counts = con.execute("""
        SELECT repo, COUNT(*) AS n
        FROM prs WHERE state='OPEN'
        GROUP BY repo ORDER BY n DESC
    """).fetchall()
    con.close()

    db_open_keys: dict[str, set[int]] = {}
    con = duckdb.connect(str(DB_PATH), read_only=True)
    for (repo,) in con.execute(
        "SELECT DISTINCT repo FROM prs WHERE state='OPEN'"
    ).fetchall():
        numbers = con.execute(
            "SELECT number FROM prs WHERE state='OPEN' AND repo=?", [repo]
        ).fetchall()
        db_open_keys[repo] = {n[0] for n in numbers}
    con.close()

    token = get_token()
    headers = {
        "Authorization": f"bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    fetch_now = datetime.now(UTC).isoformat()
    rows: list[dict[str, object]] = []

    with httpx.Client(headers=headers, timeout=60.0) as client:
        for idx, (repo, n_db_open) in enumerate(repo_counts, start=1):
            owner, name = repo.split("/", 1)
            print(
                f"[{idx}/{len(repo_counts)}] {repo} "
                f"(db_open={n_db_open})...",
                flush=True,
            )
            try:
                fetched = fetch_repo(client, owner, name)
            except httpx.HTTPStatusError as exc:
                print(
                    f"    HTTP error for {repo}: {exc}", file=sys.stderr,
                )
                continue
            relevant = [
                (num, ts) for (num, ts) in fetched
                if num in db_open_keys.get(repo, set())
            ]
            print(
                f"    fetched {len(fetched)} currently-open, "
                f"{len(relevant)} match db-open set",
                flush=True,
            )
            for num, ts in relevant:
                rows.append(
                    {
                        "repo": repo,
                        "number": num,
                        "updated_at": ts,
                        "fetch_now": fetch_now,
                    }
                )

    df = pd.DataFrame(rows)
    df["updated_at"] = pd.to_datetime(df["updated_at"])
    df["fetch_now"] = pd.to_datetime(df["fetch_now"])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    total_db_open = sum(len(v) for v in db_open_keys.values())
    print()
    print(f"Wrote {len(df)} rows to {OUT_PATH}")
    print(
        f"Coverage: {len(df)} / {total_db_open} db-OPEN PRs still currently "
        f"open ({100 * len(df) / total_db_open:.1f}%)"
    )
    print(
        f"Missing: {total_db_open - len(df)} PRs (closed/merged since snapshot"
        f" — treated as non-stale)"
    )


if __name__ == "__main__":
    main()
