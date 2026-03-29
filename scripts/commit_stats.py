#!/usr/bin/env python3
"""Gather weekly commit stats across all repos and branches for a GitHub user."""
from __future__ import annotations

import csv
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import httpx

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
USERNAME = "jeffreyksmithjr"
SINCE = (datetime.now(timezone.utc) - timedelta(days=2 * 365)).strftime("%Y-%m-%dT%H:%M:%SZ")
HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
BASE = "https://api.github.com"


def week_key(dt: datetime) -> tuple[str, str]:
    """Return (week_start_date, YYYY-WNN) for a datetime."""
    iso = dt.isocalendar()
    # Monday of the ISO week
    monday = dt - timedelta(days=dt.weekday())
    return monday.strftime("%Y-%m-%d"), f"{iso[0]}-W{iso[1]:02d}"


def paginate(client: httpx.Client, url: str, params: dict | None = None) -> list[dict]:
    results: list[dict] = []
    page = 1
    while True:
        p = {**(params or {}), "per_page": 100, "page": page}
        for attempt in range(5):
            try:
                r = client.get(url, params=p, timeout=30)
                if r.status_code == 409:  # empty repo
                    return results
                if r.status_code == 422:  # unprocessable (e.g. no commits)
                    return results
                if r.status_code == 404:
                    return results
                if r.status_code == 403 or r.status_code == 429:
                    reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
                    wait = max(reset - int(time.time()), 1)
                    print(f"  Rate limited, waiting {wait}s...", file=sys.stderr)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                break
            except httpx.RequestError as e:
                wait = 2 ** attempt
                print(f"  Request error: {e}, retrying in {wait}s...", file=sys.stderr)
                time.sleep(wait)
        else:
            print(f"  Giving up on {url}", file=sys.stderr)
            return results

        data = r.json()
        if not isinstance(data, list) or not data:
            break
        results.extend(data)
        if len(data) < 100:
            break
        page += 1
    return results


def get_repos(client: httpx.Client) -> list[dict]:
    print("Fetching repos...", file=sys.stderr)
    repos = paginate(client, f"{BASE}/user/repos", {"affiliation": "owner,collaborator,organization_member", "sort": "updated"})
    print(f"  Found {len(repos)} repos", file=sys.stderr)
    return repos


def get_branches(client: httpx.Client, owner: str, repo: str) -> list[str]:
    branches = paginate(client, f"{BASE}/repos/{owner}/{repo}/branches")
    return [b["name"] for b in branches]


def get_commits_for_branch(
    client: httpx.Client, owner: str, repo: str, branch: str
) -> list[dict]:
    return paginate(
        client,
        f"{BASE}/repos/{owner}/{repo}/commits",
        {"author": USERNAME, "since": SINCE, "sha": branch},
    )


def main() -> None:
    if not GITHUB_TOKEN:
        print("GITHUB_TOKEN not set", file=sys.stderr)
        sys.exit(1)

    # week_start -> repo_full_name -> set of commit SHAs
    weekly: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    # week_start -> week_label
    week_labels: dict[str, str] = {}

    with httpx.Client(headers=HEADERS) as client:
        repos = get_repos(client)

        for repo in repos:
            full_name: str = repo["full_name"]
            owner, repo_name = full_name.split("/", 1)
            print(f"Processing {full_name}...", file=sys.stderr)

            try:
                branches = get_branches(client, owner, repo_name)
            except Exception as e:
                print(f"  Skipping branches: {e}", file=sys.stderr)
                continue

            if not branches:
                branches = [repo.get("default_branch", "main")]

            repo_shas: set[str] = set()  # dedup across branches within this repo

            for branch in branches:
                commits = get_commits_for_branch(client, owner, repo_name, branch)
                for commit in commits:
                    sha = commit["sha"]
                    if sha in repo_shas:
                        continue
                    repo_shas.add(sha)

                    # Parse commit date
                    date_str = (
                        commit.get("commit", {})
                        .get("author", {})
                        .get("date", "")
                    )
                    if not date_str:
                        continue
                    try:
                        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    except ValueError:
                        continue

                    wstart, wlabel = week_key(dt)
                    week_labels[wstart] = wlabel
                    weekly[wstart][full_name].add(sha)

            count = sum(len(v) for v in weekly.values() for k, v2 in [(k2, v2) for k2, v2 in v.items() if k2 == full_name] for v in [v2])
            print(f"  {len(repo_shas)} commits found", file=sys.stderr)

    # --- Write per-repo CSV ---
    output_path = "commit_stats_by_repo.csv"
    all_weeks = sorted(week_labels.keys())
    all_repos = sorted({repo for week in weekly.values() for repo in week})

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["week_start", "week", "repo", "commit_count"])
        for wstart in all_weeks:
            wlabel = week_labels[wstart]
            for repo in all_repos:
                count = len(weekly[wstart].get(repo, set()))
                if count > 0:
                    writer.writerow([wstart, wlabel, repo, count])

    print(f"Wrote {output_path}", file=sys.stderr)

    # --- Write weekly summary CSV ---
    summary_path = "commit_stats_weekly.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["week_start", "week", "total_commits", "repos_active"])
        for wstart in all_weeks:
            wlabel = week_labels[wstart]
            total = sum(len(shas) for shas in weekly[wstart].values())
            repos_active = len(weekly[wstart])
            writer.writerow([wstart, wlabel, total, repos_active])

    print(f"Wrote {summary_path}", file=sys.stderr)
    print(f"\nDone. {len(all_weeks)} weeks, {len(all_repos)} repos with commits.", file=sys.stderr)


if __name__ == "__main__":
    main()
