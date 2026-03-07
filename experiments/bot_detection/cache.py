from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

logger = logging.getLogger(__name__)

_CREATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS prs (
    repo TEXT NOT NULL,
    number INTEGER NOT NULL,
    author TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    body TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMP NOT NULL,
    merged_at TIMESTAMP,
    closed_at TIMESTAMP,
    state TEXT NOT NULL,
    additions INTEGER NOT NULL DEFAULT 0,
    deletions INTEGER NOT NULL DEFAULT 0,
    files_changed INTEGER NOT NULL DEFAULT 0,
    labels JSON,
    outcome TEXT,
    stale_threshold_days REAL,
    source TEXT NOT NULL DEFAULT 'unknown',
    PRIMARY KEY (repo, number)
);

CREATE TABLE IF NOT EXISTS reviews (
    repo TEXT NOT NULL,
    pr_number INTEGER NOT NULL,
    reviewer TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT '',
    body TEXT NOT NULL DEFAULT '',
    submitted_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS commits (
    repo TEXT NOT NULL,
    pr_number INTEGER NOT NULL,
    sha TEXT NOT NULL,
    author TEXT NOT NULL DEFAULT '',
    message TEXT NOT NULL DEFAULT '',
    committed_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS authors (
    login TEXT PRIMARY KEY,
    account_created_at TIMESTAMP,
    followers INTEGER NOT NULL DEFAULT 0,
    public_repos INTEGER NOT NULL DEFAULT 0,
    is_bot BOOLEAN NOT NULL DEFAULT FALSE,
    ge_score REAL,
    ge_trust_level TEXT,
    account_status TEXT
);

CREATE INDEX IF NOT EXISTS idx_prs_author_created ON prs(author, created_at);
CREATE INDEX IF NOT EXISTS idx_prs_author_repo ON prs(author, repo, created_at);
CREATE INDEX IF NOT EXISTS idx_reviews_pr ON reviews(repo, pr_number);
CREATE INDEX IF NOT EXISTS idx_commits_pr ON commits(repo, pr_number);
"""


class BotDetectionDB:
    """DuckDB-backed storage for the bot detection experiment."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.con = duckdb.connect(str(db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        for stmt in _CREATE_SCHEMA.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self.con.execute(stmt)

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> BotDetectionDB:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ----------------------------------------------------------
    # Import: neoteny DuckDB
    # ----------------------------------------------------------

    def import_neoteny_cache(
        self,
        source_path: Path,
        repo_filter: list[str] | None = None,
    ) -> dict[str, int]:
        """Import PR data from a neoteny DuckDB cache (read-only).

        Returns dict with row counts for each table.
        """
        try:
            source = duckdb.connect(str(source_path), read_only=True)
        except duckdb.IOException as e:
            logger.warning("Cannot open %s: %s", source_path, e)
            return {"prs": 0, "reviews": 0, "commits": 0}

        try:
            rows = source.execute(
                "SELECT key, value FROM cache WHERE category = 'pr_metadata'"
            ).fetchall()
        finally:
            source.close()

        pr_count = 0
        review_count = 0
        commit_count = 0

        for _key, value_str in rows:
            try:
                data = json.loads(value_str)
            except (json.JSONDecodeError, TypeError):
                continue

            repo = data.get("repo", "")
            if repo_filter and repo not in repo_filter:
                continue

            number = data.get("number")
            if number is None or not repo:
                continue

            # Insert PR
            created_at = _parse_timestamp(data.get("created_at"))
            if created_at is None:
                continue

            author = data.get("author", "")
            if not author:
                continue

            try:
                self.con.execute(
                    """INSERT OR IGNORE INTO prs
                    (repo, number, author, title, body, created_at,
                     merged_at, closed_at, state, additions, deletions,
                     files_changed, labels, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'neoteny')""",
                    [
                        repo,
                        number,
                        author,
                        data.get("title", ""),
                        data.get("body", "") or "",
                        created_at,
                        _parse_timestamp(data.get("merged_at")),
                        _parse_timestamp(data.get("closed_at")),
                        data.get("state", "UNKNOWN"),
                        data.get("additions", 0) or 0,
                        data.get("deletions", 0) or 0,
                        data.get("files_changed", 0) or 0,
                        json.dumps(data.get("labels", [])),
                    ],
                )
                pr_count += 1
            except duckdb.ConstraintException:
                pass  # Duplicate, skip

            # Insert reviews
            for review in data.get("reviews", []) or []:
                submitted = _parse_timestamp(
                    review.get("submitted_at") or review.get("submittedAt")
                )
                if submitted is None:
                    continue
                reviewer = review.get("author", "")
                if not reviewer:
                    continue
                self.con.execute(
                    """INSERT INTO reviews
                    (repo, pr_number, reviewer, state, body, submitted_at)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    [
                        repo,
                        number,
                        reviewer,
                        review.get("state", ""),
                        review.get("body", "") or "",
                        submitted,
                    ],
                )
                review_count += 1

            # Insert commits
            for commit in data.get("commits", []) or []:
                committed = _parse_timestamp(commit.get("date"))
                if committed is None:
                    continue
                sha = commit.get("sha", "")
                if not sha:
                    continue
                self.con.execute(
                    """INSERT INTO commits
                    (repo, pr_number, sha, author, message, committed_at)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    [
                        repo,
                        number,
                        sha,
                        commit.get("author", ""),
                        commit.get("message", "") or "",
                        committed,
                    ],
                )
                commit_count += 1

        counts = {"prs": pr_count, "reviews": review_count, "commits": commit_count}
        logger.info("Imported from %s: %s", source_path.name, counts)
        return counts

    # ----------------------------------------------------------
    # Import: PR 27 JSONL data
    # ----------------------------------------------------------

    def import_pr27_data(
        self,
        pr27_dir: Path,
        repo_filter: list[str] | None = None,
    ) -> dict[str, int]:
        """Import PR data from PR 27's JSONL files.

        Skips PRs that already exist (neoteny data preferred).
        Returns dict with row counts.
        """
        prs_dir = pr27_dir / "prs"
        authors_dir = pr27_dir / "authors"
        pr_count = 0
        author_count = 0

        if prs_dir.exists():
            for jsonl_file in sorted(prs_dir.glob("*.jsonl")):
                repo_name = jsonl_file.stem.replace("__", "/")
                if repo_filter and repo_name not in repo_filter:
                    continue
                with open(jsonl_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        created_at = _parse_timestamp(data.get("created_at"))
                        if created_at is None:
                            continue
                        author = data.get("author_login", "")
                        if not author:
                            continue
                        try:
                            self.con.execute(
                                """INSERT OR IGNORE INTO prs
                                (repo, number, author, title, body, created_at,
                                 merged_at, closed_at, state, additions, deletions,
                                 files_changed, labels, source)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pr27')""",
                                [
                                    data.get("repo", repo_name),
                                    data.get("number", 0),
                                    author,
                                    data.get("title", ""),
                                    data.get("body", "") or "",
                                    created_at,
                                    _parse_timestamp(data.get("merged_at")),
                                    _parse_timestamp(data.get("closed_at")),
                                    data.get("state", "UNKNOWN"),
                                    data.get("additions", 0) or 0,
                                    data.get("deletions", 0) or 0,
                                    data.get("changed_files", 0) or 0,
                                    json.dumps(data.get("labels", [])),
                                ],
                            )
                            pr_count += 1
                        except duckdb.ConstraintException:
                            pass

        if authors_dir.exists():
            for json_file in sorted(authors_dir.glob("*.json")):
                with open(json_file) as f:
                    data = json.loads(f.read())
                login = data.get("login", json_file.stem)
                # Profile data is nested under contribution_data.profile
                profile = (
                    data.get("contribution_data", {}).get("profile", {})
                )
                try:
                    self.con.execute(
                        """INSERT OR IGNORE INTO authors
                        (login, account_created_at, followers, public_repos, is_bot)
                        VALUES (?, ?, ?, ?, ?)""",
                        [
                            login,
                            _parse_timestamp(profile.get("created_at")),
                            profile.get("followers_count", 0) or 0,
                            profile.get("public_repos_count", 0) or 0,
                            profile.get("is_bot", False),
                        ],
                    )
                    author_count += 1
                except duckdb.ConstraintException:
                    pass

        counts = {"prs": pr_count, "authors": author_count}
        logger.info("Imported from PR 27: %s", counts)
        return counts

    # ----------------------------------------------------------
    # Query helpers
    # ----------------------------------------------------------

    def get_all_prs(self) -> list[dict[str, Any]]:
        """Return all PRs as list of dicts."""
        return self.con.execute("SELECT * FROM prs").fetchdf().to_dict("records")

    def get_pr_count(self) -> int:
        """Return total number of PRs."""
        return self.con.execute("SELECT COUNT(*) FROM prs").fetchone()[0]  # type: ignore[index]

    def get_distinct_repos(self) -> list[str]:
        """Return sorted list of distinct repo names."""
        rows = self.con.execute(
            "SELECT DISTINCT repo FROM prs ORDER BY repo"
        ).fetchall()
        return [r[0] for r in rows]

    def get_distinct_authors(self) -> list[str]:
        """Return sorted list of distinct author logins."""
        rows = self.con.execute(
            "SELECT DISTINCT author FROM prs ORDER BY author"
        ).fetchall()
        return [r[0] for r in rows]

    def get_author_prs_before(
        self,
        author: str,
        exclude_repo: str,
        before: datetime,
    ) -> list[dict[str, Any]]:
        """Get author's PRs on other repos before a given time.

        This is the core anti-lookahead query: strict < on timestamp,
        and the test repo is excluded.
        """
        df = self.con.execute(
            """SELECT * FROM prs
            WHERE author = ?
              AND repo != ?
              AND created_at < ?
            ORDER BY created_at""",
            [author, exclude_repo, before],
        ).fetchdf()
        return df.to_dict("records")

    def get_pr_reviews(self, repo: str, pr_number: int) -> list[dict[str, Any]]:
        """Get reviews for a specific PR."""
        df = self.con.execute(
            "SELECT * FROM reviews WHERE repo = ? AND pr_number = ?",
            [repo, pr_number],
        ).fetchdf()
        return df.to_dict("records")

    def get_pr_commits(self, repo: str, pr_number: int) -> list[dict[str, Any]]:
        """Get commits for a specific PR."""
        df = self.con.execute(
            "SELECT * FROM commits WHERE repo = ? AND pr_number = ?",
            [repo, pr_number],
        ).fetchdf()
        return df.to_dict("records")

    def get_reviews_for_author_prs_before(
        self,
        author: str,
        exclude_repo: str,
        before: datetime,
    ) -> list[dict[str, Any]]:
        """Get all reviews on the author's prior PRs (other repos, before T)."""
        df = self.con.execute(
            """SELECT r.* FROM reviews r
            JOIN prs p ON r.repo = p.repo AND r.pr_number = p.number
            WHERE p.author = ?
              AND p.repo != ?
              AND p.created_at < ?
            ORDER BY r.submitted_at""",
            [author, exclude_repo, before],
        ).fetchdf()
        return df.to_dict("records")

    def get_commits_for_author_prs_before(
        self,
        author: str,
        exclude_repo: str,
        before: datetime,
    ) -> list[dict[str, Any]]:
        """Get all commits on the author's prior PRs (other repos, before T)."""
        df = self.con.execute(
            """SELECT c.* FROM commits c
            JOIN prs p ON c.repo = p.repo AND c.pr_number = p.number
            WHERE p.author = ?
              AND p.repo != ?
              AND p.created_at < ?
            ORDER BY c.committed_at""",
            [author, exclude_repo, before],
        ).fetchdf()
        return df.to_dict("records")

    def get_repo_prs(
        self,
        repo: str,
        state: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get all PRs for a specific repo, optionally filtered by state."""
        if state:
            df = self.con.execute(
                "SELECT * FROM prs WHERE repo = ? AND state = ?",
                [repo, state],
            ).fetchdf()
        else:
            df = self.con.execute(
                "SELECT * FROM prs WHERE repo = ?", [repo]
            ).fetchdf()
        return df.to_dict("records")

    def update_pr_outcome(
        self,
        repo: str,
        number: int,
        outcome: str,
        stale_threshold_days: float,
    ) -> None:
        """Update a PR's outcome classification."""
        self.con.execute(
            """UPDATE prs
            SET outcome = ?, stale_threshold_days = ?
            WHERE repo = ? AND number = ?""",
            [outcome, stale_threshold_days, repo, number],
        )

    def filter_bots(self, patterns: list[str]) -> int:
        """Delete PRs from known bot accounts. Returns count deleted."""
        # Collect all bot authors first, then delete once
        all_authors = self.con.execute(
            "SELECT DISTINCT author FROM prs"
        ).fetchall()
        bot_authors: set[str] = set()
        for pattern in patterns:
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
            except re.error:
                logger.warning("Invalid bot pattern: %s", pattern)
                continue
            for (author,) in all_authors:
                if compiled.search(author):
                    bot_authors.add(author)

        if not bot_authors:
            return 0

        count_before = self.get_pr_count()
        placeholders = ",".join(["?"] * len(bot_authors))
        self.con.execute(
            f"DELETE FROM prs WHERE author IN ({placeholders})",
            list(bot_authors),
        )
        count_after = self.get_pr_count()
        deleted = count_before - count_after
        logger.info("Filtered %d bot PRs (%d bot authors)", deleted, len(bot_authors))
        return deleted

    def get_author_info(self, login: str) -> dict[str, Any] | None:
        """Get author metadata from the authors table."""
        rows = self.con.execute(
            "SELECT * FROM authors WHERE login = ?", [login]
        ).fetchdf()
        if rows.empty:
            return None
        return rows.to_dict("records")[0]

    def get_closed_pr_count_before(
        self,
        author: str,
        exclude_repo: str,
        before: datetime,
    ) -> int:
        """Count author's closed (non-merged) PRs on other repos before T."""
        result = self.con.execute(
            """SELECT COUNT(*) FROM prs
            WHERE author = ?
              AND repo != ?
              AND created_at < ?
              AND state = 'CLOSED'
              AND merged_at IS NULL""",
            [author, exclude_repo, before],
        ).fetchone()
        return result[0] if result else 0  # type: ignore[index]

    def update_author_status(self, login: str, status: str) -> None:
        """Update account status for an author."""
        self.con.execute(
            "UPDATE authors SET account_status = ? WHERE login = ?",
            [status, login],
        )

    def get_authors_without_status(self) -> list[str]:
        """Get author logins that haven't been checked yet."""
        rows = self.con.execute(
            "SELECT login FROM authors WHERE account_status IS NULL ORDER BY login"
        ).fetchall()
        return [r[0] for r in rows]

    def get_suspended_authors(self) -> list[str]:
        """Get authors with suspended/deleted accounts."""
        rows = self.con.execute(
            "SELECT login FROM authors WHERE account_status = 'suspended' ORDER BY login"
        ).fetchall()
        return [r[0] for r in rows]

    def get_author_aggregate_stats(self) -> Any:
        """Compute per-author aggregate features via DuckDB.

        Returns a pandas DataFrame with one row per author.
        """
        return self.con.execute("""
            SELECT
                author AS login,
                COUNT(*) AS total_prs,
                COUNT(DISTINCT repo) AS total_repos,
                COALESCE(SUM(CASE WHEN state = 'MERGED' THEN 1 ELSE 0 END)::DOUBLE
                    / NULLIF(COUNT(*), 0), 0) AS merge_rate,
                COALESCE(SUM(CASE WHEN outcome = 'rejected' THEN 1 ELSE 0 END)::DOUBLE
                    / NULLIF(COUNT(*), 0), 0) AS rejection_rate,
                COALESCE(SUM(CASE WHEN outcome = 'pocket_veto' THEN 1 ELSE 0 END)::DOUBLE
                    / NULLIF(COUNT(*), 0), 0) AS pocket_veto_rate,
                MEDIAN(additions) AS median_additions,
                MEDIAN(deletions) AS median_deletions,
                MEDIAN(files_changed) AS median_files_changed,
                AVG(LENGTH(title)) AS mean_title_length,
                AVG(LENGTH(body)) AS mean_body_length,
                COALESCE(SUM(CASE WHEN body = '' OR body IS NULL THEN 1 ELSE 0 END)::DOUBLE
                    / NULLIF(COUNT(*), 0), 0) AS empty_body_rate,
                COUNT(DISTINCT DATE_TRUNC('day', created_at)) AS active_days,
                EXTRACT(EPOCH FROM (MAX(created_at) - MIN(created_at))) / 86400.0
                    AS career_span_days,
                COUNT(*)::DOUBLE
                    / NULLIF(COUNT(DISTINCT DATE_TRUNC('day', created_at)), 0)
                    AS prs_per_active_day,
                COUNT(DISTINCT repo)::DOUBLE
                    / NULLIF(COUNT(DISTINCT DATE_TRUNC('day', created_at)), 0)
                    AS repos_per_active_day
            FROM prs
            GROUP BY author
            ORDER BY author
        """).fetchdf()

    def get_all_author_repo_pairs(self) -> list[tuple[str, str, int]]:
        """Get all (author, repo, pr_count) triples for network analysis.

        Returns list of (author, repo, count) tuples.
        """
        rows = self.con.execute("""
            SELECT author, repo, COUNT(*) AS pr_count
            FROM prs
            GROUP BY author, repo
            ORDER BY author, repo
        """).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def get_author_pr_timestamps(self, login: str) -> list[datetime]:
        """Get sorted PR timestamps for a single author."""
        rows = self.con.execute(
            "SELECT created_at FROM prs WHERE author = ? ORDER BY created_at",
            [login],
        ).fetchall()
        return [r[0] for r in rows]

    def get_all_author_pr_timestamps(self) -> dict[str, list[datetime]]:
        """Get sorted PR timestamps for all authors, keyed by login."""
        rows = self.con.execute(
            "SELECT author, created_at FROM prs ORDER BY author, created_at"
        ).fetchall()
        result: dict[str, list[datetime]] = {}
        for author, ts in rows:
            result.setdefault(author, []).append(ts)
        return result

    def get_suspicious_authors(
        self,
        limit: int = 1000,
        min_repos: int = 2,
    ) -> list[str]:
        """Get most suspicious authors for ground truth checking.

        Selects multi-repo authors ordered by merge rate ascending.
        """
        rows = self.con.execute(
            """SELECT author
            FROM prs
            GROUP BY author
            HAVING COUNT(DISTINCT repo) >= ?
            ORDER BY SUM(CASE WHEN state = 'MERGED' THEN 1 ELSE 0 END)::DOUBLE
                / COUNT(*) ASC
            LIMIT ?""",
            [min_repos, limit],
        ).fetchall()
        return [r[0] for r in rows]

    def get_author_profile_fields(self) -> Any:
        """Get profile fields (account_age, followers, public_repos) for all authors.

        Returns a pandas DataFrame.
        """
        return self.con.execute("""
            SELECT
                login,
                account_created_at,
                followers,
                public_repos,
                account_status
            FROM authors
        """).fetchdf()

    def get_all_author_titles(self, limit_per_author: int = 50) -> dict[str, list[str]]:
        """Get PR titles for all authors, grouped by login.

        Returns dict mapping login -> list of titles (up to limit_per_author each).
        Uses a single SQL query with ROW_NUMBER windowing for efficiency.
        """
        rows = self.con.execute(
            """SELECT author, title FROM (
                SELECT author, title,
                    ROW_NUMBER() OVER (
                        PARTITION BY author ORDER BY created_at DESC
                    ) AS rn
                FROM prs
            ) sub
            WHERE rn <= ?
            ORDER BY author""",
            [limit_per_author],
        ).fetchall()
        result: dict[str, list[str]] = {}
        for author, title in rows:
            result.setdefault(author, []).append(title)
        return result

    def get_author_pr_titles(self, login: str, limit: int = 50) -> list[str]:
        """Get PR titles for an author, up to limit."""
        rows = self.con.execute(
            "SELECT title FROM prs WHERE author = ? ORDER BY created_at DESC LIMIT ?",
            [login, limit],
        ).fetchall()
        return [r[0] for r in rows]

    def get_author_pr_bodies(self, login: str, limit: int = 50) -> list[str]:
        """Get PR bodies for an author, up to limit."""
        rows = self.con.execute(
            "SELECT body FROM prs WHERE author = ? ORDER BY created_at DESC LIMIT ?",
            [login, limit],
        ).fetchall()
        return [r[0] for r in rows]

    # ----------------------------------------------------------
    # Import: OSS parquet data
    # ----------------------------------------------------------

    def import_oss_parquet(
        self,
        parquet_dir: Path,
        repo_filter: list[str] | None = None,
        min_merge_rate: float = 0.10,
    ) -> dict[str, int]:
        """Import PR data from neoteny OSS parquet files.

        Scans parquet_dir for subdirectories containing raw_prs.parquet.
        Skips repos below min_merge_rate (cherry-pick/phabricator workflows).
        Returns dict with row counts including skipped_repos.
        """
        import pandas as pd

        pr_count = 0
        repo_count = 0
        skipped_repos = 0

        parquet_files = sorted(parquet_dir.glob("*/raw_prs.parquet"))
        if not parquet_files:
            logger.warning("No parquet files found in %s", parquet_dir)
            return {"prs": 0, "repos": 0, "skipped_repos": 0}

        for pq_file in parquet_files:
            df = pd.read_parquet(pq_file)
            if df.empty:
                continue

            # Get repo name from the data (owner/repo format)
            repo = df["repo"].iloc[0]
            if repo_filter and repo not in repo_filter:
                continue

            # Compute merge rate and skip low-merge repos
            total = len(df)
            merged = (df["state"] == "MERGED").sum()
            merge_rate = merged / total if total > 0 else 0.0
            if merge_rate < min_merge_rate:
                logger.warning(
                    "Skipping %s: merge rate %.1f%% (%d/%d) below threshold %.0f%%",
                    repo, merge_rate * 100, merged, total, min_merge_rate * 100,
                )
                skipped_repos += 1
                continue

            # Strip timezone from timestamps
            for col in ["created_at", "merged_at", "closed_at"]:
                if col not in df.columns:
                    continue
                if hasattr(df[col].dtype, "tz") and df[col].dtype.tz is not None:
                    df[col] = df[col].dt.tz_localize(None)

            # Convert float columns to int with NaN -> 0
            for col in ["additions", "deletions", "files_changed"]:
                if col in df.columns:
                    df[col] = df[col].fillna(0).astype(int)

            # Filter out rows with empty/null author or created_at
            df = df.dropna(subset=["author", "created_at"])
            df = df[df["author"].str.strip() != ""]

            # Insert rows
            for _, row in df.iterrows():
                author = row.get("author", "")
                created_at = row.get("created_at")
                if pd.isna(created_at) or not author:
                    continue
                try:
                    self.con.execute(
                        """INSERT OR IGNORE INTO prs
                        (repo, number, author, title, body, created_at,
                         merged_at, closed_at, state, additions, deletions,
                         files_changed, labels, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'oss_parquet')""",
                        [
                            repo,
                            int(row["number"]),
                            author,
                            row.get("title", "") or "",
                            row.get("body", "") or "",
                            _to_naive_dt(created_at),
                            _to_naive_dt(row.get("merged_at")),
                            _to_naive_dt(row.get("closed_at")),
                            row.get("state", "UNKNOWN"),
                            int(row.get("additions", 0)),
                            int(row.get("deletions", 0)),
                            int(row.get("files_changed", 0)),
                            json.dumps([]),
                        ],
                    )
                    pr_count += 1
                except duckdb.ConstraintException:
                    pass

            repo_count += 1
            logger.info(
                "Imported %s: %d PRs (merge rate %.1f%%)",
                repo, len(df), merge_rate * 100,
            )

        counts = {"prs": pr_count, "repos": repo_count, "skipped_repos": skipped_repos}
        logger.info("Imported from OSS parquet: %s", counts)
        return counts

    def get_prs_dataframe(self) -> Any:
        """Return all PRs as a pandas DataFrame."""
        return self.con.execute("SELECT * FROM prs").fetchdf()


def _to_naive_dt(value: Any) -> datetime | None:
    """Convert a pandas Timestamp or similar to a naive datetime, or None."""
    import pandas as pd

    if value is None or pd.isna(value):
        return None
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime().replace(tzinfo=None)
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    return None


def _parse_timestamp(value: str | None) -> datetime | None:
    """Parse an ISO timestamp string, returning None for empty/null values."""
    if not value:
        return None
    try:
        # Handle various ISO formats
        value = value.rstrip("Z")
        if "+" in value:
            value = value[: value.index("+")]
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
