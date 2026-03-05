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
    ge_trust_level TEXT
);
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
                submitted = _parse_timestamp(review.get("submittedAt"))
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
                try:
                    self.con.execute(
                        """INSERT OR IGNORE INTO authors
                        (login, account_created_at, followers, public_repos, is_bot)
                        VALUES (?, ?, ?, ?, ?)""",
                        [
                            login,
                            _parse_timestamp(data.get("created_at")),
                            data.get("followers", 0) or 0,
                            data.get("public_repos", 0) or 0,
                            False,
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
        total = 0
        for pattern in patterns:
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
            except re.error:
                logger.warning("Invalid bot pattern: %s", pattern)
                continue
            # Get matching authors
            authors = self.con.execute(
                "SELECT DISTINCT author FROM prs"
            ).fetchall()
            bot_authors = [a[0] for a in authors if compiled.search(a[0])]
            if bot_authors:
                placeholders = ",".join(["?"] * len(bot_authors))
                result = self.con.execute(
                    f"DELETE FROM prs WHERE author IN ({placeholders})",
                    bot_authors,
                )
                count = result.fetchone()[0] if result else 0  # type: ignore[index]
                total += count
        logger.info("Filtered %d bot PRs", total)
        return total

    def get_prs_dataframe(self) -> Any:
        """Return all PRs as a pandas DataFrame."""
        return self.con.execute("SELECT * FROM prs").fetchdf()


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
