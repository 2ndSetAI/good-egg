from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from experiments.bot_detection.cache import BotDetectionDB


@pytest.fixture
def db(tmp_path: Path) -> BotDetectionDB:
    """Create a fresh in-memory-like DB for testing."""
    db = BotDetectionDB(tmp_path / "test.duckdb")
    yield db
    db.close()


@pytest.fixture
def populated_db(db: BotDetectionDB) -> BotDetectionDB:
    """DB with sample data inserted."""
    # Insert PRs
    for i, (repo, author, state, created, merged) in enumerate([
        ("org/repo-a", "alice", "MERGED", "2024-01-10", "2024-01-15"),
        ("org/repo-a", "alice", "CLOSED", "2024-02-01", None),
        ("org/repo-b", "alice", "MERGED", "2024-01-20", "2024-01-22"),
        ("org/repo-b", "bob", "MERGED", "2024-03-01", "2024-03-05"),
        ("org/repo-a", "bob", "CLOSED", "2024-03-10", None),
    ]):
        db.con.execute(
            """INSERT INTO prs (repo, number, author, title, created_at,
               merged_at, closed_at, state, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'test')""",
            [
                repo, i + 1, author, f"PR {i + 1}",
                datetime.fromisoformat(created),
                datetime.fromisoformat(merged) if merged else None,
                datetime.fromisoformat("2024-03-15") if state == "CLOSED" else None,
                state,
            ],
        )

    # Insert reviews
    db.con.execute(
        """INSERT INTO reviews (repo, pr_number, reviewer, state, body, submitted_at)
        VALUES (?, ?, ?, ?, ?, ?)""",
        ["org/repo-a", 1, "reviewer1", "APPROVED", "LGTM", datetime(2024, 1, 12)],
    )

    # Insert commits
    db.con.execute(
        """INSERT INTO commits (repo, pr_number, sha, author, message, committed_at)
        VALUES (?, ?, ?, ?, ?, ?)""",
        ["org/repo-a", 1, "abc123", "alice", "fix bug", datetime(2024, 1, 10)],
    )

    return db


class TestSchema:
    def test_tables_created(self, db: BotDetectionDB) -> None:
        tables = db.con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        assert "prs" in table_names
        assert "reviews" in table_names
        assert "commits" in table_names
        assert "authors" in table_names


class TestQueries:
    def test_get_pr_count(self, populated_db: BotDetectionDB) -> None:
        assert populated_db.get_pr_count() == 5

    def test_get_distinct_repos(self, populated_db: BotDetectionDB) -> None:
        repos = populated_db.get_distinct_repos()
        assert repos == ["org/repo-a", "org/repo-b"]

    def test_get_distinct_authors(self, populated_db: BotDetectionDB) -> None:
        authors = populated_db.get_distinct_authors()
        assert authors == ["alice", "bob"]

    def test_get_author_prs_before(self, populated_db: BotDetectionDB) -> None:
        """Anti-lookahead: should exclude test repo and future PRs."""
        # Alice's PRs on repos other than repo-a, before 2024-02-15
        prs = populated_db.get_author_prs_before(
            author="alice",
            exclude_repo="org/repo-a",
            before=datetime(2024, 2, 15),
        )
        assert len(prs) == 1
        assert prs[0]["repo"] == "org/repo-b"

    def test_anti_lookahead_strict_less_than(self, populated_db: BotDetectionDB) -> None:
        """Verify strict < (not <=) on timestamp."""
        # Alice's PR on repo-b was created 2024-01-20
        prs = populated_db.get_author_prs_before(
            author="alice",
            exclude_repo="org/repo-a",
            before=datetime(2024, 1, 20),  # Exactly equal
        )
        assert len(prs) == 0  # Should NOT include the PR at exactly this time

    def test_anti_lookahead_excludes_test_repo(self, populated_db: BotDetectionDB) -> None:
        """Verify test repo is excluded."""
        prs = populated_db.get_author_prs_before(
            author="alice",
            exclude_repo="org/repo-b",
            before=datetime(2025, 1, 1),
        )
        repos = {pr["repo"] for pr in prs}
        assert "org/repo-b" not in repos

    def test_get_pr_reviews(self, populated_db: BotDetectionDB) -> None:
        reviews = populated_db.get_pr_reviews("org/repo-a", 1)
        assert len(reviews) == 1
        assert reviews[0]["reviewer"] == "reviewer1"

    def test_get_pr_commits(self, populated_db: BotDetectionDB) -> None:
        commits = populated_db.get_pr_commits("org/repo-a", 1)
        assert len(commits) == 1
        assert commits[0]["sha"] == "abc123"

    def test_get_repo_prs(self, populated_db: BotDetectionDB) -> None:
        prs = populated_db.get_repo_prs("org/repo-a")
        assert len(prs) == 3

    def test_get_repo_prs_filtered(self, populated_db: BotDetectionDB) -> None:
        prs = populated_db.get_repo_prs("org/repo-a", state="MERGED")
        assert len(prs) == 1


class TestDedup:
    def test_insert_or_ignore(self, db: BotDetectionDB) -> None:
        """Verify that duplicate (repo, number) pairs are ignored."""
        db.con.execute(
            """INSERT INTO prs (repo, number, author, title, created_at, state, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ["org/repo", 1, "alice", "First", datetime(2024, 1, 1), "MERGED", "neoteny"],
        )
        db.con.execute(
            """INSERT OR IGNORE INTO prs (repo, number, author, title, created_at, state, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ["org/repo", 1, "alice", "Second", datetime(2024, 1, 1), "MERGED", "pr27"],
        )
        count = db.get_pr_count()
        assert count == 1
        # Original title preserved
        prs = db.get_repo_prs("org/repo")
        assert prs[0]["title"] == "First"


class TestBotFiltering:
    def test_filter_bots(self, populated_db: BotDetectionDB) -> None:
        """Insert a bot PR and verify it gets filtered."""
        populated_db.con.execute(
            """INSERT INTO prs (repo, number, author, title, created_at, state, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ["org/repo-a", 100, "dependabot[bot]", "Bump dep", datetime(2024, 1, 1),
             "MERGED", "test"],
        )
        assert populated_db.get_pr_count() == 6
        filtered = populated_db.filter_bots(["\\[bot\\]$"])
        assert filtered >= 1
        assert populated_db.get_pr_count() == 5

    def test_filter_app_prefix_bots(self, populated_db: BotDetectionDB) -> None:
        """Verify ^app/ pattern catches app/copybara-service style authors."""
        for i, author in enumerate(
            ["app/copybara-service", "app/dependabot"],
            start=200,
        ):
            populated_db.con.execute(
                """INSERT INTO prs (repo, number, author, title, created_at,
                   state, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                ["org/repo-a", i, author, "Bot PR", datetime(2024, 1, 1),
                 "MERGED", "test"],
            )
        assert populated_db.get_pr_count() == 7
        filtered = populated_db.filter_bots(["^app/"])
        assert filtered == 2
        assert populated_db.get_pr_count() == 5


class TestAccountStatus:
    def test_update_author_status(self, db: BotDetectionDB) -> None:
        db.con.execute(
            "INSERT INTO authors (login) VALUES (?)", ["alice"],
        )
        db.update_author_status("alice", "active")
        row = db.con.execute(
            "SELECT account_status FROM authors WHERE login = ?", ["alice"],
        ).fetchone()
        assert row[0] == "active"

    def test_get_authors_without_status(self, db: BotDetectionDB) -> None:
        db.con.execute("INSERT INTO authors (login) VALUES (?)", ["alice"])
        db.con.execute(
            "INSERT INTO authors (login, account_status) VALUES (?, ?)",
            ["bob", "active"],
        )
        result = db.get_authors_without_status()
        assert result == ["alice"]

    def test_get_suspended_authors(self, db: BotDetectionDB) -> None:
        db.con.execute(
            "INSERT INTO authors (login, account_status) VALUES (?, ?)",
            ["alice", "suspended"],
        )
        db.con.execute(
            "INSERT INTO authors (login, account_status) VALUES (?, ?)",
            ["bob", "active"],
        )
        result = db.get_suspended_authors()
        assert result == ["alice"]

    def test_get_suspended_authors_empty(self, db: BotDetectionDB) -> None:
        db.con.execute(
            "INSERT INTO authors (login, account_status) VALUES (?, ?)",
            ["alice", "active"],
        )
        assert db.get_suspended_authors() == []


class TestUpdateOutcome:
    def test_update_pr_outcome(self, populated_db: BotDetectionDB) -> None:
        populated_db.update_pr_outcome("org/repo-a", 1, "merged", 45.0)
        prs = populated_db.con.execute(
            "SELECT outcome, stale_threshold_days FROM prs WHERE repo = ? AND number = ?",
            ["org/repo-a", 1],
        ).fetchone()
        assert prs[0] == "merged"
        assert prs[1] == 45.0


class TestParquetImport:
    """Tests for import_oss_parquet()."""

    @pytest.fixture
    def parquet_dir(self, tmp_path: Path) -> Path:
        """Create a temp dir with sample parquet files."""
        repo_dir = tmp_path / "parquet" / "org_repo-a"
        repo_dir.mkdir(parents=True)

        # Create a parquet file with tz-aware timestamps and float columns
        df = pd.DataFrame({
            "number": [1, 2, 3, 4, 5],
            "repo": ["org/repo-a"] * 5,
            "title": ["PR 1", "PR 2", "PR 3", "PR 4", "PR 5"],
            "body": ["body 1", None, "body 3", "", "body 5"],
            "author": ["alice", "bob", "alice", "charlie", ""],
            "created_at": pd.to_datetime([
                "2024-01-10", "2024-02-01", "2024-03-01",
                "2024-04-01", "2024-05-01",
            ]).tz_localize("UTC"),
            "closed_at": pd.to_datetime([
                "2024-01-15", None, "2024-03-10", None, None,
            ]).tz_localize("UTC"),
            "merged_at": pd.to_datetime([
                "2024-01-15", None, "2024-03-10", None, None,
            ]).tz_localize("UTC"),
            "state": ["MERGED", "CLOSED", "MERGED", "OPEN", "MERGED"],
            "additions": [10.0, float("nan"), 5.0, 20.0, 1.0],
            "deletions": [3.0, 1.0, float("nan"), 0.0, 0.0],
            "files_changed": [2.0, float("nan"), 1.0, 3.0, 1.0],
            "review_count": [1, 0, 2, 0, 0],
            "outcome": ["MERGED", "REJECTED", "MERGED", "UNRESOLVED", "MERGED"],
            "pr_size": [13.0, 1.0, 5.0, 20.0, 1.0],
            "enrichment_status": ["complete"] * 5,
        })
        df.to_parquet(repo_dir / "raw_prs.parquet")
        return tmp_path / "parquet"

    def test_basic_import(self, db: BotDetectionDB, parquet_dir: Path) -> None:
        """Test that parquet data is imported correctly."""
        counts = db.import_oss_parquet(parquet_dir)
        # 5 rows but author="" on row 5 gets filtered out -> 4 rows
        assert counts["prs"] == 4
        assert counts["repos"] == 1
        assert counts["skipped_repos"] == 0

    def test_timezone_stripped(self, db: BotDetectionDB, parquet_dir: Path) -> None:
        """Timestamps should be naive (no timezone) after import."""
        db.import_oss_parquet(parquet_dir)
        row = db.con.execute(
            "SELECT created_at FROM prs WHERE number = 1"
        ).fetchone()
        ts = row[0]
        # DuckDB returns naive datetime
        assert ts.tzinfo is None
        assert ts == datetime(2024, 1, 10)

    def test_nan_to_zero(self, db: BotDetectionDB, parquet_dir: Path) -> None:
        """NaN in additions/deletions/files_changed should become 0."""
        db.import_oss_parquet(parquet_dir)
        row = db.con.execute(
            "SELECT additions, deletions, files_changed FROM prs WHERE number = 2"
        ).fetchone()
        assert row[0] == 0  # additions was NaN
        assert row[1] == 1  # deletions was 1.0
        assert row[2] == 0  # files_changed was NaN

    def test_empty_author_filtered(self, db: BotDetectionDB, parquet_dir: Path) -> None:
        """Rows with empty author should be skipped."""
        db.import_oss_parquet(parquet_dir)
        authors = db.get_distinct_authors()
        assert "" not in authors
        # PR 5 (empty author) should not be imported
        row = db.con.execute(
            "SELECT COUNT(*) FROM prs WHERE number = 5"
        ).fetchone()
        assert row[0] == 0

    def test_dedup_parquet_first(self, db: BotDetectionDB, parquet_dir: Path) -> None:
        """Parquet data imported first wins dedup (INSERT OR IGNORE)."""
        db.import_oss_parquet(parquet_dir)
        # Insert same repo/number with different source
        db.con.execute(
            """INSERT OR IGNORE INTO prs
            (repo, number, author, title, created_at, state, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ["org/repo-a", 1, "alice", "Different Title",
             datetime(2024, 1, 10), "MERGED", "neoteny"],
        )
        prs = db.get_repo_prs("org/repo-a")
        pr1 = [p for p in prs if p["number"] == 1][0]
        assert pr1["source"] == "oss_parquet"
        assert pr1["title"] == "PR 1"

    def test_low_merge_rate_skipped(self, db: BotDetectionDB, tmp_path: Path) -> None:
        """Repos with merge rate below threshold are skipped."""
        repo_dir = tmp_path / "pq2" / "low_merge_repo"
        repo_dir.mkdir(parents=True)
        df = pd.DataFrame({
            "number": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "repo": ["org/low-merge"] * 10,
            "title": [f"PR {i}" for i in range(10)],
            "body": [""] * 10,
            "author": ["alice"] * 10,
            "created_at": pd.to_datetime(["2024-01-01"] * 10).tz_localize("UTC"),
            "closed_at": pd.to_datetime(["2024-01-15"] * 10).tz_localize("UTC"),
            "merged_at": pd.to_datetime(
                [None] * 10  # 0% merge rate
            ).tz_localize("UTC"),
            "state": ["CLOSED"] * 10,
            "additions": [1.0] * 10,
            "deletions": [0.0] * 10,
            "files_changed": [1.0] * 10,
            "review_count": [0] * 10,
            "outcome": ["REJECTED"] * 10,
            "pr_size": [1.0] * 10,
            "enrichment_status": ["complete"] * 10,
        })
        df.to_parquet(repo_dir / "raw_prs.parquet")

        counts = db.import_oss_parquet(tmp_path / "pq2")
        assert counts["prs"] == 0
        assert counts["skipped_repos"] == 1
        assert db.get_pr_count() == 0
