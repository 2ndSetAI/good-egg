from __future__ import annotations

from datetime import datetime
from pathlib import Path

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
