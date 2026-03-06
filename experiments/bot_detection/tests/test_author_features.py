from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from experiments.bot_detection.cache import BotDetectionDB


@pytest.fixture
def db(tmp_path: Path) -> BotDetectionDB:
    db = BotDetectionDB(tmp_path / "test.duckdb")
    yield db
    db.close()


def _insert_pr(
    db: BotDetectionDB,
    repo: str,
    number: int,
    author: str,
    state: str,
    created: str,
    merged: str | None = None,
    title: str = "PR title",
    body: str = "PR body",
) -> None:
    db.con.execute(
        """INSERT INTO prs (repo, number, author, title, body, created_at,
           merged_at, state, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'test')""",
        [
            repo, number, author, title, body,
            datetime.fromisoformat(created),
            datetime.fromisoformat(merged) if merged else None,
            state,
        ],
    )


@pytest.fixture
def feature_db(db: BotDetectionDB) -> BotDetectionDB:
    """DB with multiple authors and repos for aggregate stat testing."""
    # alice: 3 PRs across 2 repos, 2 merged, 1 closed
    _insert_pr(db, "org/repo-a", 1, "alice", "MERGED", "2024-01-10", "2024-01-15")
    _insert_pr(db, "org/repo-a", 2, "alice", "CLOSED", "2024-02-01")
    _insert_pr(db, "org/repo-b", 3, "alice", "MERGED", "2024-03-01", "2024-03-05")
    # bob: 2 PRs in 1 repo, both merged
    _insert_pr(db, "org/repo-a", 4, "bob", "MERGED", "2024-01-20", "2024-01-25")
    _insert_pr(db, "org/repo-a", 5, "bob", "MERGED", "2024-02-20", "2024-02-25")
    # carol: 4 PRs across 3 repos, 1 merged
    _insert_pr(db, "org/repo-a", 6, "carol", "CLOSED", "2024-01-05")
    _insert_pr(db, "org/repo-b", 7, "carol", "CLOSED", "2024-01-10")
    _insert_pr(db, "org/repo-c", 8, "carol", "CLOSED", "2024-02-01")
    _insert_pr(db, "org/repo-c", 9, "carol", "MERGED", "2024-03-01", "2024-03-10")
    return db


class TestGetAuthorAggregateStats:
    def test_returns_correct_authors(self, feature_db: BotDetectionDB) -> None:
        df = feature_db.get_author_aggregate_stats()
        assert set(df["login"]) == {"alice", "bob", "carol"}

    def test_merge_rate_alice(self, feature_db: BotDetectionDB) -> None:
        df = feature_db.get_author_aggregate_stats()
        alice = df[df["login"] == "alice"].iloc[0]
        # 2 merged out of 3
        assert abs(alice["merge_rate"] - 2 / 3) < 1e-6

    def test_merge_rate_bob(self, feature_db: BotDetectionDB) -> None:
        df = feature_db.get_author_aggregate_stats()
        bob = df[df["login"] == "bob"].iloc[0]
        assert abs(bob["merge_rate"] - 1.0) < 1e-6

    def test_total_prs_per_author(self, feature_db: BotDetectionDB) -> None:
        df = feature_db.get_author_aggregate_stats()
        expected = {"alice": 3, "bob": 2, "carol": 4}
        for _, row in df.iterrows():
            assert row["total_prs"] == expected[row["login"]]

    def test_total_prs_sum(self, feature_db: BotDetectionDB) -> None:
        """total_prs per author should sum to total PR count in DB."""
        df = feature_db.get_author_aggregate_stats()
        assert df["total_prs"].sum() == feature_db.get_pr_count()

    def test_total_repos(self, feature_db: BotDetectionDB) -> None:
        df = feature_db.get_author_aggregate_stats()
        expected = {"alice": 2, "bob": 1, "carol": 3}
        for _, row in df.iterrows():
            assert row["total_repos"] == expected[row["login"]]

    def test_no_nan_in_core_columns(self, feature_db: BotDetectionDB) -> None:
        df = feature_db.get_author_aggregate_stats()
        core = [
            "total_prs", "total_repos", "merge_rate", "rejection_rate",
            "pocket_veto_rate", "empty_body_rate", "active_days",
            "prs_per_active_day",
        ]
        for col in core:
            assert df[col].isna().sum() == 0, f"NaN found in {col}"

    def test_merge_rate_bounds(self, feature_db: BotDetectionDB) -> None:
        df = feature_db.get_author_aggregate_stats()
        assert (df["merge_rate"] >= 0).all()
        assert (df["merge_rate"] <= 1).all()

    def test_rejection_rate_bounds(self, feature_db: BotDetectionDB) -> None:
        df = feature_db.get_author_aggregate_stats()
        assert (df["rejection_rate"] >= 0).all()
        assert (df["rejection_rate"] <= 1).all()


class TestGetSuspiciousAuthors:
    def test_returns_ordered_by_merge_rate(self, feature_db: BotDetectionDB) -> None:
        authors = feature_db.get_suspicious_authors(limit=10, min_repos=1)
        # carol has lowest merge rate (1/4 = 0.25), alice next (2/3 = 0.67),
        # bob last (1.0)
        assert authors == ["carol", "alice", "bob"]

    def test_min_repos_filter(self, feature_db: BotDetectionDB) -> None:
        # bob only has 1 repo, so min_repos=2 should exclude him
        authors = feature_db.get_suspicious_authors(limit=10, min_repos=2)
        assert "bob" not in authors
        assert "alice" in authors
        assert "carol" in authors

    def test_limit(self, feature_db: BotDetectionDB) -> None:
        authors = feature_db.get_suspicious_authors(limit=1, min_repos=1)
        assert len(authors) == 1
        assert authors[0] == "carol"  # lowest merge rate
