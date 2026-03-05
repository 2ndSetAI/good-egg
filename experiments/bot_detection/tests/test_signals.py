from __future__ import annotations

from datetime import datetime, timedelta

from experiments.bot_detection.stages.stage2_extract_signals import (
    compute_burstiness,
    compute_cross_repo_tfidf,
    compute_engagement,
)


class TestBurstiness:
    def test_empty_prs(self) -> None:
        result = compute_burstiness([], datetime(2024, 6, 1))
        assert result.burst_count_1h == 0
        assert result.burst_max_rate == 0.0

    def test_single_pr(self) -> None:
        t = datetime(2024, 6, 1, 12, 0)
        prs = [
            {"repo": "org/repo-a", "created_at": t - timedelta(minutes=30)},
        ]
        result = compute_burstiness(prs, t)
        assert result.burst_count_1h == 1
        assert result.burst_repos_1h == 1
        assert result.burst_count_24h == 1

    def test_burst_in_1h_window(self) -> None:
        t = datetime(2024, 6, 1, 12, 0)
        prs = [
            {"repo": "org/repo-a", "created_at": t - timedelta(minutes=10)},
            {"repo": "org/repo-b", "created_at": t - timedelta(minutes=20)},
            {"repo": "org/repo-c", "created_at": t - timedelta(minutes=30)},
        ]
        result = compute_burstiness(prs, t)
        assert result.burst_count_1h == 3
        assert result.burst_repos_1h == 3

    def test_outside_1h_window(self) -> None:
        t = datetime(2024, 6, 1, 12, 0)
        prs = [
            {"repo": "org/repo-a", "created_at": t - timedelta(hours=2)},
        ]
        result = compute_burstiness(prs, t)
        assert result.burst_count_1h == 0
        assert result.burst_count_24h == 1

    def test_24h_window(self) -> None:
        t = datetime(2024, 6, 1, 12, 0)
        prs = [
            {"repo": f"org/repo-{i}", "created_at": t - timedelta(hours=i)}
            for i in range(1, 20)
        ]
        result = compute_burstiness(prs, t)
        assert result.burst_count_1h == 1  # Only the one within 1h
        assert result.burst_count_24h == 19  # All within 24h
        assert result.burst_repos_24h == 19

    def test_max_rate(self) -> None:
        t = datetime(2024, 6, 1, 12, 0)
        # 5 PRs in a 30-minute burst, then nothing
        prs = [
            {"repo": f"org/repo-{i}", "created_at": t - timedelta(days=2, minutes=i * 5)}
            for i in range(5)
        ]
        result = compute_burstiness(prs, t)
        assert result.burst_max_rate == 5.0  # All 5 fit in 1h window

    def test_anti_lookahead_already_applied(self) -> None:
        """Burstiness expects pre-filtered data (created_at < T, repo != test)."""
        t = datetime(2024, 6, 1, 12, 0)
        # These should already be filtered by the caller
        prs = [
            {"repo": "org/repo-a", "created_at": t - timedelta(hours=1)},
        ]
        result = compute_burstiness(prs, t)
        assert result.burst_count_1h == 1


class TestEngagement:
    def test_empty_prs(self) -> None:
        result = compute_engagement([], {}, {}, "alice")
        assert result.review_response_rate is None
        assert result.abandoned_pr_rate is None

    def test_response_rate(self) -> None:
        prs = [
            {"repo": "org/a", "number": 1, "state": "MERGED", "merged_at": True,
             "created_at": datetime(2024, 1, 1)},
        ]
        reviews = {
            ("org/a", 1): [
                {"reviewer": "bob", "state": "COMMENTED", "submitted_at": datetime(2024, 1, 2)},
                {"reviewer": "alice", "state": "COMMENTED", "submitted_at": datetime(2024, 1, 3)},
            ]
        }
        commits: dict = {("org/a", 1): []}
        result = compute_engagement(prs, reviews, commits, "alice")
        assert result.review_response_rate == 1.0

    def test_no_response(self) -> None:
        prs = [
            {"repo": "org/a", "number": 1, "state": "MERGED", "merged_at": True,
             "created_at": datetime(2024, 1, 1)},
        ]
        reviews = {
            ("org/a", 1): [
                {"reviewer": "bob", "state": "COMMENTED", "submitted_at": datetime(2024, 1, 2)},
            ]
        }
        commits: dict = {("org/a", 1): []}
        result = compute_engagement(prs, reviews, commits, "alice")
        assert result.review_response_rate == 0.0

    def test_abandoned_pr_rate(self) -> None:
        prs = [
            {"repo": "org/a", "number": 1, "state": "CLOSED", "merged_at": None,
             "created_at": datetime(2024, 1, 1)},
            {"repo": "org/b", "number": 2, "state": "CLOSED", "merged_at": None,
             "created_at": datetime(2024, 1, 5)},
        ]
        reviews: dict = {("org/a", 1): [], ("org/b", 2): []}
        commits: dict = {("org/a", 1): [], ("org/b", 2): []}
        result = compute_engagement(prs, reviews, commits, "alice")
        assert result.abandoned_pr_rate == 1.0

    def test_changes_requested_followup(self) -> None:
        prs = [
            {"repo": "org/a", "number": 1, "state": "MERGED", "merged_at": True,
             "created_at": datetime(2024, 1, 1)},
        ]
        reviews = {
            ("org/a", 1): [
                {"reviewer": "bob", "state": "CHANGES_REQUESTED",
                 "submitted_at": datetime(2024, 1, 2)},
            ]
        }
        commits = {
            ("org/a", 1): [
                {"sha": "abc", "author": "alice", "committed_at": datetime(2024, 1, 3)},
            ]
        }
        result = compute_engagement(prs, reviews, commits, "alice")
        assert result.ci_failure_followup_rate == 1.0


class TestCrossRepoTfidf:
    def test_empty_prs(self) -> None:
        result = compute_cross_repo_tfidf([])
        assert result.max_title_similarity == 0.0

    def test_single_pr(self) -> None:
        result = compute_cross_repo_tfidf([
            {"repo": "org/a", "title": "Fix bug"},
        ])
        assert result.max_title_similarity == 0.0

    def test_identical_titles_different_repos(self) -> None:
        result = compute_cross_repo_tfidf([
            {"repo": "org/a", "title": "Update README.md with installation instructions"},
            {"repo": "org/b", "title": "Update README.md with installation instructions"},
        ])
        assert result.max_title_similarity > 0.9
        assert result.duplicate_title_count >= 1

    def test_different_titles(self) -> None:
        result = compute_cross_repo_tfidf([
            {"repo": "org/a", "title": "Fix memory leak in connection pool"},
            {"repo": "org/b", "title": "Add dark mode toggle to settings page"},
        ])
        assert result.max_title_similarity < 0.5

    def test_language_entropy(self) -> None:
        prs = [
            {"repo": f"org/repo-{i}", "title": f"PR {i}"}
            for i in range(10)
        ]
        result = compute_cross_repo_tfidf(prs)
        # 10 distinct repos = high entropy
        assert result.language_entropy > 2.0

    def test_same_repo_similarity_excluded(self) -> None:
        """Similarity should only be computed across different repos."""
        result = compute_cross_repo_tfidf([
            {"repo": "org/a", "title": "Fix bug in parser"},
            {"repo": "org/a", "title": "Fix bug in parser"},  # Same repo
            {"repo": "org/b", "title": "Add new API endpoint"},
        ])
        # The identical titles are from the same repo, so max_sim
        # should come from the cross-repo comparison
        assert result.duplicate_title_count == 0  # Same-repo dupes excluded
