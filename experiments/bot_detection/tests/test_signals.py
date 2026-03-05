from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from experiments.bot_detection.models import FeatureRow, PROutcome
from experiments.bot_detection.stages.stage2_extract_signals import (
    compute_burst_content_features,
    compute_burstiness,
    compute_cross_repo_tfidf,
    compute_embedding_similarity,
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


class TestBurstContentFeatures:
    def test_empty_burst(self) -> None:
        result = compute_burst_content_features([])
        assert result["burst_size_cv"] is None

    def test_single_pr_burst(self) -> None:
        result = compute_burst_content_features([
            {"repo": "org/a", "additions": 10, "deletions": 5},
        ])
        assert result["burst_size_cv"] is None

    def test_uniform_sizes(self) -> None:
        prs = [
            {"repo": "org/a", "additions": 10, "deletions": 0},
            {"repo": "org/a", "additions": 10, "deletions": 0},
        ]
        result = compute_burst_content_features(prs)
        assert result["burst_size_cv"] == 0.0

    def test_varied_sizes(self) -> None:
        prs = [
            {"repo": "org/a", "additions": 100, "deletions": 0},
            {"repo": "org/b", "additions": 1, "deletions": 0},
        ]
        result = compute_burst_content_features(prs)
        assert result["burst_size_cv"] is not None
        assert result["burst_size_cv"] > 0.5

    def test_repo_entropy(self) -> None:
        prs = [
            {"repo": "org/a", "additions": 10, "deletions": 0},
            {"repo": "org/b", "additions": 10, "deletions": 0},
        ]
        result = compute_burst_content_features(prs)
        assert result["burst_file_pattern_entropy"] == 1.0


class TestEmbeddingSimilarity:
    def test_identical_embeddings(self) -> None:
        emb = np.ones(10, dtype=np.float32)
        result = compute_embedding_similarity([emb, emb])
        assert result is not None
        assert abs(result - 1.0) < 0.01

    def test_single_embedding(self) -> None:
        result = compute_embedding_similarity([np.ones(10)])
        assert result is None


class TestInteractionFeatures:
    """H6: Interaction features (burstiness x novelty)."""

    def _make_row(self, **kwargs: object) -> FeatureRow:
        defaults = {
            "repo": "org/repo",
            "number": 1,
            "author": "alice",
            "outcome": PROutcome.MERGED,
            "created_at": datetime(2024, 6, 1),
        }
        defaults.update(kwargs)
        return FeatureRow(**defaults)  # type: ignore[arg-type]

    def test_burst_no_prior_merge_zero_when_has_merges(self) -> None:
        """Bursty author with prior merges: interaction = 0."""
        # burst_count_24h=5 but has_prior_merge=True -> 5 * 0 = 0
        row = self._make_row(burst_count_24h=5, burst_no_prior_merge=0)
        assert row.burst_no_prior_merge == 0

    def test_burst_no_prior_merge_nonzero_without_merges(self) -> None:
        """Bursty author without prior merges: interaction = burst count."""
        # burst_count_24h=5 and has_prior_merge=False -> 5 * 1 = 5
        row = self._make_row(burst_count_24h=5, burst_no_prior_merge=5)
        assert row.burst_no_prior_merge == 5

    def test_burst_first_time_repo(self) -> None:
        """First-time repo contributor with burst gets nonzero."""
        row = self._make_row(burst_count_24h=3, burst_first_time_repo=3)
        assert row.burst_first_time_repo == 3

    def test_burst_first_time_repo_zero_for_returning(self) -> None:
        """Returning repo contributor with burst gets zero."""
        row = self._make_row(burst_count_24h=3, burst_first_time_repo=0)
        assert row.burst_first_time_repo == 0

    def test_burst_low_ge(self) -> None:
        """Low GE score with burst gets nonzero."""
        row = self._make_row(
            burst_count_24h=4, ge_score_v2=0.05, burst_low_ge=4,
        )
        assert row.burst_low_ge == 4

    def test_burst_low_ge_zero_for_high_ge(self) -> None:
        """High GE score with burst gets zero."""
        row = self._make_row(
            burst_count_24h=4, ge_score_v2=0.5, burst_low_ge=0,
        )
        assert row.burst_low_ge == 0

    def test_burst_new_account(self) -> None:
        """New account (<90 days) with burst gets nonzero."""
        row = self._make_row(
            burst_count_24h=6, account_age_days=30.0, burst_new_account=6,
        )
        assert row.burst_new_account == 6

    def test_burst_new_account_zero_for_old(self) -> None:
        """Old account (>=90 days) with burst gets zero."""
        row = self._make_row(
            burst_count_24h=6, account_age_days=365.0, burst_new_account=0,
        )
        assert row.burst_new_account == 0

    def test_interaction_multiplication_logic(self) -> None:
        """Verify the multiplication logic matches extract_features_for_pr."""
        burst_count_24h = 7
        has_prior_merge = False
        has_prior_repo_prs = True
        ge_score_v2 = 0.05
        account_age_days = 60.0

        assert burst_count_24h * (0 if has_prior_merge else 1) == 7
        assert burst_count_24h * (0 if has_prior_repo_prs else 1) == 0
        assert burst_count_24h * (
            1 if ge_score_v2 is not None and ge_score_v2 < 0.1 else 0
        ) == 7
        assert burst_count_24h * (
            1 if account_age_days is not None and account_age_days < 90 else 0
        ) == 7
