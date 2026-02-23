"""Tests for the trust scorer."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from good_egg.config import GoodEggConfig
from good_egg.models import (
    MergedPR,
    RepoMetadata,
    TrustLevel,
    UserContributionData,
    UserProfile,
)
from good_egg.scorer import TrustScorer


def _make_config(**overrides: object) -> GoodEggConfig:
    return GoodEggConfig(**overrides)  # type: ignore[arg-type]


def _make_profile(
    login: str = "testuser",
    is_bot: bool = False,
    days_old: int = 500,
) -> UserProfile:
    return UserProfile(
        login=login,
        created_at=datetime.now(UTC) - timedelta(days=days_old),
        followers_count=10,
        public_repos_count=5,
        is_bot=is_bot,
    )


def _make_contribution_data(
    login: str = "testuser",
    is_bot: bool = False,
    days_old: int = 500,
    merged_prs: list[MergedPR] | None = None,
    repos: dict[str, RepoMetadata] | None = None,
    closed_pr_count: int = 0,
) -> UserContributionData:
    return UserContributionData(
        profile=_make_profile(login=login, is_bot=is_bot, days_old=days_old),
        merged_prs=merged_prs or [],
        contributed_repos=repos or {},
        closed_pr_count=closed_pr_count,
    )


def _sample_prs_and_repos() -> (
    tuple[list[MergedPR], dict[str, RepoMetadata]]
):
    prs = [
        MergedPR(
            repo_name_with_owner="elixir-lang/elixir",
            title="Fix pattern matching edge case",
            merged_at=datetime(2024, 6, 15, tzinfo=UTC),
            additions=45,
            deletions=12,
            changed_files=3,
        ),
        MergedPR(
            repo_name_with_owner="phoenixframework/phoenix",
            title="Add WebSocket compression support",
            merged_at=datetime(2024, 3, 10, tzinfo=UTC),
            additions=200,
            deletions=30,
            changed_files=8,
        ),
        MergedPR(
            repo_name_with_owner="nerves-project/nerves",
            title="Improve firmware update reliability",
            merged_at=datetime(2023, 11, 20, tzinfo=UTC),
            additions=120,
            deletions=45,
            changed_files=5,
        ),
    ]
    repos = {
        "elixir-lang/elixir": RepoMetadata(
            name_with_owner="elixir-lang/elixir",
            stargazer_count=23000,
            fork_count=3200,
            primary_language="Elixir",
        ),
        "phoenixframework/phoenix": RepoMetadata(
            name_with_owner="phoenixframework/phoenix",
            stargazer_count=20000,
            fork_count=2800,
            primary_language="Elixir",
        ),
        "nerves-project/nerves": RepoMetadata(
            name_with_owner="nerves-project/nerves",
            stargazer_count=2100,
            fork_count=180,
            primary_language="Elixir",
        ),
    }
    return prs, repos


class TestBotDetection:
    def test_bot_gets_bot_trust_level(self) -> None:
        scorer = TrustScorer(_make_config())
        data = _make_contribution_data(is_bot=True)
        result = scorer.score(data, "org/repo")

        assert result.trust_level == TrustLevel.BOT
        assert result.raw_score == 0.0
        assert result.normalized_score == 0.0
        assert result.flags["is_bot"] is True

    def test_bot_with_prs_still_classified_as_bot(self) -> None:
        scorer = TrustScorer(_make_config())
        prs, repos = _sample_prs_and_repos()
        data = _make_contribution_data(is_bot=True, merged_prs=prs, repos=repos)
        result = scorer.score(data, "org/repo")
        assert result.trust_level == TrustLevel.BOT


class TestInsufficientData:
    def test_no_prs_returns_unknown(self) -> None:
        scorer = TrustScorer(_make_config())
        data = _make_contribution_data(merged_prs=[])
        result = scorer.score(data, "org/repo")

        assert result.trust_level == TrustLevel.UNKNOWN
        assert result.flags["has_insufficient_data"] is True
        assert result.raw_score == 0.0

    def test_no_prs_preserves_login(self) -> None:
        scorer = TrustScorer(_make_config())
        data = _make_contribution_data(login="emptyuser", merged_prs=[])
        result = scorer.score(data, "org/repo")
        assert result.user_login == "emptyuser"


class TestScoring:
    def test_scoring_produces_nonzero_score(self) -> None:
        scorer = TrustScorer(_make_config())
        prs, repos = _sample_prs_and_repos()
        data = _make_contribution_data(merged_prs=prs, repos=repos)
        result = scorer.score(data, "my-org/my-elixir-app")

        assert result.raw_score > 0.0
        assert 0.0 <= result.normalized_score <= 1.0
        assert result.total_merged_prs == 3
        assert result.unique_repos_contributed == 3

    def test_scoring_metadata_has_graph_info(self) -> None:
        scorer = TrustScorer(_make_config())
        prs, repos = _sample_prs_and_repos()
        data = _make_contribution_data(merged_prs=prs, repos=repos)
        result = scorer.score(data, "my-org/my-app")

        assert "graph_nodes" in result.scoring_metadata
        assert "graph_edges" in result.scoring_metadata
        assert result.scoring_metadata["graph_nodes"] > 0
        assert result.scoring_metadata["graph_edges"] > 0

    def test_context_repo_set_correctly(self) -> None:
        scorer = TrustScorer(_make_config())
        prs, repos = _sample_prs_and_repos()
        data = _make_contribution_data(merged_prs=prs, repos=repos)
        result = scorer.score(data, "my-org/my-elixir-app")
        assert result.context_repo == "my-org/my-elixir-app"


class TestTrustLevelClassification:
    def test_high_trust(self) -> None:
        scorer = TrustScorer(_make_config())
        # High normalized score should yield HIGH
        level = scorer._classify(0.85, {"is_bot": False})
        assert level == TrustLevel.HIGH

    def test_medium_trust(self) -> None:
        scorer = TrustScorer(_make_config())
        level = scorer._classify(0.5, {"is_bot": False})
        assert level == TrustLevel.MEDIUM

    def test_low_trust(self) -> None:
        scorer = TrustScorer(_make_config())
        level = scorer._classify(0.1, {"is_bot": False})
        assert level == TrustLevel.LOW

    def test_bot_flag_overrides(self) -> None:
        scorer = TrustScorer(_make_config())
        level = scorer._classify(0.99, {"is_bot": True})
        assert level == TrustLevel.BOT

    def test_threshold_boundary_high(self) -> None:
        scorer = TrustScorer(_make_config())
        # Default high_trust threshold is 0.7
        level = scorer._classify(0.7, {"is_bot": False})
        assert level == TrustLevel.HIGH

    def test_threshold_boundary_medium(self) -> None:
        scorer = TrustScorer(_make_config())
        # Default medium_trust threshold is 0.3
        level = scorer._classify(0.3, {"is_bot": False})
        assert level == TrustLevel.MEDIUM

    def test_just_below_medium(self) -> None:
        scorer = TrustScorer(_make_config())
        level = scorer._classify(0.29, {"is_bot": False})
        assert level == TrustLevel.LOW


class TestNewAccountFlagging:
    def test_new_account_flagged(self) -> None:
        scorer = TrustScorer(_make_config())
        prs = [
            MergedPR(
                repo_name_with_owner="org/repo",
                title="PR",
                merged_at=datetime.now(UTC) - timedelta(days=1),
            ),
        ]
        repos = {
            "org/repo": RepoMetadata(
                name_with_owner="org/repo",
                stargazer_count=100,
            ),
        }
        data = _make_contribution_data(
            days_old=10, merged_prs=prs, repos=repos
        )
        result = scorer.score(data, "org/repo")
        assert result.flags["is_new_account"] is True

    def test_old_account_not_flagged(self) -> None:
        scorer = TrustScorer(_make_config())
        prs = [
            MergedPR(
                repo_name_with_owner="org/repo",
                title="PR",
                merged_at=datetime.now(UTC) - timedelta(days=1),
            ),
        ]
        repos = {
            "org/repo": RepoMetadata(
                name_with_owner="org/repo",
                stargazer_count=100,
            ),
        }
        data = _make_contribution_data(
            days_old=500, merged_prs=prs, repos=repos
        )
        result = scorer.score(data, "org/repo")
        assert result.flags["is_new_account"] is False


class TestLanguageMatch:
    def test_language_match_when_context_repo_in_data(self) -> None:
        scorer = TrustScorer(_make_config())
        prs = [
            MergedPR(
                repo_name_with_owner="org/elixir-app",
                title="PR",
                merged_at=datetime.now(UTC) - timedelta(days=5),
            ),
        ]
        repos = {
            "org/elixir-app": RepoMetadata(
                name_with_owner="org/elixir-app",
                stargazer_count=100,
                primary_language="Elixir",
            ),
        }
        data = _make_contribution_data(merged_prs=prs, repos=repos)
        result = scorer.score(data, "org/elixir-app")
        assert result.language_match is True

    def test_no_language_match(self) -> None:
        scorer = TrustScorer(_make_config())
        prs = [
            MergedPR(
                repo_name_with_owner="org/python-app",
                title="PR",
                merged_at=datetime.now(UTC) - timedelta(days=5),
            ),
        ]
        repos = {
            "org/python-app": RepoMetadata(
                name_with_owner="org/python-app",
                stargazer_count=100,
                primary_language="Python",
            ),
        }
        data = _make_contribution_data(merged_prs=prs, repos=repos)
        # Context repo not in contributed repos, so context_language is None
        result = scorer.score(data, "other-org/elixir-lib")
        assert result.language_match is False


class TestTopContributions:
    def test_top_contributions_sorted_by_count(self) -> None:
        scorer = TrustScorer(_make_config())
        prs = [
            MergedPR(
                repo_name_with_owner="org/repo-a",
                title="PR 1",
                merged_at=datetime.now(UTC) - timedelta(days=5),
            ),
            MergedPR(
                repo_name_with_owner="org/repo-b",
                title="PR 2",
                merged_at=datetime.now(UTC) - timedelta(days=5),
            ),
            MergedPR(
                repo_name_with_owner="org/repo-b",
                title="PR 3",
                merged_at=datetime.now(UTC) - timedelta(days=10),
            ),
        ]
        repos = {
            "org/repo-a": RepoMetadata(
                name_with_owner="org/repo-a", stargazer_count=100
            ),
            "org/repo-b": RepoMetadata(
                name_with_owner="org/repo-b", stargazer_count=200
            ),
        }
        data = _make_contribution_data(merged_prs=prs, repos=repos)
        result = scorer.score(data, "ctx/repo")

        assert len(result.top_contributions) == 2
        assert result.top_contributions[0].repo_name == "org/repo-b"
        assert result.top_contributions[0].pr_count == 2
        assert result.top_contributions[1].repo_name == "org/repo-a"
        assert result.top_contributions[1].pr_count == 1


class TestSuspectedBot:
    def test_suspected_bot_flag_propagated(self) -> None:
        scorer = TrustScorer(_make_config())
        prs = [
            MergedPR(
                repo_name_with_owner="org/repo",
                title="PR",
                merged_at=datetime.now(UTC) - timedelta(days=1),
            ),
        ]
        repos = {
            "org/repo": RepoMetadata(
                name_with_owner="org/repo",
                stargazer_count=100,
            ),
        }
        profile = UserProfile(
            login="ghost-user",
            created_at=datetime.now(UTC) - timedelta(days=500),
            followers_count=0,
            public_repos_count=0,
            is_suspected_bot=True,
        )
        data = UserContributionData(
            profile=profile,
            merged_prs=prs,
            contributed_repos=repos,
        )
        result = scorer.score(data, "org/repo")
        assert result.flags.get("is_suspected_bot") is True

    def test_normal_user_no_suspected_bot_flag(self) -> None:
        scorer = TrustScorer(_make_config())
        prs, repos = _sample_prs_and_repos()
        data = _make_contribution_data(merged_prs=prs, repos=repos)
        result = scorer.score(data, "org/repo")
        assert result.flags.get("is_suspected_bot") is None


class TestNormalization:
    def test_normalize_zero(self) -> None:
        import networkx as nx

        scorer = TrustScorer(_make_config())
        graph = nx.DiGraph()
        assert scorer._normalize(0.0, graph) == 0.0

    def test_normalize_in_range(self) -> None:
        import networkx as nx

        scorer = TrustScorer(_make_config())
        graph = nx.DiGraph()
        graph.add_nodes_from(["a", "b", "c"])
        result = scorer._normalize(0.5, graph)
        assert 0.0 <= result <= 1.0


class TestDiversityScoring:
    def test_diverse_cross_ecosystem_user_gets_nonzero_score(self) -> None:
        """User with many repos across ecosystems should score > 0 against unfamiliar repo."""
        scorer = TrustScorer(_make_config())
        prs = [
            MergedPR(
                repo_name_with_owner=f"org/repo-{i}",
                title=f"PR {i}",
                merged_at=datetime.now(UTC) - timedelta(days=i * 10),
            )
            for i in range(20)
        ]
        repos = {
            f"org/repo-{i}": RepoMetadata(
                name_with_owner=f"org/repo-{i}",
                stargazer_count=500 + i * 100,
                primary_language="Python",
            )
            for i in range(20)
        }
        data = _make_contribution_data(merged_prs=prs, repos=repos)
        # Score against a Rust repo (no language match)
        result = scorer.score(data, "rust-org/rust-project")
        assert result.normalized_score > 0.0
        assert result.total_merged_prs == 20
        assert result.unique_repos_contributed == 20


class TestV2Scoring:
    def test_v2_scoring_produces_nonzero_score(self) -> None:
        config = GoodEggConfig(scoring_model="v2")
        scorer = TrustScorer(config)
        prs, repos = _sample_prs_and_repos()
        data = _make_contribution_data(merged_prs=prs, repos=repos, closed_pr_count=5)
        result = scorer.score(data, "my-org/my-elixir-app")

        assert result.raw_score > 0.0
        assert 0.0 <= result.normalized_score <= 1.0
        assert result.scoring_model == "v2"

    def test_v2_component_scores_populated(self) -> None:
        config = GoodEggConfig(scoring_model="v2")
        scorer = TrustScorer(config)
        prs, repos = _sample_prs_and_repos()
        data = _make_contribution_data(merged_prs=prs, repos=repos, closed_pr_count=5)
        result = scorer.score(data, "my-org/my-elixir-app")

        assert "graph_score" in result.component_scores
        assert "merge_rate" in result.component_scores
        assert "log_account_age" in result.component_scores
        assert 0.0 <= result.component_scores["graph_score"] <= 1.0

    def test_v2_merge_rate_calculation(self) -> None:
        config = GoodEggConfig(scoring_model="v2")
        scorer = TrustScorer(config)
        prs, repos = _sample_prs_and_repos()
        # 3 merged PRs + 2 closed = 5 total, merge rate = 3/5 = 0.6
        data = _make_contribution_data(merged_prs=prs, repos=repos, closed_pr_count=2)
        result = scorer.score(data, "my-org/my-elixir-app")

        assert abs(result.component_scores["merge_rate"] - 0.6) < 1e-9

    def test_v2_merge_rate_zero_denominator(self) -> None:
        config = GoodEggConfig(scoring_model="v2")
        scorer = TrustScorer(config)
        prs = [
            MergedPR(
                repo_name_with_owner="org/repo",
                title="PR",
                merged_at=datetime.now(UTC) - timedelta(days=1),
            ),
        ]
        repos = {
            "org/repo": RepoMetadata(
                name_with_owner="org/repo",
                stargazer_count=100,
            ),
        }
        # 1 merged + 0 closed = should use merged_count / (merged + closed)
        data = _make_contribution_data(merged_prs=prs, repos=repos, closed_pr_count=0)
        result = scorer.score(data, "org/repo")

        # merge_rate = 1 / (1 + 0) = 1.0
        assert abs(result.component_scores["merge_rate"] - 1.0) < 1e-9

    def test_v2_feature_disabled_merge_rate(self) -> None:
        config = GoodEggConfig(
            scoring_model="v2",
            v2={"features": {"merge_rate": False}},
        )
        scorer = TrustScorer(config)
        prs, repos = _sample_prs_and_repos()
        data = _make_contribution_data(merged_prs=prs, repos=repos, closed_pr_count=5)
        result = scorer.score(data, "my-org/app")

        # merge_rate is still in component_scores for transparency
        assert "merge_rate" in result.component_scores
        assert result.scoring_model == "v2"

    def test_v2_bot_short_circuit(self) -> None:
        config = GoodEggConfig(scoring_model="v2")
        scorer = TrustScorer(config)
        data = _make_contribution_data(is_bot=True)
        result = scorer.score(data, "org/repo")

        assert result.trust_level == TrustLevel.BOT
        assert result.scoring_model == "v2"

    def test_v2_unknown_short_circuit(self) -> None:
        config = GoodEggConfig(scoring_model="v2")
        scorer = TrustScorer(config)
        data = _make_contribution_data(merged_prs=[])
        result = scorer.score(data, "org/repo")

        assert result.trust_level == TrustLevel.UNKNOWN
        assert result.scoring_model == "v2"

    def test_v1_mode_unchanged(self) -> None:
        """Regression: v1 scoring unchanged."""
        config = GoodEggConfig(scoring_model="v1")
        scorer = TrustScorer(config)
        prs, repos = _sample_prs_and_repos()
        data = _make_contribution_data(merged_prs=prs, repos=repos)
        result = scorer.score(data, "my-org/my-elixir-app")

        assert result.scoring_model == "v1"
        assert result.component_scores == {}
        assert result.raw_score > 0.0

    def test_v2_all_features_disabled(self) -> None:
        """With both features disabled, logit uses only intercept + graph_score."""
        config = GoodEggConfig(
            scoring_model="v2",
            v2={"features": {"merge_rate": False, "account_age": False}},
        )
        scorer = TrustScorer(config)
        prs, repos = _sample_prs_and_repos()
        data = _make_contribution_data(merged_prs=prs, repos=repos, closed_pr_count=5)
        result = scorer.score(data, "my-org/my-elixir-app")

        # Should still produce a valid sigmoid score in (0, 1)
        assert 0.0 < result.normalized_score < 1.0
        assert result.scoring_model == "v2"
        # Component scores should still be populated for transparency
        assert "graph_score" in result.component_scores
        assert "merge_rate" in result.component_scores
        assert "log_account_age" in result.component_scores

    def test_v2_negative_merge_rate_weight_behavior(self) -> None:
        """The negative merge_rate_weight means higher merge rate lowers the logit contribution.

        User with 100% merge rate vs user with 50% merge rate: the one with
        higher merge rate gets a more negative contribution from merge_rate_weight.
        Both should produce valid scores in (0, 1).
        """
        config = GoodEggConfig(scoring_model="v2")
        scorer = TrustScorer(config)

        prs = [
            MergedPR(
                repo_name_with_owner="org/repo",
                title="PR",
                merged_at=datetime.now(UTC) - timedelta(days=10),
            )
            for _ in range(10)
        ]
        repos = {
            "org/repo": RepoMetadata(
                name_with_owner="org/repo",
                stargazer_count=5000,
                primary_language="Python",
            ),
        }

        # User A: 100% merge rate (10 merged, 0 closed)
        data_high_mr = _make_contribution_data(
            merged_prs=prs, repos=repos, closed_pr_count=0
        )
        result_high_mr = scorer.score(data_high_mr, "org/repo")

        # User B: 50% merge rate (10 merged, 10 closed)
        data_low_mr = _make_contribution_data(
            merged_prs=prs, repos=repos, closed_pr_count=10
        )
        result_low_mr = scorer.score(data_low_mr, "org/repo")

        # Both produce valid scores
        assert 0.0 < result_high_mr.normalized_score < 1.0
        assert 0.0 < result_low_mr.normalized_score < 1.0

        # With negative merge_rate_weight (-0.7783), higher merge rate means
        # more negative contribution, so the high-merge-rate user gets a LOWER score
        assert result_high_mr.normalized_score < result_low_mr.normalized_score

    def test_v2_opposing_signals(self) -> None:
        """User with high graph score but low merge rate (many closed PRs).

        The combined model should still produce a valid score in (0, 1)
        and component_scores should be populated correctly.
        """
        config = GoodEggConfig(scoring_model="v2")
        scorer = TrustScorer(config)

        # Many merged PRs to popular repos -> high graph score
        prs = [
            MergedPR(
                repo_name_with_owner=f"big-org/popular-{i}",
                title=f"Major contribution {i}",
                merged_at=datetime.now(UTC) - timedelta(days=i * 5),
            )
            for i in range(15)
        ]
        repos = {
            f"big-org/popular-{i}": RepoMetadata(
                name_with_owner=f"big-org/popular-{i}",
                stargazer_count=10000 + i * 1000,
                primary_language="Python",
            )
            for i in range(15)
        }

        # 15 merged + 30 closed = 33% merge rate (low)
        data = _make_contribution_data(
            merged_prs=prs, repos=repos, closed_pr_count=30
        )
        result = scorer.score(data, "big-org/popular-0")

        # Valid score in (0, 1)
        assert 0.0 < result.normalized_score < 1.0
        assert result.scoring_model == "v2"

        # Component scores are populated
        assert "graph_score" in result.component_scores
        assert "merge_rate" in result.component_scores
        assert "log_account_age" in result.component_scores

        # Merge rate should be 15 / (15 + 30) = 1/3
        expected_mr = 15.0 / 45.0
        assert abs(result.component_scores["merge_rate"] - expected_mr) < 1e-9

    def test_v2_sigmoid_computation(self) -> None:
        """Verify the combined model produces reasonable sigmoid output."""
        config = GoodEggConfig(scoring_model="v2")
        scorer = TrustScorer(config)
        prs, repos = _sample_prs_and_repos()
        data = _make_contribution_data(merged_prs=prs, repos=repos, closed_pr_count=5)
        result = scorer.score(data, "my-org/my-elixir-app")

        # The normalized score should be a valid probability from sigmoid
        assert 0.0 < result.normalized_score < 1.0
