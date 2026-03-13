"""Tests for the trust scorer."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from good_egg.config import (
    BadEggConfig,
    BadEggModelConfig,
    BadEggThresholds,
    GoodEggConfig,
)
from good_egg.models import (
    MergedPR,
    RepoMetadata,
    SuspicionLevel,
    TrustLevel,
    UserContributionData,
    UserProfile,
)
from good_egg.scorer import TrustScorer, score_pr_author


def _make_config(**overrides: object) -> GoodEggConfig:
    if "scoring_model" not in overrides:
        overrides["scoring_model"] = "v1"
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
    repo_contributors: dict[str, list[str]] | None = None,
) -> UserContributionData:
    return UserContributionData(
        profile=_make_profile(login=login, is_bot=is_bot, days_old=days_old),
        merged_prs=merged_prs or [],
        contributed_repos=repos or {},
        closed_pr_count=closed_pr_count,
        repo_contributors=repo_contributors or {},
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

        expected = 1.0 / (1.0 + math.exp(-result.raw_score))
        assert abs(result.normalized_score - expected) < 1e-9
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


class TestExistingContributorSkip:
    @pytest.mark.asyncio
    @patch("good_egg.github_client.GitHubClient")
    async def test_skip_when_contributor_exists(
        self, mock_client_cls: AsyncMock
    ) -> None:
        """score_pr_author should return EXISTING_CONTRIBUTOR when user has merged PRs."""
        mock_client = AsyncMock()
        mock_client.check_existing_contributor = AsyncMock(return_value=3)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        config = GoodEggConfig(skip_known_contributors=True, scoring_model="v1")
        result = await score_pr_author(
            login="testuser",
            repo_owner="my-org",
            repo_name="my-repo",
            config=config,
            token="fake-token",
        )

        assert result.trust_level == TrustLevel.EXISTING_CONTRIBUTOR
        assert result.flags["is_existing_contributor"] is True
        assert result.flags["scoring_skipped"] is True
        assert result.scoring_metadata["context_repo_merged_pr_count"] == 3
        assert result.scoring_model == "v1"
        mock_client.get_user_contribution_data.assert_not_called()

    @pytest.mark.asyncio
    @patch("good_egg.github_client.GitHubClient")
    async def test_no_skip_when_count_is_zero(
        self, mock_client_cls: AsyncMock
    ) -> None:
        """score_pr_author should proceed to full scoring when count is 0."""
        mock_client = AsyncMock()
        mock_client.check_existing_contributor = AsyncMock(return_value=0)
        prs, repos = _sample_prs_and_repos()
        contrib_data = _make_contribution_data(merged_prs=prs, repos=repos)
        mock_client.get_user_contribution_data = AsyncMock(return_value=contrib_data)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        config = GoodEggConfig(skip_known_contributors=True, scoring_model="v1")
        result = await score_pr_author(
            login="testuser",
            repo_owner="my-org",
            repo_name="my-repo",
            config=config,
            token="fake-token",
        )

        assert result.trust_level != TrustLevel.EXISTING_CONTRIBUTOR
        mock_client.get_user_contribution_data.assert_called_once()

    @pytest.mark.asyncio
    @patch("good_egg.github_client.GitHubClient")
    async def test_force_score_bypasses_check(
        self, mock_client_cls: AsyncMock
    ) -> None:
        """skip_known_contributors=False should bypass the existing contributor check."""
        mock_client = AsyncMock()
        prs, repos = _sample_prs_and_repos()
        contrib_data = _make_contribution_data(merged_prs=prs, repos=repos)
        mock_client.get_user_contribution_data = AsyncMock(return_value=contrib_data)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        config = GoodEggConfig(skip_known_contributors=False, scoring_model="v1")
        result = await score_pr_author(
            login="testuser",
            repo_owner="my-org",
            repo_name="my-repo",
            config=config,
            token="fake-token",
        )

        assert result.trust_level != TrustLevel.EXISTING_CONTRIBUTOR
        mock_client.check_existing_contributor.assert_not_called()
        mock_client.get_user_contribution_data.assert_called_once()

    @pytest.mark.asyncio
    @patch("good_egg.github_client.GitHubClient")
    async def test_existing_contributor_has_no_fresh_account(
        self, mock_client_cls: AsyncMock
    ) -> None:
        """Existing contributor early return should have fresh_account=None."""
        mock_client = AsyncMock()
        mock_client.check_existing_contributor = AsyncMock(return_value=3)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        config = GoodEggConfig(skip_known_contributors=True, scoring_model="v1")
        result = await score_pr_author(
            login="testuser",
            repo_owner="my-org",
            repo_name="my-repo",
            config=config,
            token="fake-token",
        )

        assert result.trust_level == TrustLevel.EXISTING_CONTRIBUTOR
        assert result.fresh_account is None


class TestV3Scoring:
    def test_v3_merge_rate_as_score(self) -> None:
        config = GoodEggConfig(scoring_model="v3")
        scorer = TrustScorer(config)
        prs, repos = _sample_prs_and_repos()
        # 3 merged + 2 closed = 3/5 = 0.6
        data = _make_contribution_data(merged_prs=prs, repos=repos, closed_pr_count=2)
        result = scorer.score(data, "my-org/my-app")

        assert abs(result.raw_score - 0.6) < 1e-9
        assert abs(result.normalized_score - 0.6) < 1e-9
        assert result.scoring_model == "v3"

    def test_v3_component_scores_shape(self) -> None:
        config = GoodEggConfig(scoring_model="v3")
        scorer = TrustScorer(config)
        prs, repos = _sample_prs_and_repos()
        data = _make_contribution_data(merged_prs=prs, repos=repos, closed_pr_count=5)
        result = scorer.score(data, "my-org/my-app")

        assert list(result.component_scores.keys()) == ["merge_rate"]
        assert 0.0 <= result.component_scores["merge_rate"] <= 1.0

    def test_v3_no_graph_metadata(self) -> None:
        config = GoodEggConfig(scoring_model="v3")
        scorer = TrustScorer(config)
        prs, repos = _sample_prs_and_repos()
        data = _make_contribution_data(merged_prs=prs, repos=repos, closed_pr_count=5)
        result = scorer.score(data, "my-org/my-app")

        assert "graph_nodes" not in result.scoring_metadata
        assert "graph_edges" not in result.scoring_metadata
        assert "closed_pr_count" in result.scoring_metadata

    def test_v3_trust_level_classification(self) -> None:
        config = GoodEggConfig(scoring_model="v3")
        scorer = TrustScorer(config)
        prs, repos = _sample_prs_and_repos()
        # 3 merged + 0 closed = 100% merge rate -> HIGH
        data = _make_contribution_data(merged_prs=prs, repos=repos, closed_pr_count=0)
        result = scorer.score(data, "my-org/my-app")

        assert result.trust_level == TrustLevel.HIGH

    def test_v3_low_merge_rate(self) -> None:
        config = GoodEggConfig(scoring_model="v3")
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
        # 1 merged + 10 closed = 1/11 ~ 0.09 -> LOW
        data = _make_contribution_data(merged_prs=prs, repos=repos, closed_pr_count=10)
        result = scorer.score(data, "org/repo")

        assert result.trust_level == TrustLevel.LOW

    def test_v3_is_default_model(self) -> None:
        config = GoodEggConfig()
        assert config.scoring_model == "v3"

    def test_v3_bot_short_circuit(self) -> None:
        config = GoodEggConfig(scoring_model="v3")
        scorer = TrustScorer(config)
        data = _make_contribution_data(is_bot=True)
        result = scorer.score(data, "org/repo")

        assert result.trust_level == TrustLevel.BOT
        assert result.scoring_model == "v3"

    def test_v3_insufficient_data_short_circuit(self) -> None:
        config = GoodEggConfig(scoring_model="v3")
        scorer = TrustScorer(config)
        data = _make_contribution_data(merged_prs=[])
        result = scorer.score(data, "org/repo")

        assert result.trust_level == TrustLevel.UNKNOWN
        assert result.scoring_model == "v3"

    def test_v3_medium_merge_rate(self) -> None:
        """A 50% merge rate lands in the MEDIUM trust band."""
        config = GoodEggConfig(scoring_model="v3")
        scorer = TrustScorer(config)
        prs, repos = _sample_prs_and_repos()
        # 3 merged + 3 closed = 3/6 = 0.5
        data = _make_contribution_data(merged_prs=prs, repos=repos, closed_pr_count=3)
        result = scorer.score(data, "my-org/my-app")

        assert abs(result.normalized_score - 0.5) < 1e-9
        assert result.trust_level == TrustLevel.MEDIUM

    def test_v3_zero_total_prs(self) -> None:
        """Edge case: merged_prs present but closed_pr_count causes 0 denominator."""
        config = GoodEggConfig(scoring_model="v3")
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
        data = _make_contribution_data(merged_prs=prs, repos=repos, closed_pr_count=0)
        result = scorer.score(data, "org/repo")

        assert result.raw_score == 1.0
        assert result.normalized_score == 1.0


class TestSuspicionScoring:
    def test_suspicion_score_computed_when_enabled(self) -> None:
        scorer = TrustScorer(_make_config())
        prs, repos = _sample_prs_and_repos()
        data = _make_contribution_data(merged_prs=prs, repos=repos)
        result = scorer.score(data, "org/repo")

        assert result.suspicion_score is not None
        assert result.suspicion_score.probability >= 0.0
        assert result.suspicion_score.probability <= 1.0

    def test_suspicion_score_none_when_disabled(self) -> None:
        config = GoodEggConfig(bad_egg=BadEggConfig(enabled=False))
        scorer = TrustScorer(config)
        prs, repos = _sample_prs_and_repos()
        data = _make_contribution_data(merged_prs=prs, repos=repos)
        result = scorer.score(data, "org/repo")

        assert result.suspicion_score is None

    def test_suspicion_score_none_for_bot(self) -> None:
        scorer = TrustScorer(_make_config())
        data = _make_contribution_data(is_bot=True)
        result = scorer.score(data, "org/repo")

        assert result.suspicion_score is None

    def test_suspicion_score_none_for_no_prs(self) -> None:
        scorer = TrustScorer(_make_config())
        data = _make_contribution_data(merged_prs=[])
        result = scorer.score(data, "org/repo")

        assert result.suspicion_score is None

    def test_suspicion_score_with_v2(self) -> None:
        config = GoodEggConfig(scoring_model="v2")
        scorer = TrustScorer(config)
        prs, repos = _sample_prs_and_repos()
        data = _make_contribution_data(merged_prs=prs, repos=repos, closed_pr_count=5)
        result = scorer.score(data, "org/repo")

        assert result.suspicion_score is not None
        assert result.scoring_model == "v2"

    def test_tier_classification_high(self) -> None:
        zero_model = BadEggModelConfig(
            intercept=2.0,
            merge_rate_weight=0.0,
            total_prs_weight=0.0,
            career_span_days_weight=0.0,
            mean_title_length_weight=0.0,
            isolation_score_weight=0.0,
            total_repos_weight=0.0,
            median_additions_weight=0.0,
            median_files_changed_weight=0.0,
        )
        config = GoodEggConfig(
            bad_egg=BadEggConfig(
                model=zero_model,
                thresholds=BadEggThresholds(high=0.10, elevated=0.05),
            )
        )
        scorer = TrustScorer(config)
        prs, repos = _sample_prs_and_repos()
        data = _make_contribution_data(merged_prs=prs, repos=repos)
        result = scorer.score(data, "org/repo")

        # With intercept=2.0 and all other weights=0, sigmoid(2.0) ~ 0.88
        assert result.suspicion_score is not None
        assert result.suspicion_score.suspicion_level == SuspicionLevel.HIGH

    def test_tier_classification_elevated(self) -> None:
        zero_model = BadEggModelConfig(
            intercept=-2.9,
            merge_rate_weight=0.0,
            total_prs_weight=0.0,
            career_span_days_weight=0.0,
            mean_title_length_weight=0.0,
            isolation_score_weight=0.0,
            total_repos_weight=0.0,
            median_additions_weight=0.0,
            median_files_changed_weight=0.0,
        )
        config = GoodEggConfig(
            bad_egg=BadEggConfig(
                model=zero_model,
                thresholds=BadEggThresholds(high=0.10, elevated=0.05),
            )
        )
        scorer = TrustScorer(config)
        prs, repos = _sample_prs_and_repos()
        data = _make_contribution_data(merged_prs=prs, repos=repos)
        result = scorer.score(data, "org/repo")

        # sigmoid(-2.9) ~ 0.052, between 0.05 and 0.10
        assert result.suspicion_score is not None
        assert result.suspicion_score.suspicion_level == SuspicionLevel.ELEVATED

    def test_tier_classification_normal(self) -> None:
        zero_model = BadEggModelConfig(
            intercept=-5.0,
            merge_rate_weight=0.0,
            total_prs_weight=0.0,
            career_span_days_weight=0.0,
            mean_title_length_weight=0.0,
            isolation_score_weight=0.0,
            total_repos_weight=0.0,
            median_additions_weight=0.0,
            median_files_changed_weight=0.0,
        )
        config = GoodEggConfig(
            bad_egg=BadEggConfig(
                model=zero_model,
                thresholds=BadEggThresholds(high=0.10, elevated=0.05),
            )
        )
        scorer = TrustScorer(config)
        prs, repos = _sample_prs_and_repos()
        data = _make_contribution_data(merged_prs=prs, repos=repos)
        result = scorer.score(data, "org/repo")

        # sigmoid(-5.0) ~ 0.0067, below 0.05
        assert result.suspicion_score is not None
        assert result.suspicion_score.suspicion_level == SuspicionLevel.NORMAL

    def test_merge_rate_computation(self) -> None:
        scorer = TrustScorer(_make_config())
        prs, repos = _sample_prs_and_repos()
        data = _make_contribution_data(
            merged_prs=prs, repos=repos, closed_pr_count=3,
        )
        result = scorer.score(data, "org/repo")

        assert result.suspicion_score is not None
        expected = len(prs) / (len(prs) + 3)
        assert (
            abs(result.suspicion_score.component_scores["merge_rate"] - expected)
            < 1e-9
        )

    def test_median_additions_odd(self) -> None:
        scorer = TrustScorer(_make_config())
        prs = [
            MergedPR(
                repo_name_with_owner="org/repo",
                title="PR",
                merged_at=datetime(2024, 1, 1, tzinfo=UTC),
                additions=10,
            ),
            MergedPR(
                repo_name_with_owner="org/repo",
                title="PR",
                merged_at=datetime(2024, 2, 1, tzinfo=UTC),
                additions=30,
            ),
            MergedPR(
                repo_name_with_owner="org/repo",
                title="PR",
                merged_at=datetime(2024, 3, 1, tzinfo=UTC),
                additions=50,
            ),
        ]
        repos = {
            "org/repo": RepoMetadata(
                name_with_owner="org/repo", stargazer_count=100,
            ),
        }
        data = _make_contribution_data(merged_prs=prs, repos=repos)
        result = scorer.score(data, "org/repo")

        assert result.suspicion_score is not None
        expected = math.log1p(30.0)  # median of [10, 30, 50]
        assert (
            abs(
                result.suspicion_score.component_scores["log_median_additions"]
                - expected
            )
            < 1e-9
        )

    def test_median_additions_even(self) -> None:
        scorer = TrustScorer(_make_config())
        prs = [
            MergedPR(
                repo_name_with_owner="org/repo",
                title="PR",
                merged_at=datetime(2024, 1, 1, tzinfo=UTC),
                additions=10,
            ),
            MergedPR(
                repo_name_with_owner="org/repo",
                title="PR",
                merged_at=datetime(2024, 2, 1, tzinfo=UTC),
                additions=30,
            ),
        ]
        repos = {
            "org/repo": RepoMetadata(
                name_with_owner="org/repo", stargazer_count=100,
            ),
        }
        data = _make_contribution_data(merged_prs=prs, repos=repos)
        result = scorer.score(data, "org/repo")

        assert result.suspicion_score is not None
        expected = math.log1p(20.0)  # average of [10, 30]
        assert (
            abs(
                result.suspicion_score.component_scores["log_median_additions"]
                - expected
            )
            < 1e-9
        )


class TestFreshAccountAdvisory:
    def test_fresh_account_flagged_under_365(self) -> None:
        config = GoodEggConfig(scoring_model="v3")
        scorer = TrustScorer(config)
        prs, repos = _sample_prs_and_repos()
        data = _make_contribution_data(
            days_old=200, merged_prs=prs, repos=repos
        )
        result = scorer.score(data, "org/repo")

        assert result.fresh_account is not None
        assert result.fresh_account.is_fresh is True
        assert result.fresh_account.account_age_days == 200
        assert result.fresh_account.threshold_days == 365

    def test_fresh_account_not_flagged_over_365(self) -> None:
        config = GoodEggConfig(scoring_model="v3")
        scorer = TrustScorer(config)
        prs, repos = _sample_prs_and_repos()
        data = _make_contribution_data(
            days_old=500, merged_prs=prs, repos=repos
        )
        result = scorer.score(data, "org/repo")

        assert result.fresh_account is not None
        assert result.fresh_account.is_fresh is False

    def test_fresh_account_flagged_at_364(self) -> None:
        """364 days is strictly less than 365, so the account is fresh."""
        config = GoodEggConfig(scoring_model="v3")
        scorer = TrustScorer(config)
        prs, repos = _sample_prs_and_repos()
        data = _make_contribution_data(
            days_old=364, merged_prs=prs, repos=repos
        )
        result = scorer.score(data, "org/repo")

        assert result.fresh_account is not None
        assert result.fresh_account.is_fresh is True

    def test_fresh_account_boundary_exactly_365(self) -> None:
        config = GoodEggConfig(scoring_model="v3")
        scorer = TrustScorer(config)
        prs, repos = _sample_prs_and_repos()
        data = _make_contribution_data(
            days_old=365, merged_prs=prs, repos=repos
        )
        result = scorer.score(data, "org/repo")

        assert result.fresh_account is not None
        assert result.fresh_account.is_fresh is False

    def test_fresh_account_none_on_bot_short_circuit(self) -> None:
        config = GoodEggConfig(scoring_model="v3")
        scorer = TrustScorer(config)
        data = _make_contribution_data(is_bot=True, days_old=100)
        result = scorer.score(data, "org/repo")

        assert result.trust_level == TrustLevel.BOT
        assert result.fresh_account is None

    def test_fresh_account_on_insufficient_data(self) -> None:
        config = GoodEggConfig(scoring_model="v3")
        scorer = TrustScorer(config)
        data = _make_contribution_data(merged_prs=[], days_old=100)
        result = scorer.score(data, "org/repo")

        assert result.trust_level == TrustLevel.UNKNOWN
        assert result.fresh_account is not None
        assert result.fresh_account.is_fresh is True

    def test_fresh_account_populated_in_v1(self) -> None:
        scorer = TrustScorer(_make_config())
        prs, repos = _sample_prs_and_repos()
        data = _make_contribution_data(
            days_old=200, merged_prs=prs, repos=repos
        )
        result = scorer.score(data, "org/repo")

        assert result.fresh_account is not None
        assert result.fresh_account.is_fresh is True

    def test_fresh_account_populated_in_v2(self) -> None:
        config = GoodEggConfig(scoring_model="v2")
        scorer = TrustScorer(config)
        prs, repos = _sample_prs_and_repos()
        data = _make_contribution_data(
            days_old=200, merged_prs=prs, repos=repos
        )
        result = scorer.score(data, "org/repo")

        assert result.fresh_account is not None
        assert result.fresh_account.is_fresh is True


class TestIsolationScore:
    def test_all_isolated(self) -> None:
        """All repos have no overlapping contributors -> isolation = 1.0."""
        scorer = TrustScorer(_make_config())
        prs = [
            MergedPR(
                repo_name_with_owner="org/repo-a",
                title="PR",
                merged_at=datetime(2024, 1, 1, tzinfo=UTC),
            ),
            MergedPR(
                repo_name_with_owner="org/repo-b",
                title="PR",
                merged_at=datetime(2024, 2, 1, tzinfo=UTC),
            ),
        ]
        repos = {
            "org/repo-a": RepoMetadata(
                name_with_owner="org/repo-a", stargazer_count=100,
            ),
            "org/repo-b": RepoMetadata(
                name_with_owner="org/repo-b", stargazer_count=100,
            ),
        }
        data = _make_contribution_data(
            merged_prs=prs, repos=repos,
            repo_contributors={
                "org/repo-a": ["alice"],
                "org/repo-b": ["bob"],
            },
        )
        score = scorer._compute_isolation_score(data)
        assert score == 1.0

    def test_none_isolated(self) -> None:
        """All repos share a multi-repo contributor -> isolation = 0.0."""
        scorer = TrustScorer(_make_config())
        data = _make_contribution_data(
            merged_prs=[
                MergedPR(
                    repo_name_with_owner="org/repo-a",
                    title="PR",
                    merged_at=datetime(2024, 1, 1, tzinfo=UTC),
                ),
                MergedPR(
                    repo_name_with_owner="org/repo-b",
                    title="PR",
                    merged_at=datetime(2024, 2, 1, tzinfo=UTC),
                ),
            ],
            repos={
                "org/repo-a": RepoMetadata(
                    name_with_owner="org/repo-a", stargazer_count=100,
                ),
                "org/repo-b": RepoMetadata(
                    name_with_owner="org/repo-b", stargazer_count=100,
                ),
            },
            repo_contributors={
                "org/repo-a": ["shared-dev"],
                "org/repo-b": ["shared-dev"],
            },
        )
        score = scorer._compute_isolation_score(data)
        assert score == 0.0

    def test_no_repo_contributors(self) -> None:
        """No repo_contributors data -> isolation defaults to 1.0."""
        scorer = TrustScorer(_make_config())
        data = _make_contribution_data(
            merged_prs=[
                MergedPR(
                    repo_name_with_owner="org/repo",
                    title="PR",
                    merged_at=datetime(2024, 1, 1, tzinfo=UTC),
                ),
            ],
            repos={
                "org/repo": RepoMetadata(
                    name_with_owner="org/repo", stargazer_count=100,
                ),
            },
        )
        score = scorer._compute_isolation_score(data)
        assert score == 1.0

    def test_skipped_popular_repos_non_isolated(self) -> None:
        """Repos in contributed_repos but not in repo_contributors count as non-isolated."""
        scorer = TrustScorer(_make_config())
        data = _make_contribution_data(
            merged_prs=[
                MergedPR(
                    repo_name_with_owner="org/small",
                    title="PR",
                    merged_at=datetime(2024, 1, 1, tzinfo=UTC),
                ),
                MergedPR(
                    repo_name_with_owner="org/popular",
                    title="PR",
                    merged_at=datetime(2024, 2, 1, tzinfo=UTC),
                ),
            ],
            repos={
                "org/small": RepoMetadata(
                    name_with_owner="org/small", stargazer_count=100,
                ),
                "org/popular": RepoMetadata(
                    name_with_owner="org/popular", stargazer_count=10000,
                ),
            },
            repo_contributors={
                "org/small": ["alice"],
                # org/popular skipped (too popular)
            },
        )
        score = scorer._compute_isolation_score(data)
        # 1 isolated (small) + 1 non-isolated (popular skipped) = 0.5
        assert score == 0.5
