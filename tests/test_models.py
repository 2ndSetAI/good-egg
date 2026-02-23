"""Tests for data models."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from good_egg.models import (
    MergedPR,
    RepoMetadata,
    TrustLevel,
    TrustScore,
    UserContributionData,
    UserProfile,
)


class TestUserProfile:
    def test_creation(self, sample_user_profile: UserProfile) -> None:
        assert sample_user_profile.login == "testuser"
        assert sample_user_profile.followers_count == 50
        assert not sample_user_profile.is_bot

    def test_account_age_days(self) -> None:
        profile = UserProfile(
            login="test",
            created_at=datetime.now(UTC) - timedelta(days=100),
        )
        assert 99 <= profile.account_age_days <= 101

    def test_bot_profile(self, sample_bot_profile: UserProfile) -> None:
        assert sample_bot_profile.is_bot
        assert sample_bot_profile.login == "dependabot[bot]"

    def test_new_account(self, sample_new_account_profile: UserProfile) -> None:
        assert sample_new_account_profile.account_age_days <= 15

    def test_suspected_bot_field(self) -> None:
        profile = UserProfile(
            login="ghost",
            created_at=datetime.now(UTC) - timedelta(days=500),
            is_suspected_bot=True,
        )
        assert profile.is_suspected_bot is True

    def test_suspected_bot_default_false(self) -> None:
        profile = UserProfile(
            login="normal",
            created_at=datetime.now(UTC) - timedelta(days=100),
        )
        assert profile.is_suspected_bot is False


class TestRepoMetadata:
    def test_creation(self, sample_repo_metadata: RepoMetadata) -> None:
        assert sample_repo_metadata.name_with_owner == "elixir-lang/elixir"
        assert sample_repo_metadata.stargazer_count == 23000
        assert sample_repo_metadata.primary_language == "Elixir"

    def test_defaults(self) -> None:
        repo = RepoMetadata(name_with_owner="owner/repo")
        assert repo.stargazer_count == 0
        assert repo.primary_language is None
        assert not repo.is_archived
        assert not repo.is_fork


class TestMergedPR:
    def test_creation(self, sample_merged_pr: MergedPR) -> None:
        assert sample_merged_pr.repo_name_with_owner == "elixir-lang/elixir"
        assert sample_merged_pr.additions == 45

    def test_days_ago(self) -> None:
        pr = MergedPR(
            repo_name_with_owner="owner/repo",
            title="Test PR",
            merged_at=datetime.now(UTC) - timedelta(days=30),
        )
        assert 29 <= pr.days_ago <= 31


class TestUserContributionData:
    def test_creation(self, sample_user_contribution_data: UserContributionData) -> None:
        assert sample_user_contribution_data.profile.login == "testuser"
        assert len(sample_user_contribution_data.merged_prs) == 3
        assert len(sample_user_contribution_data.contributed_repos) == 3

    def test_empty_defaults(self, sample_user_profile: UserProfile) -> None:
        data = UserContributionData(profile=sample_user_profile)
        assert data.merged_prs == []
        assert data.contributed_repos == {}

    def test_closed_pr_count_default(self, sample_user_profile: UserProfile) -> None:
        data = UserContributionData(profile=sample_user_profile)
        assert data.closed_pr_count == 0

    def test_closed_pr_count_set(self, sample_user_profile: UserProfile) -> None:
        data = UserContributionData(
            profile=sample_user_profile, closed_pr_count=15
        )
        assert data.closed_pr_count == 15


class TestTrustScore:
    def test_creation(self, sample_trust_score: TrustScore) -> None:
        assert sample_trust_score.user_login == "testuser"
        assert sample_trust_score.trust_level == TrustLevel.HIGH
        assert sample_trust_score.normalized_score == 0.72

    def test_trust_levels(self) -> None:
        assert TrustLevel.HIGH.value == "HIGH"
        assert TrustLevel.BOT.value == "BOT"

    def test_defaults(self) -> None:
        score = TrustScore(user_login="u", context_repo="o/r")
        assert score.raw_score == 0.0
        assert score.trust_level == TrustLevel.UNKNOWN
        assert score.flags == {}

    def test_scoring_model_default(self) -> None:
        score = TrustScore(user_login="u", context_repo="o/r")
        assert score.scoring_model == "v1"
        assert score.component_scores == {}

    def test_scoring_model_v2(self) -> None:
        score = TrustScore(
            user_login="u",
            context_repo="o/r",
            scoring_model="v2",
            component_scores={"graph_score": 0.65, "merge_rate": 0.8},
        )
        assert score.scoring_model == "v2"
        assert score.component_scores["graph_score"] == 0.65

    def test_serialization_roundtrip_v2(self) -> None:
        import json
        score = TrustScore(
            user_login="u",
            context_repo="o/r",
            scoring_model="v2",
            component_scores={"graph_score": 0.65},
        )
        data = json.loads(score.model_dump_json())
        restored = TrustScore(**data)
        assert restored.scoring_model == "v2"
        assert restored.component_scores == {"graph_score": 0.65}

    def test_backward_compat_no_new_fields(self) -> None:
        # Old-style TrustScore without new fields still works
        score = TrustScore(
            user_login="u",
            context_repo="o/r",
            raw_score=0.5,
            normalized_score=0.7,
            trust_level=TrustLevel.HIGH,
        )
        assert score.scoring_model == "v1"
        assert score.component_scores == {}

    def test_top_contributions(self, sample_trust_score: TrustScore) -> None:
        assert len(sample_trust_score.top_contributions) == 2
        assert sample_trust_score.top_contributions[0].repo_name == "elixir-lang/elixir"
