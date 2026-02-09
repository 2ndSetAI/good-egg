"""Shared test fixtures for Good Egg tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from good_egg.models import (
    ContributionSummary,
    MergedPR,
    RepoMetadata,
    TrustLevel,
    TrustScore,
    UserContributionData,
    UserProfile,
)


@pytest.fixture
def sample_user_profile() -> UserProfile:
    return UserProfile(
        login="testuser",
        created_at=datetime(2020, 1, 1, tzinfo=UTC),
        followers_count=50,
        public_repos_count=20,
        is_bot=False,
    )


@pytest.fixture
def sample_bot_profile() -> UserProfile:
    return UserProfile(
        login="dependabot[bot]",
        created_at=datetime(2019, 1, 1, tzinfo=UTC),
        followers_count=0,
        public_repos_count=0,
        is_bot=True,
    )


@pytest.fixture
def sample_new_account_profile() -> UserProfile:
    return UserProfile(
        login="newuser",
        created_at=datetime.now(UTC) - timedelta(days=10),
        followers_count=2,
        public_repos_count=1,
        is_bot=False,
    )


@pytest.fixture
def sample_repo_metadata() -> RepoMetadata:
    return RepoMetadata(
        name_with_owner="elixir-lang/elixir",
        stargazer_count=23000,
        fork_count=3200,
        primary_language="Elixir",
        is_archived=False,
        is_fork=False,
    )


@pytest.fixture
def sample_merged_pr() -> MergedPR:
    return MergedPR(
        repo_name_with_owner="elixir-lang/elixir",
        title="Fix pattern matching edge case",
        merged_at=datetime(2024, 6, 15, tzinfo=UTC),
        additions=45,
        deletions=12,
        changed_files=3,
    )


@pytest.fixture
def sample_merged_prs() -> list[MergedPR]:
    return [
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


@pytest.fixture
def sample_repos_metadata() -> dict[str, RepoMetadata]:
    return {
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


@pytest.fixture
def sample_user_contribution_data(
    sample_user_profile: UserProfile,
    sample_merged_prs: list[MergedPR],
    sample_repos_metadata: dict[str, RepoMetadata],
) -> UserContributionData:
    return UserContributionData(
        profile=sample_user_profile,
        merged_prs=sample_merged_prs,
        contributed_repos=sample_repos_metadata,
    )


@pytest.fixture
def sample_trust_score() -> TrustScore:
    return TrustScore(
        user_login="testuser",
        context_repo="my-org/my-elixir-app",
        raw_score=0.0045,
        normalized_score=0.72,
        trust_level=TrustLevel.HIGH,
        percentile=85.0,
        account_age_days=1800,
        total_merged_prs=3,
        unique_repos_contributed=3,
        top_contributions=[
            ContributionSummary(
                repo_name="elixir-lang/elixir",
                pr_count=1,
                language="Elixir",
                stars=23000,
            ),
            ContributionSummary(
                repo_name="phoenixframework/phoenix",
                pr_count=1,
                language="Elixir",
                stars=20000,
            ),
        ],
        language_match=True,
        flags={
            "is_bot": False,
            "is_new_account": False,
            "has_insufficient_data": False,
            "used_cached_data": False,
        },
        scoring_metadata={"graph_nodes": 7, "graph_edges": 6},
    )
