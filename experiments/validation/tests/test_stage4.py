from __future__ import annotations

from datetime import UTC, datetime

from experiments.validation.stages.stage4_score import (
    _apply_anti_lookahead,
)
from good_egg.models import (
    MergedPR,
    RepoMetadata,
    UserContributionData,
    UserProfile,
)


def _make_user_data() -> UserContributionData:
    return UserContributionData(
        profile=UserProfile(
            login="alice",
            created_at=datetime(2020, 1, 1, tzinfo=UTC),
        ),
        merged_prs=[
            MergedPR(
                repo_name_with_owner="org/repo-a",
                title="PR 1",
                merged_at=datetime(2024, 3, 1, tzinfo=UTC),
            ),
            MergedPR(
                repo_name_with_owner="org/repo-b",
                title="PR 2",
                merged_at=datetime(2024, 6, 15, tzinfo=UTC),
            ),
            MergedPR(
                repo_name_with_owner="org/repo-a",
                title="PR 3",
                merged_at=datetime(2024, 9, 1, tzinfo=UTC),
            ),
        ],
        contributed_repos={
            "org/repo-a": RepoMetadata(
                name_with_owner="org/repo-a",
                stargazer_count=1000,
                primary_language="Python",
            ),
            "org/repo-b": RepoMetadata(
                name_with_owner="org/repo-b",
                stargazer_count=500,
                primary_language="Rust",
            ),
        },
    )


def test_anti_lookahead_filters_future_prs() -> None:
    user_data = _make_user_data()
    cutoff = datetime(2024, 7, 1, tzinfo=UTC)

    filtered = _apply_anti_lookahead(user_data, cutoff)

    # Should only include PRs merged before July 1
    assert len(filtered.merged_prs) == 2
    assert all(
        pr.merged_at < cutoff for pr in filtered.merged_prs
    )


def test_anti_lookahead_filters_repos() -> None:
    user_data = _make_user_data()
    cutoff = datetime(2024, 4, 1, tzinfo=UTC)

    filtered = _apply_anti_lookahead(user_data, cutoff)

    # Only repo-a has PRs before April 1
    assert len(filtered.merged_prs) == 1
    assert "org/repo-a" in filtered.contributed_repos
    assert "org/repo-b" not in filtered.contributed_repos


def test_anti_lookahead_no_prs_before_cutoff() -> None:
    user_data = _make_user_data()
    cutoff = datetime(2023, 1, 1, tzinfo=UTC)

    filtered = _apply_anti_lookahead(user_data, cutoff)
    assert len(filtered.merged_prs) == 0
    assert len(filtered.contributed_repos) == 0


def test_anti_lookahead_all_prs_before_cutoff() -> None:
    user_data = _make_user_data()
    cutoff = datetime(2025, 1, 1, tzinfo=UTC)

    filtered = _apply_anti_lookahead(user_data, cutoff)
    assert len(filtered.merged_prs) == 3


def test_anti_lookahead_preserves_profile() -> None:
    user_data = _make_user_data()
    cutoff = datetime(2024, 7, 1, tzinfo=UTC)

    filtered = _apply_anti_lookahead(user_data, cutoff)
    assert filtered.profile.login == "alice"
