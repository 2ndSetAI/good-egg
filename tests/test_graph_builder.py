"""Tests for the trust graph builder."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from good_egg.config import GoodEggConfig
from good_egg.graph_builder import TrustGraphBuilder
from good_egg.models import (
    MergedPR,
    RepoMetadata,
    UserContributionData,
    UserProfile,
)


def _make_config(**overrides: object) -> GoodEggConfig:
    return GoodEggConfig(**overrides)  # type: ignore[arg-type]


def _make_user_data(
    login: str = "testuser",
    merged_prs: list[MergedPR] | None = None,
    repos: dict[str, RepoMetadata] | None = None,
) -> UserContributionData:
    return UserContributionData(
        profile=UserProfile(
            login=login,
            created_at=datetime(2020, 1, 1, tzinfo=UTC),
            followers_count=10,
            public_repos_count=5,
        ),
        merged_prs=merged_prs or [],
        contributed_repos=repos or {},
    )


class TestBuildGraph:
    def test_empty_prs(self) -> None:
        builder = TrustGraphBuilder(_make_config())
        data = _make_user_data()
        graph = builder.build_graph(data, "owner/repo")
        assert len(graph.nodes) == 1
        assert "user:testuser" in graph.nodes
        assert len(graph.edges) == 0

    def test_single_pr_creates_bipartite_edges(self) -> None:
        builder = TrustGraphBuilder(_make_config())
        pr = MergedPR(
            repo_name_with_owner="elixir-lang/elixir",
            title="Fix bug",
            merged_at=datetime.now(UTC) - timedelta(days=30),
        )
        repo = RepoMetadata(
            name_with_owner="elixir-lang/elixir",
            stargazer_count=23000,
            primary_language="Elixir",
        )
        data = _make_user_data(
            merged_prs=[pr],
            repos={"elixir-lang/elixir": repo},
        )
        graph = builder.build_graph(data, "my-org/app")

        assert "user:testuser" in graph.nodes
        assert "repo:elixir-lang/elixir" in graph.nodes
        assert graph.has_edge("user:testuser", "repo:elixir-lang/elixir")
        assert graph.has_edge("repo:elixir-lang/elixir", "user:testuser")

    def test_reverse_edge_is_fraction_of_forward(self) -> None:
        builder = TrustGraphBuilder(_make_config())
        pr = MergedPR(
            repo_name_with_owner="org/repo",
            title="PR",
            merged_at=datetime.now(UTC) - timedelta(days=10),
        )
        repo = RepoMetadata(
            name_with_owner="org/repo",
            stargazer_count=1000,
        )
        data = _make_user_data(
            merged_prs=[pr],
            repos={"org/repo": repo},
        )
        graph = builder.build_graph(data, "ctx/repo")

        forward = graph["user:testuser"]["repo:org/repo"]["weight"]
        reverse = graph["repo:org/repo"]["user:testuser"]["weight"]
        assert abs(reverse - forward * 0.3) < 1e-9

    def test_node_attributes(self) -> None:
        builder = TrustGraphBuilder(_make_config())
        pr = MergedPR(
            repo_name_with_owner="org/repo",
            title="PR",
            merged_at=datetime.now(UTC),
        )
        repo = RepoMetadata(
            name_with_owner="org/repo",
            stargazer_count=500,
            primary_language="Python",
        )
        data = _make_user_data(
            merged_prs=[pr],
            repos={"org/repo": repo},
        )
        graph = builder.build_graph(data, "ctx/repo")

        assert graph.nodes["user:testuser"]["kind"] == "user"
        assert graph.nodes["repo:org/repo"]["kind"] == "repo"
        assert graph.nodes["repo:org/repo"]["language"] == "Python"
        assert graph.nodes["repo:org/repo"]["stars"] == 500


class TestEdgeWeights:
    def test_recency_decay(self) -> None:
        builder = TrustGraphBuilder(_make_config())
        # At half-life days, decay should be ~0.5
        half_life = builder.config.recency.half_life_days
        decay = builder._recency_decay(half_life)
        assert abs(decay - 0.5) < 0.01

    def test_recency_decay_zero_days(self) -> None:
        builder = TrustGraphBuilder(_make_config())
        decay = builder._recency_decay(0)
        assert abs(decay - 1.0) < 1e-9

    def test_recency_decay_beyond_max(self) -> None:
        builder = TrustGraphBuilder(_make_config())
        decay = builder._recency_decay(builder.config.recency.max_age_days + 1)
        assert decay == 0.0

    def test_repo_quality_with_stars(self) -> None:
        builder = TrustGraphBuilder(_make_config())
        meta = RepoMetadata(
            name_with_owner="org/repo",
            stargazer_count=10000,
            primary_language="Python",
        )
        quality = builder._repo_quality(meta)
        # log1p(10000 * 1.5) = log1p(15000) > 0
        assert quality > 0
        expected = math.log1p(10000 * 1.5)
        assert abs(quality - expected) < 1e-9

    def test_repo_quality_archived_penalty(self) -> None:
        builder = TrustGraphBuilder(_make_config())
        meta_active = RepoMetadata(
            name_with_owner="org/repo",
            stargazer_count=1000,
            primary_language="Python",
        )
        meta_archived = RepoMetadata(
            name_with_owner="org/repo",
            stargazer_count=1000,
            primary_language="Python",
            is_archived=True,
        )
        q_active = builder._repo_quality(meta_active)
        q_archived = builder._repo_quality(meta_archived)
        assert abs(q_archived - q_active * 0.5) < 1e-9

    def test_repo_quality_fork_penalty(self) -> None:
        builder = TrustGraphBuilder(_make_config())
        meta_orig = RepoMetadata(
            name_with_owner="org/repo",
            stargazer_count=1000,
            primary_language="Python",
        )
        meta_fork = RepoMetadata(
            name_with_owner="org/repo",
            stargazer_count=1000,
            primary_language="Python",
            is_fork=True,
        )
        q_orig = builder._repo_quality(meta_orig)
        q_fork = builder._repo_quality(meta_fork)
        assert abs(q_fork - q_orig * 0.3) < 1e-9

    def test_repo_quality_none_metadata(self) -> None:
        builder = TrustGraphBuilder(_make_config())
        assert builder._repo_quality(None) == 1.0

    def test_language_multiplier(self) -> None:
        builder = TrustGraphBuilder(_make_config())
        assert builder._get_language_multiplier("Elixir") == 8.0
        assert builder._get_language_multiplier("Python") == 1.5
        assert builder._get_language_multiplier("JavaScript") == 1.0
        # Unknown language gets default
        assert builder._get_language_multiplier("Fortran") == 2.0
        assert builder._get_language_multiplier(None) == 2.0


class TestSelfContribution:
    def test_self_contribution_discount(self) -> None:
        builder = TrustGraphBuilder(_make_config())
        pr_own = MergedPR(
            repo_name_with_owner="testuser/my-project",
            title="Update readme",
            merged_at=datetime.now(UTC) - timedelta(days=5),
        )
        pr_external = MergedPR(
            repo_name_with_owner="other-org/project",
            title="Fix bug",
            merged_at=datetime.now(UTC) - timedelta(days=5),
        )
        repos = {
            "testuser/my-project": RepoMetadata(
                name_with_owner="testuser/my-project",
                stargazer_count=100,
                primary_language="Python",
            ),
            "other-org/project": RepoMetadata(
                name_with_owner="other-org/project",
                stargazer_count=100,
                primary_language="Python",
            ),
        }
        data = _make_user_data(
            login="testuser",
            merged_prs=[pr_own, pr_external],
            repos=repos,
        )
        graph = builder.build_graph(data, "ctx/repo")

        own_weight = graph["user:testuser"]["repo:testuser/my-project"]["weight"]
        ext_weight = graph["user:testuser"]["repo:other-org/project"]["weight"]
        assert abs(own_weight - ext_weight * 0.3) < 1e-9

    def test_is_self_contribution(self) -> None:
        assert TrustGraphBuilder._is_self_contribution("alice", "alice/repo")
        assert TrustGraphBuilder._is_self_contribution("Alice", "alice/repo")
        assert not TrustGraphBuilder._is_self_contribution("alice", "bob/repo")
        assert not TrustGraphBuilder._is_self_contribution("alice", "org/repo")


class TestPRCap:
    def test_caps_prs_per_repo(self) -> None:
        builder = TrustGraphBuilder(_make_config())
        # Create 25 PRs to the same repo (cap is 20)
        prs = [
            MergedPR(
                repo_name_with_owner="org/repo",
                title=f"PR {i}",
                merged_at=datetime.now(UTC) - timedelta(days=i),
            )
            for i in range(25)
        ]
        repo = RepoMetadata(
            name_with_owner="org/repo",
            stargazer_count=1000,
            primary_language="Python",
        )
        data = _make_user_data(merged_prs=prs, repos={"org/repo": repo})

        # Build with cap=20 (should use only 20 most recent)
        graph = builder.build_graph(data, "ctx/repo")

        # Now compute expected weight with all 25 to confirm they differ
        builder_no_cap = TrustGraphBuilder(_make_config())
        builder_no_cap.MAX_PRS_PER_REPO = 100
        graph_no_cap = builder_no_cap.build_graph(data, "ctx/repo")

        capped_weight = graph["user:testuser"]["repo:org/repo"]["weight"]
        uncapped_weight = graph_no_cap["user:testuser"]["repo:org/repo"]["weight"]

        # The capped graph should have less total weight (fewer PRs)
        assert capped_weight < uncapped_weight


class TestPersonalizationVector:
    def test_context_repo_gets_highest_weight(self) -> None:
        builder = TrustGraphBuilder(_make_config())
        pr1 = MergedPR(
            repo_name_with_owner="org/context-app",
            title="PR",
            merged_at=datetime.now(UTC),
        )
        pr2 = MergedPR(
            repo_name_with_owner="org/other",
            title="PR",
            merged_at=datetime.now(UTC),
        )
        repos = {
            "org/context-app": RepoMetadata(
                name_with_owner="org/context-app",
                stargazer_count=100,
                primary_language="Elixir",
            ),
            "org/other": RepoMetadata(
                name_with_owner="org/other",
                stargazer_count=100,
                primary_language="Python",
            ),
        }
        data = _make_user_data(merged_prs=[pr1, pr2], repos=repos)
        graph = builder.build_graph(data, "org/context-app")

        pv = builder.build_personalization_vector(
            graph, "org/context-app", "Elixir"
        )

        # Context repo should have the highest weight
        ctx_weight = pv["repo:org/context-app"]
        other_weight = pv["repo:org/other"]
        user_weight = pv["user:testuser"]

        assert ctx_weight > other_weight
        assert user_weight == 0.0

    def test_same_language_gets_medium_weight(self) -> None:
        builder = TrustGraphBuilder(_make_config())
        pr1 = MergedPR(
            repo_name_with_owner="org/elixir-lib",
            title="PR",
            merged_at=datetime.now(UTC),
        )
        pr2 = MergedPR(
            repo_name_with_owner="org/python-lib",
            title="PR",
            merged_at=datetime.now(UTC),
        )
        repos = {
            "org/elixir-lib": RepoMetadata(
                name_with_owner="org/elixir-lib",
                stargazer_count=100,
                primary_language="Elixir",
            ),
            "org/python-lib": RepoMetadata(
                name_with_owner="org/python-lib",
                stargazer_count=100,
                primary_language="Python",
            ),
        }
        data = _make_user_data(merged_prs=[pr1, pr2], repos=repos)
        graph = builder.build_graph(data, "ctx/repo")

        pv = builder.build_personalization_vector(graph, "ctx/repo", "Elixir")

        elixir_weight = pv["repo:org/elixir-lib"]
        python_weight = pv["repo:org/python-lib"]
        assert elixir_weight > python_weight

    def test_personalization_sums_to_one(self) -> None:
        builder = TrustGraphBuilder(_make_config())
        pr = MergedPR(
            repo_name_with_owner="org/repo",
            title="PR",
            merged_at=datetime.now(UTC),
        )
        repos = {
            "org/repo": RepoMetadata(
                name_with_owner="org/repo",
                stargazer_count=100,
                primary_language="Python",
            ),
        }
        data = _make_user_data(merged_prs=[pr], repos=repos)
        graph = builder.build_graph(data, "org/repo")

        pv = builder.build_personalization_vector(graph, "org/repo", "Python")
        total = sum(pv.values())
        assert abs(total - 1.0) < 1e-9

    def test_user_nodes_get_zero_weight(self) -> None:
        builder = TrustGraphBuilder(_make_config())
        pr = MergedPR(
            repo_name_with_owner="org/repo",
            title="PR",
            merged_at=datetime.now(UTC),
        )
        repos = {
            "org/repo": RepoMetadata(
                name_with_owner="org/repo",
                stargazer_count=100,
            ),
        }
        data = _make_user_data(merged_prs=[pr], repos=repos)
        graph = builder.build_graph(data, "org/repo")

        pv = builder.build_personalization_vector(graph, "org/repo", None)
        for node, weight in pv.items():
            if graph.nodes[node].get("kind") == "user":
                assert weight == 0.0
