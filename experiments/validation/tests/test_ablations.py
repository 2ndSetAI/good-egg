from __future__ import annotations

from datetime import UTC, datetime

from experiments.validation.ablations import (
    NoQualityGraphBuilder,
    NoSelfPenaltyGraphBuilder,
    get_ablation_variants,
    make_scorer,
)
from good_egg.config import GoodEggConfig
from good_egg.models import (
    MergedPR,
    RepoMetadata,
    UserContributionData,
    UserProfile,
)


def _make_user_data() -> UserContributionData:
    """Build diverse test data that exercises multiple scoring dimensions.

    - Multiple languages (Python, Rust, Go) -> language match/norm
    - Self-owned repo -> self-penalty
    - Old PR (2022) -> recency decay
    - Varied star counts -> repo quality
    - Multiple repos -> diversity/volume
    """
    return UserContributionData(
        profile=UserProfile(
            login="testuser",
            created_at=datetime(2020, 1, 1, tzinfo=UTC),
            followers_count=50,
            public_repos_count=10,
        ),
        merged_prs=[
            MergedPR(
                repo_name_with_owner="org/target-repo",
                title="Target PR",
                merged_at=datetime(2024, 5, 1, tzinfo=UTC),
                additions=100,
            ),
            MergedPR(
                repo_name_with_owner="other/rust-lib",
                title="Rust PR",
                merged_at=datetime(2024, 4, 1, tzinfo=UTC),
                additions=50,
            ),
            MergedPR(
                repo_name_with_owner="testuser/own-repo",
                title="Self PR",
                merged_at=datetime(2024, 3, 1, tzinfo=UTC),
                additions=200,
            ),
            MergedPR(
                repo_name_with_owner="old/go-project",
                title="Old PR",
                merged_at=datetime(2022, 6, 1, tzinfo=UTC),
                additions=30,
            ),
            MergedPR(
                repo_name_with_owner="big/framework",
                title="Framework PR",
                merged_at=datetime(2024, 2, 1, tzinfo=UTC),
                additions=300,
            ),
        ],
        contributed_repos={
            "org/target-repo": RepoMetadata(
                name_with_owner="org/target-repo",
                stargazer_count=5000,
                primary_language="Python",
            ),
            "other/rust-lib": RepoMetadata(
                name_with_owner="other/rust-lib",
                stargazer_count=2000,
                primary_language="Rust",
            ),
            "testuser/own-repo": RepoMetadata(
                name_with_owner="testuser/own-repo",
                stargazer_count=10,
                primary_language="Python",
            ),
            "old/go-project": RepoMetadata(
                name_with_owner="old/go-project",
                stargazer_count=500,
                primary_language="Go",
                is_archived=True,
            ),
            "big/framework": RepoMetadata(
                name_with_owner="big/framework",
                stargazer_count=50000,
                primary_language="Python",
            ),
        },
    )


def test_get_ablation_variants_count() -> None:
    config = GoodEggConfig()
    variants = get_ablation_variants(config)
    assert len(variants) == 10


def test_ablation_configs_differ() -> None:
    config = GoodEggConfig()
    variants = get_ablation_variants(config)

    # No recency should have huge half_life_days
    nr = variants["no_recency"]
    assert nr.config.recency.half_life_days == 999999

    # No language match should set same_language_weight = other_weight
    nlm = variants["no_language_match"]
    assert (
        nlm.config.graph_scoring.same_language_weight
        == nlm.config.graph_scoring.other_weight
    )

    # No diversity/volume
    ndv = variants["no_diversity_volume"]
    assert ndv.config.graph_scoring.diversity_scale == 0.0
    assert ndv.config.graph_scoring.volume_scale == 0.0


def test_no_quality_builder() -> None:
    config = GoodEggConfig()
    builder = NoQualityGraphBuilder(config)
    meta = RepoMetadata(
        name_with_owner="org/repo", stargazer_count=50000,
    )
    assert builder._repo_quality(meta) == 1.0
    assert builder._repo_quality(None) == 1.0


def test_no_self_penalty_builder() -> None:
    assert not NoSelfPenaltyGraphBuilder._is_self_contribution(
        "alice", "alice/repo",
    )
    assert not NoSelfPenaltyGraphBuilder._is_self_contribution(
        "bob", "bob/project",
    )


def test_make_scorer_with_builder() -> None:
    config = GoodEggConfig()
    variants = get_ablation_variants(config)
    scorer = make_scorer(variants["no_repo_quality"])
    assert isinstance(
        scorer._graph_builder, NoQualityGraphBuilder,
    )


def test_ablation_scores_differ_from_full() -> None:
    """Each ablation should produce a different score than full."""
    config = GoodEggConfig()
    from good_egg.scorer import TrustScorer

    user_data = _make_user_data()
    context_repo = "org/target-repo"

    full_scorer = TrustScorer(config)
    full_score = full_scorer.score(user_data, context_repo)

    variants = get_ablation_variants(config)
    different_count = 0

    for _name, variant in variants.items():
        abl_scorer = make_scorer(variant)
        abl_score = abl_scorer.score(user_data, context_repo)
        if (
            abl_score.normalized_score
            != full_score.normalized_score
        ):
            different_count += 1

    # At least some ablations should produce different scores
    assert different_count >= 3, (
        f"Only {different_count}/10 ablations differed"
    )


def test_recursive_quality_builder() -> None:
    """RecursiveQualityGraphBuilder uses quality map instead of stars."""
    from experiments.validation.ablations import (
        RecursiveQualityGraphBuilder,
    )

    config = GoodEggConfig()

    # Without quality map, falls back to 1.0
    builder_empty = RecursiveQualityGraphBuilder(config)
    meta = RepoMetadata(
        name_with_owner="org/repo", stargazer_count=50000,
    )
    assert builder_empty._repo_quality(meta) == 1.0

    # With quality map, uses the mapped value
    qmap = {"org/repo": 2.5, "other/lib": 0.8}
    builder = RecursiveQualityGraphBuilder(config, repo_quality_map=qmap)
    assert builder._repo_quality(meta) == 2.5
    assert builder._repo_quality(None) == 1.0

    # Archived penalty applied on top
    archived_meta = RepoMetadata(
        name_with_owner="org/repo",
        stargazer_count=50000,
        is_archived=True,
    )
    assert builder._repo_quality(archived_meta) == 2.5 * 0.5

    # Fork penalty applied
    fork_meta = RepoMetadata(
        name_with_owner="org/repo",
        stargazer_count=50000,
        is_fork=True,
    )
    assert builder._repo_quality(fork_meta) == 2.5 * 0.7

    # Floor at 0.1
    low_qmap = {"org/repo": 0.01}
    builder_low = RecursiveQualityGraphBuilder(
        config, repo_quality_map=low_qmap,
    )
    assert builder_low._repo_quality(meta) == 0.1


def test_make_recursive_quality_scorer() -> None:
    """make_recursive_quality_scorer creates scorer with quality map."""
    from experiments.validation.ablations import (
        RecursiveQualityGraphBuilder,
        make_recursive_quality_scorer,
    )

    config = GoodEggConfig()
    variants = get_ablation_variants(config)
    rq_variant = variants["recursive_quality"]
    qmap = {"org/repo": 2.0}

    scorer = make_recursive_quality_scorer(rq_variant, qmap)
    assert isinstance(
        scorer._graph_builder, RecursiveQualityGraphBuilder,
    )
    assert scorer._graph_builder._repo_quality_map == qmap


def test_recursive_quality_variant_in_ablations() -> None:
    """recursive_quality is included in ablation variants."""
    config = GoodEggConfig()
    variants = get_ablation_variants(config)
    assert "recursive_quality" in variants
    from experiments.validation.ablations import (
        RecursiveQualityGraphBuilder,
    )
    assert (
        variants["recursive_quality"].builder_class
        is RecursiveQualityGraphBuilder
    )
