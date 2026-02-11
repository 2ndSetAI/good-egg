from __future__ import annotations

from dataclasses import dataclass

from good_egg.config import (
    GoodEggConfig,
    GraphScoringConfig,
    LanguageNormalization,
    RecencyConfig,
)
from good_egg.graph_builder import TrustGraphBuilder
from good_egg.models import RepoMetadata
from good_egg.scorer import TrustScorer


class NoQualityGraphBuilder(TrustGraphBuilder):
    """Graph builder that disables repo quality scoring."""

    def _repo_quality(self, meta: RepoMetadata | None) -> float:
        return 1.0


class RecursiveQualityGraphBuilder(TrustGraphBuilder):
    """Graph builder using recursive repo quality instead of stars.

    Repo quality = mean GE normalized score of authors who contributed to it.
    """

    def __init__(
        self, config: GoodEggConfig,
        repo_quality_map: dict[str, float] | None = None,
    ) -> None:
        super().__init__(config)
        self._repo_quality_map = repo_quality_map or {}

    def _repo_quality(self, meta: RepoMetadata | None) -> float:
        if meta is None:
            return 1.0
        repo_name = meta.name_with_owner
        quality = self._repo_quality_map.get(repo_name, 1.0)
        # Apply same archived/fork penalties
        if meta.is_archived:
            quality *= 0.5
        if meta.is_fork:
            quality *= 0.7
        return max(quality, 0.1)  # floor to avoid zeroing out


class NoSelfPenaltyGraphBuilder(TrustGraphBuilder):
    """Graph builder that disables self-contribution penalty."""

    @staticmethod
    def _is_self_contribution(login: str, repo_name: str) -> bool:
        return False


class NoQualityNoSelfPenaltyGraphBuilder(TrustGraphBuilder):
    """Combined: no quality + no self penalty. NOT IN PLAN - don't create this."""
    pass


class NoDiversityNoSelfPenaltyBuilder(TrustGraphBuilder):
    """Combined: no self penalty (diversity handled via config)."""

    @staticmethod
    def _is_self_contribution(login: str, repo_name: str) -> bool:
        return False


@dataclass
class AblationVariant:
    """An ablation variant with its config and optional custom builder."""
    name: str
    config: GoodEggConfig
    builder_class: type[TrustGraphBuilder] | None = None


def _deep_copy_config(config: GoodEggConfig) -> GoodEggConfig:
    """Deep copy a GoodEggConfig."""
    return GoodEggConfig.model_validate(config.model_dump())


def get_ablation_variants(
    base_config: GoodEggConfig,
) -> dict[str, AblationVariant]:
    """Generate all 10 ablation variant configurations."""
    variants: dict[str, AblationVariant] = {}

    # 1. No recency
    cfg = _deep_copy_config(base_config)
    cfg.recency = RecencyConfig(half_life_days=999999, max_age_days=999999)
    variants["no_recency"] = AblationVariant(name="no_recency", config=cfg)

    # 2. No repo quality (subclass override)
    cfg = _deep_copy_config(base_config)
    variants["no_repo_quality"] = AblationVariant(
        name="no_repo_quality", config=cfg, builder_class=NoQualityGraphBuilder,
    )

    # 3. No self-penalty (subclass override)
    cfg = _deep_copy_config(base_config)
    variants["no_self_penalty"] = AblationVariant(
        name="no_self_penalty", config=cfg,
        builder_class=NoSelfPenaltyGraphBuilder,
    )

    # 4. No language match
    cfg = _deep_copy_config(base_config)
    cfg.graph_scoring = GraphScoringConfig(
        **{**cfg.graph_scoring.model_dump(),
           "same_language_weight": cfg.graph_scoring.other_weight}
    )
    variants["no_language_match"] = AblationVariant(
        name="no_language_match", config=cfg,
    )

    # 5. No diversity/volume
    cfg = _deep_copy_config(base_config)
    cfg.graph_scoring = GraphScoringConfig(
        **{**cfg.graph_scoring.model_dump(),
           "diversity_scale": 0.0, "volume_scale": 0.0}
    )
    variants["no_diversity_volume"] = AblationVariant(
        name="no_diversity_volume", config=cfg,
    )

    # 6. No language normalization
    cfg = _deep_copy_config(base_config)
    all_ones = {k: 1.0 for k in cfg.language_normalization.multipliers}
    cfg.language_normalization = LanguageNormalization(
        multipliers=all_ones, default=1.0,
    )
    variants["no_language_norm"] = AblationVariant(
        name="no_language_norm", config=cfg,
    )

    # 7. No recency + No quality (combined)
    cfg = _deep_copy_config(base_config)
    cfg.recency = RecencyConfig(half_life_days=999999, max_age_days=999999)
    variants["no_recency_no_quality"] = AblationVariant(
        name="no_recency_no_quality", config=cfg,
        builder_class=NoQualityGraphBuilder,
    )

    # 8. No lang match + No lang norm (combined)
    cfg = _deep_copy_config(base_config)
    cfg.graph_scoring = GraphScoringConfig(
        **{**cfg.graph_scoring.model_dump(),
           "same_language_weight": cfg.graph_scoring.other_weight}
    )
    all_ones = {k: 1.0 for k in cfg.language_normalization.multipliers}
    cfg.language_normalization = LanguageNormalization(
        multipliers=all_ones, default=1.0,
    )
    variants["no_lang_match_no_lang_norm"] = AblationVariant(
        name="no_lang_match_no_lang_norm", config=cfg,
    )

    # 9. No diversity + No self-penalty (combined)
    cfg = _deep_copy_config(base_config)
    cfg.graph_scoring = GraphScoringConfig(
        **{**cfg.graph_scoring.model_dump(),
           "diversity_scale": 0.0, "volume_scale": 0.0}
    )
    variants["no_diversity_no_self_penalty"] = AblationVariant(
        name="no_diversity_no_self_penalty", config=cfg,
        builder_class=NoDiversityNoSelfPenaltyBuilder,
    )

    # 10. Recursive quality (quality based on author scores, not stars)
    cfg = _deep_copy_config(base_config)
    variants["recursive_quality"] = AblationVariant(
        name="recursive_quality", config=cfg,
        builder_class=RecursiveQualityGraphBuilder,
    )

    return variants


def make_scorer(variant: AblationVariant) -> TrustScorer:
    """Create a TrustScorer for an ablation variant.

    If the variant has a custom builder_class, replaces the scorer's
    internal graph builder after construction.
    """
    scorer = TrustScorer(variant.config)
    if variant.builder_class is not None:
        scorer._graph_builder = variant.builder_class(variant.config)
    return scorer


def make_recursive_quality_scorer(
    variant: AblationVariant,
    repo_quality_map: dict[str, float],
) -> TrustScorer:
    """Create a scorer with recursive repo quality."""
    scorer = TrustScorer(variant.config)
    scorer._graph_builder = RecursiveQualityGraphBuilder(
        variant.config, repo_quality_map=repo_quality_map,
    )
    return scorer
