"""Configuration models for Good Egg."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class GraphScoringConfig(BaseModel):
    """Graph-based scoring algorithm parameters."""
    alpha: float = 0.85
    max_iterations: int = 100
    tolerance: float = 1e-6
    context_repo_weight: float = 0.5
    same_language_weight: float = 0.3
    other_weight: float = 0.03
    diversity_scale: float = 0.5
    volume_scale: float = 0.3


class EdgeWeightConfig(BaseModel):
    """Edge weight multipliers for different contribution types."""
    merged_pr: float = 1.0
    maintainer: float = 2.0
    star: float = 0.1
    review: float = 0.5


class RecencyConfig(BaseModel):
    """Recency decay parameters."""
    half_life_days: int = 180
    max_age_days: int = 730


class ThresholdConfig(BaseModel):
    """Trust level thresholds."""
    high_trust: float = 0.7
    medium_trust: float = 0.3
    new_account_days: int = 30


class CacheTTLConfig(BaseModel):
    """Cache time-to-live settings in hours."""
    repo_metadata_hours: int = 168  # 7 days
    user_profile_hours: int = 24   # 1 day
    user_prs_hours: int = 336      # 14 days

    def to_seconds(self) -> dict[str, int]:
        """Convert TTLs to seconds for the cache layer."""
        return {
            "repo_metadata": self.repo_metadata_hours * 3600,
            "user_profile": self.user_profile_hours * 3600,
            "user_prs": self.user_prs_hours * 3600,
        }


class LanguageNormalization(BaseModel):
    """Language ecosystem size normalization multipliers.

    Smaller ecosystems get higher multipliers so that contributions
    to niche but high-quality projects are valued appropriately.
    """
    multipliers: dict[str, float] = Field(default_factory=lambda: {
        "JavaScript": 1.0,
        "Python": 1.13,
        "TypeScript": 1.30,
        "Java": 1.55,
        "C++": 1.66,
        "C": 1.67,
        "PHP": 1.97,
        "C#": 2.03,
        "Ruby": 2.17,
        "Kotlin": 2.24,
        "Go": 2.30,
        "Swift": 2.40,
        "Objective-C": 2.53,
        "Lua": 2.61,
        "Rust": 2.63,
        "Dart": 2.82,
        "Perl": 3.07,
        "R": 3.18,
        "Scala": 3.50,
        "Julia": 3.52,
        "Haskell": 3.63,
        "Elixir": 4.04,
        "Clojure": 4.28,
        "OCaml": 4.63,
        "Erlang": 5.18,
        "Zig": 5.44,
        "F#": 5.57,
        "Nim": 5.96,
    })
    default: float = 3.0

    def get_multiplier(self, language: str | None) -> float:
        """Get the normalization multiplier for a language."""
        if language is None:
            return self.default
        return self.multipliers.get(language, self.default)


class FetchConfig(BaseModel):
    """GitHub API fetch parameters."""
    max_prs: int = 500
    max_repos_to_enrich: int = 200
    rate_limit_safety_margin: int = 100


class GoodEggConfig(BaseModel):
    """Top-level configuration composing all sub-configs."""
    graph_scoring: GraphScoringConfig = Field(default_factory=GraphScoringConfig)
    edge_weights: EdgeWeightConfig = Field(default_factory=EdgeWeightConfig)
    recency: RecencyConfig = Field(default_factory=RecencyConfig)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    cache_ttl: CacheTTLConfig = Field(default_factory=CacheTTLConfig)
    language_normalization: LanguageNormalization = Field(
        default_factory=LanguageNormalization
    )
    fetch: FetchConfig = Field(default_factory=FetchConfig)


def load_config(path: str | Path | None = None) -> GoodEggConfig:
    """Load configuration from YAML file, environment variables, and defaults.

    Priority (highest to lowest):
    1. Environment variables (GOOD_EGG_*)
    2. YAML config file
    3. Defaults
    """
    config_data: dict[str, Any] = {}

    # Load from YAML file
    if path is not None:
        config_path = Path(path)
        if config_path.exists():
            with open(config_path) as f:
                yaml_data = yaml.safe_load(f)
                if yaml_data:
                    config_data = yaml_data
    else:
        # Try default locations
        for default_path in [".good-egg.yml", ".good-egg.yaml"]:
            p = Path(default_path)
            if p.exists():
                with open(p) as f:
                    yaml_data = yaml.safe_load(f)
                    if yaml_data:
                        config_data = yaml_data
                break

    # Apply environment variable overrides
    env_mapping = {
        "GOOD_EGG_ALPHA": ("graph_scoring", "alpha", float),
        "GOOD_EGG_MAX_PRS": ("fetch", "max_prs", int),
        "GOOD_EGG_HIGH_TRUST": ("thresholds", "high_trust", float),
        "GOOD_EGG_MEDIUM_TRUST": ("thresholds", "medium_trust", float),
        "GOOD_EGG_HALF_LIFE_DAYS": ("recency", "half_life_days", int),
        "GOOD_EGG_OTHER_WEIGHT": ("graph_scoring", "other_weight", float),
        "GOOD_EGG_DIVERSITY_SCALE": ("graph_scoring", "diversity_scale", float),
        "GOOD_EGG_VOLUME_SCALE": ("graph_scoring", "volume_scale", float),
    }

    for env_var, (section, key, type_fn) in env_mapping.items():
        value = os.environ.get(env_var)
        if value is not None:
            if section not in config_data:
                config_data[section] = {}
            config_data[section][key] = type_fn(value)

    return GoodEggConfig(**config_data)
