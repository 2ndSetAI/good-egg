"""Tests for configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from good_egg.config import (
    CacheTTLConfig,
    GoodEggConfig,
    GraphScoringConfig,
    LanguageNormalization,
    load_config,
)


class TestGraphScoringConfig:
    def test_defaults(self) -> None:
        config = GraphScoringConfig()
        assert config.alpha == 0.85
        assert config.max_iterations == 100
        assert config.tolerance == 1e-6
        assert config.context_repo_weight == 0.5


class TestLanguageNormalization:
    def test_known_language(self) -> None:
        ln = LanguageNormalization()
        assert ln.get_multiplier("Elixir") == 4.04
        assert ln.get_multiplier("Rust") == 2.63

    def test_unknown_language(self) -> None:
        ln = LanguageNormalization()
        assert ln.get_multiplier("Fortran") == 3.0

    def test_none_language(self) -> None:
        ln = LanguageNormalization()
        assert ln.get_multiplier(None) == 3.0


class TestCacheTTLConfig:
    def test_to_seconds(self) -> None:
        ttl = CacheTTLConfig()
        secs = ttl.to_seconds()
        assert secs["repo_metadata"] == 7 * 24 * 3600
        assert secs["user_profile"] == 24 * 3600
        assert secs["user_prs"] == 14 * 24 * 3600


class TestGoodEggConfig:
    def test_defaults(self) -> None:
        config = GoodEggConfig()
        assert config.graph_scoring.alpha == 0.85
        assert config.thresholds.high_trust == 0.7
        assert config.fetch.max_prs == 500


class TestLoadConfig:
    def test_load_defaults(self) -> None:
        config = load_config()
        assert isinstance(config, GoodEggConfig)
        assert config.graph_scoring.alpha == 0.85

    def test_load_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / ".good-egg.yml"
        config_file.write_text(yaml.dump({
            "graph_scoring": {"alpha": 0.9},
            "thresholds": {"high_trust": 0.8},
        }))
        config = load_config(config_file)
        assert config.graph_scoring.alpha == 0.9
        assert config.thresholds.high_trust == 0.8
        # Defaults preserved
        assert config.graph_scoring.max_iterations == 100

    def test_load_nonexistent_path(self, tmp_path: Path) -> None:
        config = load_config(tmp_path / "nonexistent.yml")
        assert config.graph_scoring.alpha == 0.85

    def test_load_directory_path_falls_back_to_defaults(self, tmp_path: Path) -> None:
        config = load_config(tmp_path)
        assert isinstance(config, GoodEggConfig)
        assert config.graph_scoring.alpha == 0.85

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOD_EGG_ALPHA", "0.9")
        config = load_config()
        assert config.graph_scoring.alpha == 0.9

    def test_env_var_max_prs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOD_EGG_MAX_PRS", "200")
        config = load_config()
        assert config.fetch.max_prs == 200

    def test_yaml_plus_env_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_file = tmp_path / ".good-egg.yml"
        config_file.write_text(yaml.dump({
            "graph_scoring": {"alpha": 0.9},
        }))
        monkeypatch.setenv("GOOD_EGG_ALPHA", "0.75")
        config = load_config(config_file)
        # Env var takes precedence
        assert config.graph_scoring.alpha == 0.75


class TestDiversityConfig:
    def test_default_values(self) -> None:
        config = GraphScoringConfig()
        assert config.other_weight == 0.03
        assert config.diversity_scale == 0.5
        assert config.volume_scale == 0.3

    def test_env_var_other_weight(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOD_EGG_OTHER_WEIGHT", "0.05")
        config = load_config()
        assert config.graph_scoring.other_weight == 0.05

    def test_env_var_diversity_scale(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOD_EGG_DIVERSITY_SCALE", "0.8")
        config = load_config()
        assert config.graph_scoring.diversity_scale == 0.8

    def test_env_var_volume_scale(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOD_EGG_VOLUME_SCALE", "0.5")
        config = load_config()
        assert config.graph_scoring.volume_scale == 0.5


class TestV2Config:
    def test_defaults(self) -> None:
        from good_egg.config import V2Config
        config = V2Config()
        assert config.graph.half_life_days == 180
        assert config.graph.max_age_days == 730
        assert config.graph.archived_penalty == 0.5
        assert config.graph.fork_penalty == 0.3
        assert config.features.temporal_merge_rate is True
        assert config.features.account_age is True
        assert isinstance(config.combined_model.intercept, float)
        assert isinstance(config.combined_model.graph_score_weight, float)

    def test_custom_values(self) -> None:
        from good_egg.config import V2Config
        config = V2Config(
            graph={"half_life_days": 90, "archived_penalty": 0.2},
            features={"temporal_merge_rate": False},
        )
        assert config.graph.half_life_days == 90
        assert config.graph.archived_penalty == 0.2
        assert config.features.temporal_merge_rate is False
        # Defaults preserved
        assert config.graph.max_age_days == 730


class TestScoringModelConfig:
    def test_default_is_v1(self) -> None:
        config = GoodEggConfig()
        assert config.scoring_model == "v1"

    def test_set_to_v2(self) -> None:
        config = GoodEggConfig(scoring_model="v2")
        assert config.scoring_model == "v2"

    def test_v2_config_present_by_default(self) -> None:
        config = GoodEggConfig()
        assert config.v2 is not None
        assert config.v2.graph.half_life_days == 180

    def test_yaml_loading_with_v2(self, tmp_path: Path) -> None:
        config_file = tmp_path / ".good-egg.yml"
        config_file.write_text(yaml.dump({
            "scoring_model": "v2",
            "v2": {
                "graph": {"half_life_days": 90},
                "features": {"account_age": False},
            },
        }))
        config = load_config(config_file)
        assert config.scoring_model == "v2"
        assert config.v2.graph.half_life_days == 90
        assert config.v2.features.account_age is False

    def test_env_var_scoring_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOD_EGG_SCORING_MODEL", "v2")
        config = load_config()
        assert config.scoring_model == "v2"

    def test_input_scoring_model_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INPUT_SCORING_MODEL", "v2")
        config = load_config()
        assert config.scoring_model == "v2"

    def test_v2_config_ignored_when_v1(self) -> None:
        config = GoodEggConfig(scoring_model="v1")
        # v2 config is still present but not used by v1 scorer
        assert config.v2 is not None
        assert config.scoring_model == "v1"
