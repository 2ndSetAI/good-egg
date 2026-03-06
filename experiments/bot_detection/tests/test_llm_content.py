from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from experiments.bot_detection.llm_client import llm_classify
from experiments.bot_detection.stages.stage6_llm_content import _parse_llm_score


class TestParseLlmScore:
    def test_valid_json(self) -> None:
        resp = json.dumps({"score": 0.8, "reasoning": "looks spammy"})
        assert _parse_llm_score(resp) == 0.8

    def test_json_in_code_fence(self) -> None:
        resp = '```json\n{"score": 0.3, "reasoning": "ok"}\n```'
        assert _parse_llm_score(resp) == 0.3

    def test_clamps_above_one(self) -> None:
        resp = json.dumps({"score": 1.5, "reasoning": "x"})
        assert _parse_llm_score(resp) == 1.0

    def test_clamps_below_zero(self) -> None:
        resp = json.dumps({"score": -0.2, "reasoning": "x"})
        assert _parse_llm_score(resp) == 0.0

    def test_invalid_json_returns_default(self) -> None:
        assert _parse_llm_score("not json at all") == 0.5

    def test_missing_score_key_returns_default(self) -> None:
        resp = json.dumps({"reasoning": "no score here"})
        assert _parse_llm_score(resp) == 0.5


class TestLlmClassifyCache:
    def test_caches_result(self, tmp_path: Path) -> None:
        with patch(
            "experiments.bot_detection.llm_client._litellm_call",
            return_value='{"score": 0.7}',
        ) as mock_call:
            # First call hits litellm
            result1 = llm_classify("test-model", "test prompt", cache_dir=tmp_path)
            assert result1 == '{"score": 0.7}'
            assert mock_call.call_count == 1

            # Second call with same prompt hits cache
            result2 = llm_classify("test-model", "test prompt", cache_dir=tmp_path)
            assert result2 == '{"score": 0.7}'
            assert mock_call.call_count == 1  # No additional call

    def test_different_prompts_not_cached(self, tmp_path: Path) -> None:
        with patch(
            "experiments.bot_detection.llm_client._litellm_call",
            return_value='{"score": 0.5}',
        ) as mock_call:
            llm_classify("test-model", "prompt A", cache_dir=tmp_path)
            llm_classify("test-model", "prompt B", cache_dir=tmp_path)
            assert mock_call.call_count == 2


class TestStage6Integration:
    def test_run_stage6(self, tmp_path: Path) -> None:
        from experiments.bot_detection.models import StudyConfig

        # Create a minimal author_features.parquet
        features_dir = tmp_path / "data" / "features"
        features_dir.mkdir(parents=True)
        df = pd.DataFrame({
            "login": ["spam-user", "good-user", "inactive-user"],
            "merge_rate": [0.1, 0.9, 0.3],
            "total_repos": [5, 10, 1],  # inactive-user has < 2 repos
        })
        df.to_parquet(features_dir / "author_features.parquet", index=False)

        config = StudyConfig(
            paths={"features": "data/features", "local_db": "data/test.duckdb"},
            author_analysis={
                "llm_model": "test-model",
                "llm_max_titles": 10,
                "llm_pre_filter_merge_rate": 0.5,
                "llm_pre_filter_min_repos": 2,
            },
        )

        llm_response = json.dumps({"score": 0.9, "reasoning": "spam"})

        with (
            patch(
                "experiments.bot_detection.llm_client._litellm_call",
                return_value=llm_response,
            ),
            patch(
                "experiments.bot_detection.stages.stage6_llm_content.BotDetectionDB",
            ) as mock_db_cls,
        ):
            mock_db = MagicMock()
            mock_db.get_author_pr_titles.return_value = ["Fix typo", "Update README"]
            mock_db_cls.return_value = mock_db

            from experiments.bot_detection.stages.stage6_llm_content import (
                run_stage6_llm_content,
            )

            run_stage6_llm_content(tmp_path, config)

        result = pd.read_parquet(features_dir / "author_features.parquet")
        assert "llm_spam_score" in result.columns

        # spam-user: merge_rate=0.1 < 0.5, total_repos=5 >= 2 -> filtered -> LLM
        spam_score = result.loc[result["login"] == "spam-user", "llm_spam_score"].iloc[0]
        assert spam_score == pytest.approx(0.9)

        # good-user: merge_rate=0.9 >= 0.5 -> not filtered -> 0.0
        good_score = result.loc[result["login"] == "good-user", "llm_spam_score"].iloc[0]
        assert good_score == pytest.approx(0.0)

        # inactive-user: total_repos=1 < 2 -> not filtered -> 0.0
        inactive_score = result.loc[
            result["login"] == "inactive-user", "llm_spam_score"
        ].iloc[0]
        assert inactive_score == pytest.approx(0.0)
