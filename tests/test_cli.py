"""Tests for the CLI module."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

from good_egg.cli import main
from good_egg.models import TrustLevel, TrustScore


def _make_trust_score() -> TrustScore:
    """Build a TrustScore fixture for CLI tests."""
    return TrustScore(
        user_login="testuser",
        context_repo="owner/repo",
        raw_score=0.005,
        normalized_score=0.72,
        trust_level=TrustLevel.HIGH,
        account_age_days=365,
        total_merged_prs=42,
        unique_repos_contributed=10,
        top_contributions=[],
        language_match=True,
        flags={},
        scoring_metadata={},
    )


class TestScoreCommand:
    def test_score_command_no_token(self) -> None:
        """Invoking score without a token should print an error and exit non-zero."""
        runner = CliRunner(env={"GITHUB_TOKEN": ""})
        result = runner.invoke(main, ["score", "testuser", "--repo", "a/b"])
        assert result.exit_code != 0
        output = result.output + (result.stderr or "")
        assert "GitHub token required" in output

    def test_score_command_bad_repo_format(self) -> None:
        """Invoking score with a malformed --repo should print an error and exit non-zero."""
        runner = CliRunner(env={"GITHUB_TOKEN": "ghp_fake123"})
        result = runner.invoke(main, ["score", "testuser", "--repo", "badformat"])
        assert result.exit_code != 0
        assert "owner/name" in result.output or "owner/name" in (result.stderr or "")

    @patch("good_egg.cli.score_pr_author", new_callable=AsyncMock)
    @patch("good_egg.cli.load_config")
    def test_score_command_success(
        self, mock_load_config: MagicMock, mock_score: AsyncMock
    ) -> None:
        """A successful score invocation should display the trust level."""
        mock_load_config.return_value = MagicMock()
        trust_score = _make_trust_score()
        mock_score.return_value = trust_score

        runner = CliRunner(env={"GITHUB_TOKEN": "ghp_fake123"})
        result = runner.invoke(main, ["score", "testuser", "--repo", "owner/repo"])
        assert result.exit_code == 0
        assert "HIGH" in result.output
        assert "testuser" in result.output

    @patch("good_egg.cli.score_pr_author", new_callable=AsyncMock)
    @patch("good_egg.cli.load_config")
    def test_score_command_json_output(
        self, mock_load_config: MagicMock, mock_score: AsyncMock
    ) -> None:
        """The --json flag should produce valid JSON output."""
        mock_load_config.return_value = MagicMock()
        trust_score = _make_trust_score()
        mock_score.return_value = trust_score

        runner = CliRunner(env={"GITHUB_TOKEN": "ghp_fake123"})
        result = runner.invoke(
            main, ["score", "testuser", "--repo", "owner/repo", "--json"]
        )
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["user_login"] == "testuser"
        assert parsed["trust_level"] == "HIGH"
        assert parsed["normalized_score"] == 0.72

    @patch("good_egg.cli.score_pr_author", new_callable=AsyncMock)
    @patch("good_egg.cli.load_config")
    def test_score_command_verbose(
        self, mock_load_config: MagicMock, mock_score: AsyncMock
    ) -> None:
        """The -v flag should include detailed output like account age."""
        mock_load_config.return_value = MagicMock()
        trust_score = _make_trust_score()
        mock_score.return_value = trust_score

        runner = CliRunner(env={"GITHUB_TOKEN": "ghp_fake123"})
        result = runner.invoke(
            main, ["score", "testuser", "--repo", "owner/repo", "-v"]
        )
        assert result.exit_code == 0
        assert "Account age" in result.output
        assert "365" in result.output
        assert "Merged PRs" in result.output

    @patch("good_egg.cli.score_pr_author", new_callable=AsyncMock)
    @patch("good_egg.cli.load_config")
    def test_scoring_model_v2_option(
        self, mock_load_config: MagicMock, mock_score: AsyncMock
    ) -> None:
        """The --scoring-model v2 flag should be accepted."""
        mock_config = MagicMock()
        mock_load_config.return_value = mock_config
        trust_score = _make_trust_score()
        mock_score.return_value = trust_score

        runner = CliRunner(env={"GITHUB_TOKEN": "ghp_fake123"})
        result = runner.invoke(
            main,
            ["score", "testuser", "--repo", "owner/repo", "--scoring-model", "v2"],
        )
        assert result.exit_code == 0


class TestCacheCommands:
    @patch("good_egg.cli.Cache")
    def test_cache_stats(self, mock_cache_cls: MagicMock) -> None:
        """cache-stats should display cache statistics."""
        mock_cache = MagicMock()
        mock_cache.stats.return_value = {
            "total_entries": 100,
            "active_entries": 80,
            "expired_entries": 20,
            "db_size_bytes": 51200,
            "categories": {"repo_metadata": 50, "user_profile": 30},
        }
        mock_cache_cls.return_value = mock_cache

        runner = CliRunner()
        result = runner.invoke(main, ["cache-stats"])
        assert result.exit_code == 0
        assert "Total entries: 100" in result.output
        assert "Active entries: 80" in result.output
        assert "Expired entries: 20" in result.output
        assert "51,200 bytes" in result.output
        assert "repo_metadata: 50" in result.output
        assert "user_profile: 30" in result.output
        mock_cache.close.assert_called_once()

    @patch("good_egg.cli.Cache")
    def test_cache_clear(self, mock_cache_cls: MagicMock) -> None:
        """cache-clear without --category should clean up expired entries."""
        mock_cache = MagicMock()
        mock_cache.cleanup_expired.return_value = 15
        mock_cache_cls.return_value = mock_cache

        runner = CliRunner()
        result = runner.invoke(main, ["cache-clear"])
        assert result.exit_code == 0
        assert "Removed 15 expired entries" in result.output
        mock_cache.cleanup_expired.assert_called_once()
        mock_cache.close.assert_called_once()

    @patch("good_egg.cli.Cache")
    def test_cache_clear_category(self, mock_cache_cls: MagicMock) -> None:
        """cache-clear --category should invalidate only that category."""
        mock_cache = MagicMock()
        mock_cache_cls.return_value = mock_cache

        runner = CliRunner()
        result = runner.invoke(main, ["cache-clear", "--category", "repo_metadata"])
        assert result.exit_code == 0
        assert "Cleared cache category: repo_metadata" in result.output
        mock_cache.invalidate_category.assert_called_once_with("repo_metadata")
        mock_cache.close.assert_called_once()
