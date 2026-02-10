"""Tests for the MCP server module."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from good_egg.mcp_server import (
    _error_json,
    _parse_repo,
    cache_stats,
    check_pr_author,
    clear_cache,
    get_trust_details,
    main,
    score_user,
)
from good_egg.models import ContributionSummary, TrustLevel, TrustScore


def _make_trust_score() -> TrustScore:
    """Build a TrustScore fixture for MCP tests."""
    return TrustScore(
        user_login="testuser",
        context_repo="owner/repo",
        raw_score=0.005,
        normalized_score=0.72,
        trust_level=TrustLevel.HIGH,
        account_age_days=365,
        total_merged_prs=42,
        unique_repos_contributed=10,
        top_contributions=[
            ContributionSummary(
                repo_name="owner/repo", pr_count=20, language="Python", stars=100
            ),
        ],
        language_match=True,
        flags={"is_bot": False, "is_new_account": False},
        scoring_metadata={"graph_nodes": 50, "graph_edges": 120},
    )


class TestParseRepo:
    def test_valid_repo(self) -> None:
        assert _parse_repo("owner/repo") == ("owner", "repo")

    def test_no_slash(self) -> None:
        with pytest.raises(ValueError, match="owner/name"):
            _parse_repo("badformat")

    def test_empty_owner(self) -> None:
        with pytest.raises(ValueError, match="owner/name"):
            _parse_repo("/repo")

    def test_empty_name(self) -> None:
        with pytest.raises(ValueError, match="owner/name"):
            _parse_repo("owner/")

    def test_too_many_slashes(self) -> None:
        with pytest.raises(ValueError, match="owner/name"):
            _parse_repo("a/b/c")

    def test_empty_string(self) -> None:
        with pytest.raises(ValueError, match="owner/name"):
            _parse_repo("")


class TestErrorJson:
    def test_returns_json_with_error_key(self) -> None:
        result = json.loads(_error_json("something broke"))
        assert result == {"error": "something broke"}

    def test_empty_message(self) -> None:
        result = json.loads(_error_json(""))
        assert result == {"error": ""}


class TestMain:
    @patch("good_egg.mcp_server.FastMCP")
    def test_main_calls_run(self, mock_fastmcp_cls: MagicMock) -> None:
        mock_server = MagicMock()
        mock_fastmcp_cls.return_value = mock_server
        main()
        mock_fastmcp_cls.assert_called_once_with("good-egg")
        assert mock_server.tool.call_count == 5
        registered = [
            call.args[0]
            for call in mock_server.tool.return_value.call_args_list
        ]
        assert registered == [
            score_user,
            check_pr_author,
            get_trust_details,
            cache_stats,
            clear_cache,
        ]
        mock_server.run.assert_called_once_with(transport="stdio")


class TestMcpNotInstalled:
    @patch("good_egg.mcp_server.FastMCP", None)
    def test_exits_when_mcp_missing(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


class TestScoreUser:
    @pytest.mark.asyncio
    @patch("good_egg.mcp_server._get_cache")
    @patch("good_egg.mcp_server._get_config")
    @patch("good_egg.mcp_server.score_pr_author", new_callable=AsyncMock)
    async def test_success(
        self,
        mock_score: AsyncMock,
        mock_config: MagicMock,
        mock_cache: MagicMock,
    ) -> None:
        mock_config.return_value = MagicMock()
        mock_cache_inst = MagicMock()
        mock_cache.return_value = mock_cache_inst
        trust = _make_trust_score()
        mock_score.return_value = trust

        result = await score_user("testuser", "owner/repo")
        parsed = json.loads(result)

        assert parsed["user_login"] == "testuser"
        assert parsed["trust_level"] == "HIGH"
        assert parsed["normalized_score"] == 0.72
        assert parsed["total_merged_prs"] == 42
        mock_cache_inst.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_bad_repo_format(self) -> None:
        result = await score_user("testuser", "badformat")
        parsed = json.loads(result)
        assert "error" in parsed
        assert "owner/name" in parsed["error"]

    @pytest.mark.asyncio
    async def test_empty_repo_parts(self) -> None:
        result = await score_user("testuser", "/repo")
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    @patch("good_egg.mcp_server._get_cache")
    @patch("good_egg.mcp_server._get_config")
    @patch("good_egg.mcp_server.score_pr_author", new_callable=AsyncMock)
    async def test_scoring_error(
        self,
        mock_score: AsyncMock,
        mock_config: MagicMock,
        mock_cache: MagicMock,
    ) -> None:
        mock_config.return_value = MagicMock()
        mock_cache_inst = MagicMock()
        mock_cache.return_value = mock_cache_inst
        mock_score.side_effect = RuntimeError("API failure")

        result = await score_user("testuser", "owner/repo")
        parsed = json.loads(result)
        assert "error" in parsed
        assert "API failure" in parsed["error"]
        mock_cache_inst.close.assert_called_once()


class TestCheckPrAuthor:
    @pytest.mark.asyncio
    @patch("good_egg.mcp_server._get_cache")
    @patch("good_egg.mcp_server._get_config")
    @patch("good_egg.mcp_server.score_pr_author", new_callable=AsyncMock)
    async def test_success(
        self,
        mock_score: AsyncMock,
        mock_config: MagicMock,
        mock_cache: MagicMock,
    ) -> None:
        mock_config.return_value = MagicMock()
        mock_cache_inst = MagicMock()
        mock_cache.return_value = mock_cache_inst
        mock_score.return_value = _make_trust_score()

        result = await check_pr_author("testuser", "owner/repo")
        parsed = json.loads(result)

        assert parsed["user_login"] == "testuser"
        assert parsed["trust_level"] == "HIGH"
        assert parsed["normalized_score"] == 0.72
        assert parsed["total_merged_prs"] == 42
        assert "raw_score" not in parsed
        assert "top_contributions" not in parsed
        mock_cache_inst.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_bad_repo_format(self) -> None:
        result = await check_pr_author("testuser", "noslash")
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    @patch("good_egg.mcp_server._get_cache")
    @patch("good_egg.mcp_server._get_config")
    @patch("good_egg.mcp_server.score_pr_author", new_callable=AsyncMock)
    async def test_scoring_error(
        self,
        mock_score: AsyncMock,
        mock_config: MagicMock,
        mock_cache: MagicMock,
    ) -> None:
        mock_config.return_value = MagicMock()
        mock_cache_inst = MagicMock()
        mock_cache.return_value = mock_cache_inst
        mock_score.side_effect = RuntimeError("timeout")

        result = await check_pr_author("testuser", "owner/repo")
        parsed = json.loads(result)
        assert "error" in parsed
        mock_cache_inst.close.assert_called_once()


class TestGetTrustDetails:
    @pytest.mark.asyncio
    @patch("good_egg.mcp_server._get_cache")
    @patch("good_egg.mcp_server._get_config")
    @patch("good_egg.mcp_server.score_pr_author", new_callable=AsyncMock)
    async def test_success(
        self,
        mock_score: AsyncMock,
        mock_config: MagicMock,
        mock_cache: MagicMock,
    ) -> None:
        mock_config.return_value = MagicMock()
        mock_cache_inst = MagicMock()
        mock_cache.return_value = mock_cache_inst
        mock_score.return_value = _make_trust_score()

        result = await get_trust_details("testuser", "owner/repo")
        parsed = json.loads(result)

        assert parsed["user_login"] == "testuser"
        assert parsed["trust_level"] == "HIGH"
        assert parsed["normalized_score"] == 0.72
        assert parsed["raw_score"] == 0.005
        assert parsed["account_age_days"] == 365
        assert parsed["total_merged_prs"] == 42
        assert parsed["unique_repos_contributed"] == 10
        assert parsed["language_match"] is True
        assert len(parsed["top_contributions"]) == 1
        assert parsed["top_contributions"][0]["repo_name"] == "owner/repo"
        assert parsed["flags"]["is_bot"] is False
        assert parsed["scoring_metadata"]["graph_nodes"] == 50
        mock_cache_inst.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_bad_repo_format(self) -> None:
        result = await get_trust_details("testuser", "bad")
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    @patch("good_egg.mcp_server._get_cache")
    @patch("good_egg.mcp_server._get_config")
    @patch("good_egg.mcp_server.score_pr_author", new_callable=AsyncMock)
    async def test_scoring_error(
        self,
        mock_score: AsyncMock,
        mock_config: MagicMock,
        mock_cache: MagicMock,
    ) -> None:
        mock_config.return_value = MagicMock()
        mock_cache_inst = MagicMock()
        mock_cache.return_value = mock_cache_inst
        mock_score.side_effect = ValueError("not found")

        result = await get_trust_details("testuser", "owner/repo")
        parsed = json.loads(result)
        assert "error" in parsed
        mock_cache_inst.close.assert_called_once()


class TestCacheStats:
    @pytest.mark.asyncio
    @patch("good_egg.mcp_server._get_cache")
    @patch("good_egg.mcp_server._get_config")
    async def test_success(
        self,
        mock_config: MagicMock,
        mock_cache: MagicMock,
    ) -> None:
        mock_config.return_value = MagicMock()
        mock_cache_inst = MagicMock()
        mock_cache_inst.stats.return_value = {
            "total_entries": 100,
            "expired_entries": 20,
            "active_entries": 80,
            "categories": {"repo_metadata": 50},
            "db_size_bytes": 51200,
        }
        mock_cache.return_value = mock_cache_inst

        result = await cache_stats()
        parsed = json.loads(result)

        assert parsed["total_entries"] == 100
        assert parsed["active_entries"] == 80
        assert parsed["categories"]["repo_metadata"] == 50
        mock_cache_inst.close.assert_called_once()

    @pytest.mark.asyncio
    @patch("good_egg.mcp_server._get_cache")
    @patch("good_egg.mcp_server._get_config")
    async def test_error(
        self,
        mock_config: MagicMock,
        mock_cache: MagicMock,
    ) -> None:
        mock_config.return_value = MagicMock()
        mock_cache_inst = MagicMock()
        mock_cache_inst.stats.side_effect = RuntimeError("db locked")
        mock_cache.return_value = mock_cache_inst

        result = await cache_stats()
        parsed = json.loads(result)
        assert "error" in parsed
        mock_cache_inst.close.assert_called_once()


class TestClearCache:
    @pytest.mark.asyncio
    @patch("good_egg.mcp_server._get_cache")
    @patch("good_egg.mcp_server._get_config")
    async def test_clear_expired(
        self,
        mock_config: MagicMock,
        mock_cache: MagicMock,
    ) -> None:
        mock_config.return_value = MagicMock()
        mock_cache_inst = MagicMock()
        mock_cache_inst.cleanup_expired.return_value = 15
        mock_cache.return_value = mock_cache_inst

        result = await clear_cache()
        parsed = json.loads(result)

        assert parsed["expired_entries_removed"] == 15
        mock_cache_inst.cleanup_expired.assert_called_once()
        mock_cache_inst.close.assert_called_once()

    @pytest.mark.asyncio
    @patch("good_egg.mcp_server._get_cache")
    @patch("good_egg.mcp_server._get_config")
    async def test_clear_category(
        self,
        mock_config: MagicMock,
        mock_cache: MagicMock,
    ) -> None:
        mock_config.return_value = MagicMock()
        mock_cache_inst = MagicMock()
        mock_cache.return_value = mock_cache_inst

        result = await clear_cache(category="repo_metadata")
        parsed = json.loads(result)

        assert parsed["cleared_category"] == "repo_metadata"
        mock_cache_inst.invalidate_category.assert_called_once_with("repo_metadata")
        mock_cache_inst.close.assert_called_once()

    @pytest.mark.asyncio
    @patch("good_egg.mcp_server._get_cache")
    @patch("good_egg.mcp_server._get_config")
    async def test_error(
        self,
        mock_config: MagicMock,
        mock_cache: MagicMock,
    ) -> None:
        mock_config.return_value = MagicMock()
        mock_cache_inst = MagicMock()
        mock_cache_inst.cleanup_expired.side_effect = RuntimeError("db error")
        mock_cache.return_value = mock_cache_inst

        result = await clear_cache()
        parsed = json.loads(result)
        assert "error" in parsed
        mock_cache_inst.close.assert_called_once()
