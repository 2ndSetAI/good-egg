"""Tests for GitHub Action entry point."""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, patch

import pytest

from good_egg.action import _set_output, run_action
from good_egg.models import TrustLevel, TrustScore


@pytest.fixture
def pr_event_file(tmp_path):
    """Create a temporary PR event file."""
    event = {
        "action": "opened",
        "number": 42,
        "pull_request": {
            "number": 42,
            "user": {"login": "testuser"},
            "head": {"sha": "abc123def456"},
        },
        "repository": {"full_name": "my-org/my-repo"},
    }
    event_file = tmp_path / "event.json"
    event_file.write_text(json.dumps(event))
    return str(event_file)


@pytest.fixture
def mock_env(pr_event_file, tmp_path):
    """Set up environment variables for action."""
    output_file = tmp_path / "output.txt"
    output_file.touch()
    return {
        "GITHUB_TOKEN": "ghp_testtoken123",
        "GITHUB_EVENT_PATH": pr_event_file,
        "GITHUB_REPOSITORY": "my-org/my-repo",
        "GITHUB_OUTPUT": str(output_file),
        "INPUT_COMMENT": "true",
        "INPUT_CHECK-RUN": "false",
        "INPUT_FAIL-ON-LOW": "false",
    }


def _make_mock_score(trust_level=TrustLevel.HIGH, normalized_score=0.72):
    return TrustScore(
        user_login="testuser",
        context_repo="my-org/my-repo",
        raw_score=0.005,
        normalized_score=normalized_score,
        trust_level=trust_level,
        account_age_days=365,
        total_merged_prs=42,
        unique_repos_contributed=10,
        flags={},
    )


class TestRunAction:
    @pytest.mark.asyncio
    async def test_success_with_comment(self, mock_env):
        mock_score = _make_mock_score()
        mock_client = AsyncMock()
        mock_client.find_existing_comment = AsyncMock(return_value=None)
        mock_client.post_pr_comment = AsyncMock(return_value={})
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.dict(os.environ, mock_env, clear=False), \
             patch("good_egg.action.GitHubClient", return_value=mock_client), \
             patch("good_egg.action.score_pr_author", new_callable=AsyncMock,
                   return_value=mock_score):
            await run_action()
            mock_client.post_pr_comment.assert_called_once()

    @pytest.mark.asyncio
    async def test_updates_existing_comment(self, mock_env):
        mock_score = _make_mock_score()
        mock_client = AsyncMock()
        mock_client.find_existing_comment = AsyncMock(return_value=99)
        mock_client.update_pr_comment = AsyncMock(return_value={})
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.dict(os.environ, mock_env, clear=False), \
             patch("good_egg.action.GitHubClient", return_value=mock_client), \
             patch("good_egg.action.score_pr_author", new_callable=AsyncMock,
                   return_value=mock_score):
            await run_action()
            mock_client.update_pr_comment.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_token_exits(self, mock_env):
        mock_env_no_token = {**mock_env, "GITHUB_TOKEN": ""}
        with patch.dict(os.environ, mock_env_no_token, clear=False), \
             pytest.raises(SystemExit) as exc_info:
            await run_action()
        assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_fail_on_low_trust(self, mock_env):
        mock_env_fail = {**mock_env, "INPUT_FAIL-ON-LOW": "true"}
        mock_score = _make_mock_score(trust_level=TrustLevel.LOW, normalized_score=0.1)
        mock_client = AsyncMock()
        mock_client.find_existing_comment = AsyncMock(return_value=None)
        mock_client.post_pr_comment = AsyncMock(return_value={})
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.dict(os.environ, mock_env_fail, clear=False), \
             patch("good_egg.action.GitHubClient", return_value=mock_client), \
             patch("good_egg.action.score_pr_author", new_callable=AsyncMock,
                   return_value=mock_score), \
             pytest.raises(SystemExit) as exc_info:
            await run_action()
        assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_sets_outputs(self, mock_env, tmp_path):
        output_file = tmp_path / "output.txt"
        output_file.touch()
        mock_env["GITHUB_OUTPUT"] = str(output_file)

        mock_score = _make_mock_score()
        mock_client = AsyncMock()
        mock_client.find_existing_comment = AsyncMock(return_value=None)
        mock_client.post_pr_comment = AsyncMock(return_value={})
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.dict(os.environ, mock_env, clear=False), \
             patch("good_egg.action.GitHubClient", return_value=mock_client), \
             patch("good_egg.action.score_pr_author", new_callable=AsyncMock,
                   return_value=mock_score):
            await run_action()

        output_content = output_file.read_text()
        assert "score=0.72" in output_content
        assert "trust-level=HIGH" in output_content
        assert "user=testuser" in output_content
        assert "skipped=false" in output_content

    @pytest.mark.asyncio
    async def test_scoring_model_input(self, mock_env, tmp_path):
        """INPUT_SCORING_MODEL should be read and applied."""
        output_file = tmp_path / "output.txt"
        output_file.touch()
        mock_env_v2 = {
            **mock_env,
            "INPUT_SCORING_MODEL": "v2",
            "GITHUB_OUTPUT": str(output_file),
        }
        mock_score = TrustScore(
            **{**_make_mock_score().model_dump(), "scoring_model": "v2"}
        )
        mock_client = AsyncMock()
        mock_client.find_existing_comment = AsyncMock(return_value=None)
        mock_client.post_pr_comment = AsyncMock(return_value={})
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.dict(os.environ, mock_env_v2, clear=False), \
             patch("good_egg.action.GitHubClient", return_value=mock_client), \
             patch("good_egg.action.score_pr_author", new_callable=AsyncMock,
                   return_value=mock_score):
            await run_action()

        output_content = output_file.read_text()
        assert "scoring-model=v2" in output_content


class TestSetOutput:
    def test_writes_to_file(self, tmp_path):
        output_file = tmp_path / "output.txt"
        output_file.touch()
        with patch.dict(os.environ, {"GITHUB_OUTPUT": str(output_file)}):
            _set_output("test-key", "test-value")
        assert "test-key=test-value" in output_file.read_text()

    def test_no_output_file(self):
        with patch.dict(os.environ, {}, clear=True):
            # Should not raise
            _set_output("test-key", "test-value")


class TestMalformedInput:
    """Tests for malformed action inputs."""

    @pytest.mark.asyncio
    async def test_malformed_event_json(self, tmp_path):
        """Malformed event JSON should exit gracefully."""
        event_file = tmp_path / "bad_event.json"
        event_file.write_text("not valid json{{{")

        env = {
            "GITHUB_TOKEN": "ghp_testtoken123",
            "GITHUB_EVENT_PATH": str(event_file),
            "GITHUB_REPOSITORY": "my-org/my-repo",
            "INPUT_COMMENT": "true",
            "INPUT_CHECK-RUN": "false",
            "INPUT_FAIL-ON-LOW": "false",
        }

        with patch.dict(os.environ, env, clear=False), \
             pytest.raises((SystemExit, json.JSONDecodeError)):
            await run_action()

    @pytest.mark.asyncio
    async def test_missing_pull_request_data(self, tmp_path):
        """Missing pull_request data should exit with code 1."""
        event_file = tmp_path / "no_pr_event.json"
        event_file.write_text(json.dumps({"action": "opened"}))

        env = {
            "GITHUB_TOKEN": "ghp_testtoken123",
            "GITHUB_EVENT_PATH": str(event_file),
            "GITHUB_REPOSITORY": "my-org/my-repo",
            "INPUT_COMMENT": "true",
            "INPUT_CHECK-RUN": "false",
            "INPUT_FAIL-ON-LOW": "false",
        }

        with patch.dict(os.environ, env, clear=False), \
             pytest.raises(SystemExit) as exc_info:
            await run_action()
        assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_invalid_repository_format(self, tmp_path):
        """Invalid GITHUB_REPOSITORY format should exit with code 1."""
        event_file = tmp_path / "event.json"
        event_file.write_text(json.dumps({
            "pull_request": {
                "number": 1,
                "user": {"login": "testuser"},
                "head": {"sha": "abc123"},
            }
        }))

        env = {
            "GITHUB_TOKEN": "ghp_testtoken123",
            "GITHUB_EVENT_PATH": str(event_file),
            "GITHUB_REPOSITORY": "invalid-no-slash",
            "INPUT_COMMENT": "true",
            "INPUT_CHECK-RUN": "false",
            "INPUT_FAIL-ON-LOW": "false",
        }

        with patch.dict(os.environ, env, clear=False), \
             pytest.raises(SystemExit) as exc_info:
            await run_action()
        assert exc_info.value.code == 1


class TestExistingContributorAction:
    @pytest.mark.asyncio
    async def test_skips_scoring_for_existing_contributor(self, mock_env, tmp_path):
        """Action should report skipped when score_pr_author returns EXISTING_CONTRIBUTOR."""
        output_file = tmp_path / "output.txt"
        output_file.touch()
        mock_env["GITHUB_OUTPUT"] = str(output_file)

        mock_score = TrustScore(
            user_login="testuser",
            context_repo="my-org/my-repo",
            trust_level=TrustLevel.EXISTING_CONTRIBUTOR,
            flags={"is_existing_contributor": True, "scoring_skipped": True},
            scoring_metadata={"context_repo_merged_pr_count": 7},
        )
        mock_client = AsyncMock()
        mock_client.find_existing_comment = AsyncMock(return_value=None)
        mock_client.post_pr_comment = AsyncMock(return_value={})
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.dict(os.environ, mock_env, clear=False), \
             patch("good_egg.action.GitHubClient", return_value=mock_client), \
             patch("good_egg.action.score_pr_author", new_callable=AsyncMock,
                   return_value=mock_score):
            await run_action()

        output_content = output_file.read_text()
        assert "trust-level=EXISTING_CONTRIBUTOR" in output_content
        assert "skipped=true" in output_content

    @pytest.mark.asyncio
    async def test_respects_skip_known_contributors_false(self, mock_env, tmp_path):
        """When INPUT_SKIP_KNOWN_CONTRIBUTORS=false, config should propagate."""
        output_file = tmp_path / "output.txt"
        output_file.touch()
        mock_env_override = {
            **mock_env,
            "INPUT_SKIP_KNOWN_CONTRIBUTORS": "false",
            "GITHUB_OUTPUT": str(output_file),
        }

        mock_score = _make_mock_score()
        mock_client = AsyncMock()
        mock_client.find_existing_comment = AsyncMock(return_value=None)
        mock_client.post_pr_comment = AsyncMock(return_value={})
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.dict(os.environ, mock_env_override, clear=False), \
             patch("good_egg.action.GitHubClient", return_value=mock_client), \
             patch("good_egg.action.score_pr_author", new_callable=AsyncMock,
                   return_value=mock_score) as mock_score_fn:
            await run_action()

        # Verify config passed has skip disabled
        call_kwargs = mock_score_fn.call_args
        config_passed = call_kwargs.kwargs.get("config")
        assert config_passed.skip_known_contributors is False

        output_content = output_file.read_text()
        assert "skipped=false" in output_content
