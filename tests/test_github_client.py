"""Tests for the async GitHub API client."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import respx

from good_egg.cache import Cache
from good_egg.config import GoodEggConfig
from good_egg.exceptions import GitHubAPIError, RateLimitExhaustedError, UserNotFoundError
from good_egg.formatter import COMMENT_MARKER
from good_egg.github_client import GitHubClient
from good_egg.models import MergedPR, RepoMetadata, UserProfile

FIXTURES_DIR = Path(__file__).parent / "fixtures"
GRAPHQL_URL = "https://api.github.com/graphql"
BASE_URL = "https://api.github.com"


def _load_fixture(name: str) -> dict:
    with open(FIXTURES_DIR / name) as f:
        return json.load(f)


def _make_client(
    config: GoodEggConfig | None = None, cache: Cache | None = None
) -> GitHubClient:
    return GitHubClient(token="test-token", config=config or GoodEggConfig(), cache=cache)


# ---------------------------------------------------------------------------
# fetch_user_profile
# ---------------------------------------------------------------------------


class TestFetchUserProfile:
    @respx.mock
    async def test_fetch_user_profile(self) -> None:
        respx.post(GRAPHQL_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": {
                        "user": {
                            "login": "testuser",
                            "createdAt": "2020-01-01T00:00:00Z",
                            "__typename": "User",
                            "followers": {"totalCount": 50},
                            "repositories": {"totalCount": 20},
                        }
                    }
                },
            )
        )

        async with _make_client() as client:
            profile = await client.fetch_user_profile("testuser")

        assert isinstance(profile, UserProfile)
        assert profile.login == "testuser"
        assert profile.followers_count == 50
        assert profile.public_repos_count == 20
        assert profile.is_bot is False

    @respx.mock
    async def test_fetch_user_profile_bot_detection(self) -> None:
        respx.post(GRAPHQL_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": {
                        "user": {
                            "login": "dependabot[bot]",
                            "createdAt": "2019-01-01T00:00:00Z",
                            "__typename": "Bot",
                            "followers": {"totalCount": 0},
                            "repositories": {"totalCount": 0},
                        }
                    }
                },
            )
        )

        async with _make_client() as client:
            profile = await client.fetch_user_profile("dependabot[bot]")

        assert profile.is_bot is True
        assert profile.login == "dependabot[bot]"

    @respx.mock
    async def test_fetch_user_not_found(self) -> None:
        respx.post(GRAPHQL_URL).mock(
            return_value=httpx.Response(
                200,
                json={"data": {"user": None}},
            )
        )

        async with _make_client() as client:
            with pytest.raises(UserNotFoundError) as exc_info:
                await client.fetch_user_profile("ghost-user-xyz")

        assert "ghost-user-xyz" in str(exc_info.value)


# ---------------------------------------------------------------------------
# fetch_user_merged_prs
# ---------------------------------------------------------------------------


class TestFetchUserMergedPRs:
    @respx.mock
    async def test_fetch_user_merged_prs(self) -> None:
        fixture = _load_fixture("user_prs.json")
        respx.post(GRAPHQL_URL).mock(
            return_value=httpx.Response(200, json=fixture)
        )

        async with _make_client() as client:
            prs = await client.fetch_user_merged_prs("testuser")

        assert len(prs) == 2
        assert all(isinstance(pr, MergedPR) for pr in prs)
        assert prs[0].title == "Fix pattern matching edge case"
        assert prs[0].repo_name_with_owner == "elixir-lang/elixir"
        assert prs[1].title == "Add WebSocket compression support"
        assert prs[1].additions == 200

    @respx.mock
    async def test_fetch_user_merged_prs_skips_null_repository(self) -> None:
        """PRs whose repository is null (deleted/inaccessible) should be skipped."""
        fixture = {
            "data": {
                "user": {
                    "login": "testuser",
                    "createdAt": "2020-01-01T00:00:00Z",
                    "__typename": "User",
                    "followers": {"totalCount": 50},
                    "repositories": {"totalCount": 20},
                    "pullRequests": {
                        "totalCount": 3,
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                        "nodes": [
                            {
                                "title": "PR to deleted repo",
                                "mergedAt": "2024-06-15T00:00:00Z",
                                "additions": 10,
                                "deletions": 5,
                                "changedFiles": 1,
                                "repository": None,
                            },
                            {
                                "title": "PR to accessible repo",
                                "mergedAt": "2024-03-10T00:00:00Z",
                                "additions": 20,
                                "deletions": 3,
                                "changedFiles": 2,
                                "repository": {
                                    "nameWithOwner": "elixir-lang/elixir",
                                    "stargazerCount": 23000,
                                    "forkCount": 3200,
                                    "primaryLanguage": {"name": "Elixir"},
                                    "isArchived": False,
                                    "isFork": False,
                                },
                            },
                        ],
                    },
                }
            }
        }
        respx.post(GRAPHQL_URL).mock(
            return_value=httpx.Response(200, json=fixture)
        )

        async with _make_client() as client:
            prs = await client.fetch_user_merged_prs("testuser")

        assert len(prs) == 1
        assert prs[0].title == "PR to accessible repo"

    @respx.mock
    async def test_fetch_user_merged_prs_pagination(self) -> None:
        """Two pages of results should be fetched and concatenated."""
        page1 = {
            "data": {
                "user": {
                    "login": "testuser",
                    "createdAt": "2020-01-01T00:00:00Z",
                    "__typename": "User",
                    "followers": {"totalCount": 50},
                    "repositories": {"totalCount": 20},
                    "pullRequests": {
                        "totalCount": 3,
                        "pageInfo": {
                            "hasNextPage": True,
                            "endCursor": "cursor-page-1",
                        },
                        "nodes": [
                            {
                                "title": "PR on page 1",
                                "mergedAt": "2024-06-15T00:00:00Z",
                                "additions": 10,
                                "deletions": 5,
                                "changedFiles": 1,
                                "repository": {
                                    "nameWithOwner": "elixir-lang/elixir",
                                    "stargazerCount": 23000,
                                    "forkCount": 3200,
                                    "primaryLanguage": {"name": "Elixir"},
                                    "isArchived": False,
                                    "isFork": False,
                                },
                            }
                        ],
                    },
                }
            }
        }
        page2 = {
            "data": {
                "user": {
                    "login": "testuser",
                    "createdAt": "2020-01-01T00:00:00Z",
                    "__typename": "User",
                    "followers": {"totalCount": 50},
                    "repositories": {"totalCount": 20},
                    "pullRequests": {
                        "totalCount": 3,
                        "pageInfo": {
                            "hasNextPage": False,
                            "endCursor": None,
                        },
                        "nodes": [
                            {
                                "title": "PR on page 2",
                                "mergedAt": "2024-03-10T00:00:00Z",
                                "additions": 20,
                                "deletions": 3,
                                "changedFiles": 2,
                                "repository": {
                                    "nameWithOwner": "phoenixframework/phoenix",
                                    "stargazerCount": 20000,
                                    "forkCount": 2800,
                                    "primaryLanguage": {"name": "Elixir"},
                                    "isArchived": False,
                                    "isFork": False,
                                },
                            }
                        ],
                    },
                }
            }
        }

        route = respx.post(GRAPHQL_URL)
        route.side_effect = [
            httpx.Response(200, json=page1),
            httpx.Response(200, json=page2),
        ]

        async with _make_client() as client:
            prs = await client.fetch_user_merged_prs("testuser")

        assert len(prs) == 2
        assert prs[0].title == "PR on page 1"
        assert prs[1].title == "PR on page 2"


# ---------------------------------------------------------------------------
# fetch_repo_metadata_batch
# ---------------------------------------------------------------------------


class TestFetchRepoMetadataBatch:
    @respx.mock
    async def test_fetch_repo_metadata_batch(self) -> None:
        respx.post(GRAPHQL_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": {
                        "repo_0": {
                            "nameWithOwner": "elixir-lang/elixir",
                            "stargazerCount": 23000,
                            "forkCount": 3200,
                            "primaryLanguage": {"name": "Elixir"},
                            "isArchived": False,
                            "isFork": False,
                        },
                        "repo_1": {
                            "nameWithOwner": "phoenixframework/phoenix",
                            "stargazerCount": 20000,
                            "forkCount": 2800,
                            "primaryLanguage": {"name": "Elixir"},
                            "isArchived": False,
                            "isFork": False,
                        },
                        "repo_2": None,  # inaccessible repo, should be skipped
                    }
                },
            )
        )

        repos = [
            "elixir-lang/elixir",
            "phoenixframework/phoenix",
            "private-org/secret-repo",
        ]

        async with _make_client() as client:
            metadata = await client.fetch_repo_metadata_batch(repos)

        assert len(metadata) == 2
        assert "elixir-lang/elixir" in metadata
        assert "phoenixframework/phoenix" in metadata
        assert "private-org/secret-repo" not in metadata

        elixir = metadata["elixir-lang/elixir"]
        assert isinstance(elixir, RepoMetadata)
        assert elixir.stargazer_count == 23000
        assert elixir.primary_language == "Elixir"


# ---------------------------------------------------------------------------
# get_user_contribution_data
# ---------------------------------------------------------------------------


class TestGetUserContributionData:
    @respx.mock
    async def test_get_user_contribution_data(self) -> None:
        """Integration test: profile + PRs fetched in one combined query."""
        fixture = _load_fixture("user_prs.json")
        respx.post(GRAPHQL_URL).mock(
            return_value=httpx.Response(200, json=fixture)
        )

        async with _make_client() as client:
            data = await client.get_user_contribution_data("testuser")

        assert data.profile.login == "testuser"
        assert data.profile.followers_count == 50
        assert data.profile.is_bot is False
        assert len(data.merged_prs) == 2
        assert "elixir-lang/elixir" in data.contributed_repos
        assert "phoenixframework/phoenix" in data.contributed_repos
        assert data.contributed_repos["elixir-lang/elixir"].stargazer_count == 23000

    @respx.mock
    async def test_null_repository_skipped(self) -> None:
        """PRs with null repository (deleted/inaccessible) should be skipped."""
        fixture = {
            "data": {
                "user": {
                    "login": "testuser",
                    "createdAt": "2020-01-01T00:00:00Z",
                    "__typename": "User",
                    "followers": {"totalCount": 50},
                    "repositories": {"totalCount": 20},
                    "pullRequests": {
                        "totalCount": 3,
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                        "nodes": [
                            {
                                "title": "PR to deleted repo",
                                "mergedAt": "2024-06-15T00:00:00Z",
                                "additions": 10,
                                "deletions": 5,
                                "changedFiles": 1,
                                "repository": None,
                            },
                            {
                                "title": "PR to accessible repo",
                                "mergedAt": "2024-03-10T00:00:00Z",
                                "additions": 20,
                                "deletions": 3,
                                "changedFiles": 2,
                                "repository": {
                                    "nameWithOwner": "elixir-lang/elixir",
                                    "stargazerCount": 23000,
                                    "forkCount": 3200,
                                    "primaryLanguage": {"name": "Elixir"},
                                    "isArchived": False,
                                    "isFork": False,
                                },
                            },
                        ],
                    },
                }
            }
        }
        respx.post(GRAPHQL_URL).mock(
            return_value=httpx.Response(200, json=fixture)
        )

        async with _make_client() as client:
            data = await client.get_user_contribution_data("testuser")

        assert len(data.merged_prs) == 1
        assert data.merged_prs[0].title == "PR to accessible repo"
        assert "elixir-lang/elixir" in data.contributed_repos


# ---------------------------------------------------------------------------
# REST: PR comments
# ---------------------------------------------------------------------------


class TestPRComments:
    @respx.mock
    async def test_post_pr_comment(self) -> None:
        respx.post(f"{BASE_URL}/repos/my-org/my-repo/issues/42/comments").mock(
            return_value=httpx.Response(
                201,
                json={"id": 100, "body": "Hello"},
            )
        )

        async with _make_client() as client:
            result = await client.post_pr_comment("my-org", "my-repo", 42, "Hello")

        assert result["id"] == 100

    @respx.mock
    async def test_find_existing_comment_found(self) -> None:
        respx.get(f"{BASE_URL}/repos/my-org/my-repo/issues/42/comments").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {"id": 10, "body": "Unrelated comment"},
                    {"id": 20, "body": f"{COMMENT_MARKER}\nGood Egg: HIGH Trust"},
                ],
            )
        )

        async with _make_client() as client:
            comment_id = await client.find_existing_comment("my-org", "my-repo", 42)

        assert comment_id == 20

    @respx.mock
    async def test_find_existing_comment_not_found(self) -> None:
        respx.get(f"{BASE_URL}/repos/my-org/my-repo/issues/42/comments").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {"id": 10, "body": "Just a regular comment"},
                    {"id": 11, "body": "Another comment"},
                ],
            )
        )

        async with _make_client() as client:
            comment_id = await client.find_existing_comment("my-org", "my-repo", 42)

        assert comment_id is None


# ---------------------------------------------------------------------------
# REST: Check runs
# ---------------------------------------------------------------------------


class TestCheckRun:
    @respx.mock
    async def test_create_check_run(self) -> None:
        respx.post(f"{BASE_URL}/repos/my-org/my-repo/check-runs").mock(
            return_value=httpx.Response(
                201,
                json={"id": 999, "name": "Good Egg Trust Score"},
            )
        )

        async with _make_client() as client:
            result = await client.create_check_run(
                "my-org", "my-repo", "abc123", "Title", "Summary"
            )

        assert result["id"] == 999
        assert result["name"] == "Good Egg Trust Score"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @respx.mock
    async def test_rate_limit_error(self) -> None:
        respx.post(GRAPHQL_URL).mock(
            return_value=httpx.Response(
                403,
                json={"message": "API rate limit exceeded"},
                headers={"X-RateLimit-Reset": "1700000000"},
            )
        )

        async with _make_client() as client:
            with pytest.raises(RateLimitExhaustedError) as exc_info:
                await client.fetch_user_profile("testuser")

        assert exc_info.value.reset_at == datetime.fromtimestamp(
            1700000000, tz=UTC
        )

    @respx.mock
    async def test_rate_limit_error_429(self) -> None:
        respx.post(GRAPHQL_URL).mock(
            return_value=httpx.Response(
                429,
                json={"message": "rate limit"},
                headers={"X-RateLimit-Reset": "1700000000"},
            )
        )

        async with _make_client() as client:
            with pytest.raises(RateLimitExhaustedError):
                await client.fetch_user_profile("testuser")

    @respx.mock
    async def test_graphql_error(self) -> None:
        respx.post(GRAPHQL_URL).mock(
            return_value=httpx.Response(
                500,
                json={"message": "Internal Server Error"},
                headers={"X-RateLimit-Remaining": "4999"},
            )
        )

        async with _make_client() as client:
            with pytest.raises(GitHubAPIError) as exc_info:
                await client.fetch_user_profile("testuser")

        assert exc_info.value.status_code == 500
        assert exc_info.value.rate_limit_remaining == 4999


# ---------------------------------------------------------------------------
# Bot detection
# ---------------------------------------------------------------------------


class TestBotDetection:
    """Tests for the enhanced _is_bot_login heuristic."""

    @pytest.mark.parametrize(
        "login",
        [
            "dependabot[bot]",
            "renovate[bot]",
            "my-org-bot",
            "ci_bot",
            "my-deploy-app",
            "dependabot",
            "renovate",
            "greenkeeper",
            "snyk-bot",
            "codecov[bot]",
            "stale[bot]",
            "mergify[bot]",
            "allcontributors[bot]",
            "github-actions[bot]",
            "pre-commit-ci[bot]",
        ],
    )
    def test_is_bot_login_positive(self, login: str) -> None:
        assert GitHubClient._is_bot_login(login) is True

    @pytest.mark.parametrize(
        "login",
        [
            "gvanrossum",
            "torvalds",
            "josevalim",
            "sarah-dev",
            "john_smith",
            "mybotproject",  # "bot" in middle, not suffix/prefix match
        ],
    )
    def test_is_bot_login_negative(self, login: str) -> None:
        assert GitHubClient._is_bot_login(login) is False

    @respx.mock
    async def test_enhanced_bot_suffix_detection(self) -> None:
        """Login ending with _bot should be detected as bot."""
        respx.post(GRAPHQL_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": {
                        "user": {
                            "login": "ci_bot",
                            "createdAt": "2020-01-01T00:00:00Z",
                            "__typename": "User",
                            "followers": {"totalCount": 0},
                            "repositories": {"totalCount": 0},
                        }
                    }
                },
            )
        )

        async with _make_client() as client:
            profile = await client.fetch_user_profile("ci_bot")

        assert profile.is_bot is True

    @respx.mock
    async def test_suspected_bot_heuristic(self) -> None:
        """Zero followers + zero repos + old account = suspected bot."""
        respx.post(GRAPHQL_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": {
                        "user": {
                            "login": "ghost-account",
                            "createdAt": "2018-01-01T00:00:00Z",
                            "__typename": "User",
                            "followers": {"totalCount": 0},
                            "repositories": {"totalCount": 0},
                        }
                    }
                },
            )
        )

        async with _make_client() as client:
            profile = await client.fetch_user_profile("ghost-account")

        assert profile.is_bot is False
        assert profile.is_suspected_bot is True

    @respx.mock
    async def test_not_suspected_bot_when_has_followers(self) -> None:
        """Account with followers should not be suspected bot."""
        respx.post(GRAPHQL_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": {
                        "user": {
                            "login": "realuser",
                            "createdAt": "2018-01-01T00:00:00Z",
                            "__typename": "User",
                            "followers": {"totalCount": 10},
                            "repositories": {"totalCount": 0},
                        }
                    }
                },
            )
        )

        async with _make_client() as client:
            profile = await client.fetch_user_profile("realuser")

        assert profile.is_bot is False
        assert profile.is_suspected_bot is False

    async def test_bot_login_short_circuits_contribution_data(self) -> None:
        """Known bot logins should short-circuit without API calls."""
        async with _make_client() as client:
            data = await client.get_user_contribution_data("renovate[bot]")

        assert data.profile.is_bot is True
        assert data.merged_prs == []

    async def test_bot_prefix_short_circuits_contribution_data(self) -> None:
        """Bot prefix patterns should short-circuit."""
        async with _make_client() as client:
            data = await client.get_user_contribution_data("dependabot")

        assert data.profile.is_bot is True


# ---------------------------------------------------------------------------
# Context repo fetching
# ---------------------------------------------------------------------------


class TestContextRepoFetching:
    @respx.mock
    async def test_context_repo_metadata_fetched_when_missing(self) -> None:
        """When user hasn't contributed to context_repo, its metadata is fetched."""
        fixture = _load_fixture("user_prs.json")
        # First call: user contribution data
        # Second call: context repo metadata batch fetch
        context_repo_response = {
            "data": {
                "repo_0": {
                    "nameWithOwner": "rust-lang/rust",
                    "stargazerCount": 90000,
                    "forkCount": 12000,
                    "primaryLanguage": {"name": "Rust"},
                    "isArchived": False,
                    "isFork": False,
                }
            }
        }
        route = respx.post(GRAPHQL_URL)
        route.side_effect = [
            httpx.Response(200, json=fixture),
            httpx.Response(200, json=context_repo_response),
        ]

        async with _make_client() as client:
            data = await client.get_user_contribution_data(
                "testuser", context_repo="rust-lang/rust"
            )

        assert "rust-lang/rust" in data.contributed_repos
        assert data.contributed_repos["rust-lang/rust"].primary_language == "Rust"

    @respx.mock
    async def test_context_repo_not_refetched_when_present(self) -> None:
        """When user has contributed to context_repo, no extra fetch needed."""
        fixture = _load_fixture("user_prs.json")
        respx.post(GRAPHQL_URL).mock(
            return_value=httpx.Response(200, json=fixture)
        )

        async with _make_client() as client:
            data = await client.get_user_contribution_data(
                "testuser", context_repo="elixir-lang/elixir"
            )

        # elixir-lang/elixir is already in the fixture's PR repos
        assert "elixir-lang/elixir" in data.contributed_repos


# ---------------------------------------------------------------------------
# Cache integration
# ---------------------------------------------------------------------------


class TestCacheIntegration:
    @respx.mock
    async def test_cache_stores_contribution_data(self, tmp_path) -> None:
        """Contribution data should be stored in cache after fetching."""
        cache = Cache(db_path=tmp_path / "test.db")
        fixture = _load_fixture("user_prs.json")
        respx.post(GRAPHQL_URL).mock(
            return_value=httpx.Response(200, json=fixture)
        )

        async with _make_client(cache=cache) as client:
            await client.get_user_contribution_data("testuser")

        # Verify data was cached
        cached = cache.get("contribution_data:testuser")
        assert cached is not None
        assert cached["profile"]["login"] == "testuser"
        cache.close()

    @respx.mock
    async def test_cache_hit_skips_api_call(self, tmp_path) -> None:
        """When data is cached, API should not be called."""
        cache = Cache(db_path=tmp_path / "test.db")
        fixture = _load_fixture("user_prs.json")

        # First call stores in cache
        route = respx.post(GRAPHQL_URL)
        route.mock(return_value=httpx.Response(200, json=fixture))

        async with _make_client(cache=cache) as client:
            await client.get_user_contribution_data("testuser")

        call_count_after_first = route.call_count

        # Second call should use cache
        async with _make_client(cache=cache) as client:
            data = await client.get_user_contribution_data("testuser")

        assert route.call_count == call_count_after_first
        assert data.profile.login == "testuser"
        cache.close()

    @respx.mock
    async def test_cache_stores_repo_metadata(self, tmp_path) -> None:
        """Repo metadata should be cached individually."""
        cache = Cache(db_path=tmp_path / "test.db")
        respx.post(GRAPHQL_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": {
                        "repo_0": {
                            "nameWithOwner": "org/repo",
                            "stargazerCount": 500,
                            "forkCount": 50,
                            "primaryLanguage": {"name": "Python"},
                            "isArchived": False,
                            "isFork": False,
                        }
                    }
                },
            )
        )

        async with _make_client(cache=cache) as client:
            metadata = await client.fetch_repo_metadata_batch(["org/repo"])

        assert "org/repo" in metadata
        cached = cache.get("repo_metadata:org/repo")
        assert cached is not None
        assert cached["primary_language"] == "Python"
        cache.close()

    @respx.mock
    async def test_no_cache_still_works(self) -> None:
        """When cache is None, everything still works as before."""
        fixture = _load_fixture("user_prs.json")
        respx.post(GRAPHQL_URL).mock(
            return_value=httpx.Response(200, json=fixture)
        )

        async with _make_client(cache=None) as client:
            data = await client.get_user_contribution_data("testuser")

        assert data.profile.login == "testuser"
        assert len(data.merged_prs) == 2


# ---------------------------------------------------------------------------
# REST error handling
# ---------------------------------------------------------------------------


class TestRESTErrorHandling:
    """Tests for REST endpoint error handling with retry."""

    @respx.mock
    async def test_post_comment_403_no_retry(self) -> None:
        """403 on comment post should not retry (client error)."""
        route = respx.post(f"{BASE_URL}/repos/my-org/my-repo/issues/42/comments")
        route.mock(return_value=httpx.Response(403, json={"message": "Forbidden"}))

        async with _make_client() as client:
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                await client.post_pr_comment("my-org", "my-repo", 42, "Hello")

        assert exc_info.value.response.status_code == 403
        assert route.call_count == 1

    @respx.mock
    async def test_post_comment_404_no_retry(self) -> None:
        """404 on comment post should not retry."""
        route = respx.post(f"{BASE_URL}/repos/my-org/my-repo/issues/42/comments")
        route.mock(return_value=httpx.Response(404, json={"message": "Not Found"}))

        async with _make_client() as client:
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                await client.post_pr_comment("my-org", "my-repo", 42, "Hello")

        assert exc_info.value.response.status_code == 404
        assert route.call_count == 1

    @respx.mock
    async def test_update_comment_500_retries_then_fails(self) -> None:
        """500 on comment update should retry 4 times then fail."""
        route = respx.patch(f"{BASE_URL}/repos/my-org/my-repo/issues/comments/99")
        route.mock(return_value=httpx.Response(500, json={"message": "Internal Server Error"}))

        async with _make_client() as client:
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                await client.update_pr_comment("my-org", "my-repo", 99, "Updated")

        assert exc_info.value.response.status_code == 500
        assert route.call_count == 4

    @respx.mock
    async def test_find_existing_comment_500_retries_then_fails(self) -> None:
        """500 on find existing comment should retry 4 times then fail."""
        route = respx.get(f"{BASE_URL}/repos/my-org/my-repo/issues/42/comments")
        route.mock(return_value=httpx.Response(500, json={"message": "Internal Server Error"}))

        async with _make_client() as client:
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                await client.find_existing_comment("my-org", "my-repo", 42)

        assert exc_info.value.response.status_code == 500
        assert route.call_count == 4

    @respx.mock
    async def test_create_check_run_422_no_retry(self) -> None:
        """422 on create check run should not retry (client error)."""
        route = respx.post(f"{BASE_URL}/repos/my-org/my-repo/check-runs")
        route.mock(return_value=httpx.Response(422, json={"message": "Unprocessable"}))

        async with _make_client() as client:
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                await client.create_check_run("my-org", "my-repo", "sha123", "Title", "Summary")

        assert exc_info.value.response.status_code == 422
        assert route.call_count == 1

    @respx.mock
    async def test_create_check_run_500_retries_then_fails(self) -> None:
        """500 on create check run should retry 4 times then fail."""
        route = respx.post(f"{BASE_URL}/repos/my-org/my-repo/check-runs")
        route.mock(return_value=httpx.Response(500, json={"message": "Internal Server Error"}))

        async with _make_client() as client:
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                await client.create_check_run("my-org", "my-repo", "sha123", "Title", "Summary")

        assert exc_info.value.response.status_code == 500
        assert route.call_count == 4


# ---------------------------------------------------------------------------
# Cascading failures
# ---------------------------------------------------------------------------


class TestCascadingFailures:
    """Tests for cascading failure scenarios."""

    @respx.mock
    async def test_user_prs_ok_but_context_repo_fetch_fails(self) -> None:
        """User+PRs fetch OK but context repo metadata fetch fails -> GitHubAPIError."""
        fixture = _load_fixture("user_prs.json")
        # First call succeeds (user contribution data)
        # Second call fails (context repo metadata batch fetch)
        route = respx.post(GRAPHQL_URL)
        route.side_effect = [
            httpx.Response(200, json=fixture),
            # 4 retries for the context repo metadata fetch (502 is transient)
            httpx.Response(502, json={"message": "Bad Gateway"}),
            httpx.Response(502, json={"message": "Bad Gateway"}),
            httpx.Response(502, json={"message": "Bad Gateway"}),
            httpx.Response(502, json={"message": "Bad Gateway"}),
        ]

        async with _make_client() as client:
            with pytest.raises(GitHubAPIError):
                await client.get_user_contribution_data(
                    "testuser", context_repo="rust-lang/rust"
                )


# ---------------------------------------------------------------------------
# Timeout handling
# ---------------------------------------------------------------------------


class TestTimeoutHandling:
    """Tests for timeout handling."""

    @respx.mock
    async def test_graphql_timeout_raises(self) -> None:
        """GraphQL request timeout should raise."""
        respx.post(GRAPHQL_URL).mock(side_effect=httpx.ReadTimeout("Read timed out"))

        async with _make_client() as client:
            with pytest.raises(httpx.ReadTimeout):
                await client.fetch_user_profile("testuser")

    @respx.mock
    async def test_rest_timeout_raises(self) -> None:
        """REST request timeout should raise."""
        respx.post(f"{BASE_URL}/repos/my-org/my-repo/issues/42/comments").mock(
            side_effect=httpx.ReadTimeout("Read timed out")
        )

        async with _make_client() as client:
            with pytest.raises(httpx.ReadTimeout):
                await client.post_pr_comment("my-org", "my-repo", 42, "Hello")


# ---------------------------------------------------------------------------
# Retry behavior
# ---------------------------------------------------------------------------


class TestRetryBehavior:
    """Tests for retry/backoff behavior."""

    @respx.mock
    async def test_retry_on_rate_limit(self) -> None:
        """Rate limit error should be retried after sleeping."""
        route = respx.post(GRAPHQL_URL)
        route.side_effect = [
            httpx.Response(
                429,
                json={"message": "rate limit"},
                headers={"X-RateLimit-Reset": "1700000000"},
            ),
            httpx.Response(
                200,
                json={
                    "data": {
                        "user": {
                            "login": "testuser",
                            "createdAt": "2020-01-01T00:00:00Z",
                            "__typename": "User",
                            "followers": {"totalCount": 50},
                            "repositories": {"totalCount": 20},
                        }
                    }
                },
            ),
        ]

        from unittest.mock import patch as mock_patch

        with mock_patch("good_egg.github_client.asyncio.sleep", return_value=None) as mock_sleep:
            async with _make_client() as client:
                profile = await client.fetch_user_profile("testuser")

        assert profile.login == "testuser"
        assert route.call_count == 2
        assert mock_sleep.call_count >= 1

    @respx.mock
    async def test_retry_on_502(self) -> None:
        """502 Bad Gateway should be retried."""
        route = respx.post(GRAPHQL_URL)
        route.side_effect = [
            httpx.Response(
                502,
                json={"message": "Bad Gateway"},
                headers={"X-RateLimit-Remaining": "4999"},
            ),
            httpx.Response(
                200,
                json={
                    "data": {
                        "user": {
                            "login": "testuser",
                            "createdAt": "2020-01-01T00:00:00Z",
                            "__typename": "User",
                            "followers": {"totalCount": 50},
                            "repositories": {"totalCount": 20},
                        }
                    }
                },
            ),
        ]

        async with _make_client() as client:
            profile = await client.fetch_user_profile("testuser")

        assert profile.login == "testuser"
        assert route.call_count == 2

    @respx.mock
    async def test_max_retries_exceeded(self) -> None:
        """Persistent 502 should fail after 4 attempts."""
        route = respx.post(GRAPHQL_URL)
        route.mock(
            return_value=httpx.Response(
                502,
                json={"message": "Bad Gateway"},
                headers={"X-RateLimit-Remaining": "4999"},
            )
        )

        async with _make_client() as client:
            with pytest.raises(GitHubAPIError) as exc_info:
                await client.fetch_user_profile("testuser")

        assert exc_info.value.status_code == 502
        assert route.call_count == 4

    @respx.mock
    async def test_no_retry_on_404(self) -> None:
        """User not found (404-like) should not be retried."""
        route = respx.post(GRAPHQL_URL)
        route.mock(
            return_value=httpx.Response(200, json={"data": {"user": None}})
        )

        async with _make_client() as client:
            with pytest.raises(UserNotFoundError):
                await client.fetch_user_profile("ghost-user")

        assert route.call_count == 1

    @respx.mock
    async def test_rest_retry_on_5xx(self) -> None:
        """REST 503 should be retried, then succeed."""
        route = respx.post(f"{BASE_URL}/repos/my-org/my-repo/issues/42/comments")
        route.side_effect = [
            httpx.Response(503, json={"message": "Service Unavailable"}),
            httpx.Response(201, json={"id": 100, "body": "Hello"}),
        ]

        async with _make_client() as client:
            result = await client.post_pr_comment("my-org", "my-repo", 42, "Hello")

        assert result["id"] == 100
        assert route.call_count == 2

    @respx.mock
    async def test_rest_no_retry_on_4xx(self) -> None:
        """REST 403 should not be retried."""
        route = respx.post(f"{BASE_URL}/repos/my-org/my-repo/issues/42/comments")
        route.mock(return_value=httpx.Response(403, json={"message": "Forbidden"}))

        async with _make_client() as client:
            with pytest.raises(httpx.HTTPStatusError):
                await client.post_pr_comment("my-org", "my-repo", 42, "Hello")

        assert route.call_count == 1
