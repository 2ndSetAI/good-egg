"""MCP server for AI assistant integration."""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    FastMCP = None  # type: ignore[assignment,misc]

from good_egg.cache import Cache
from good_egg.config import GoodEggConfig, load_config
from good_egg.scorer import score_pr_author


def _get_config() -> GoodEggConfig:
    """Load the Good Egg configuration."""
    return load_config()


def _get_cache(config: GoodEggConfig) -> Cache:
    """Create a cache instance from configuration."""
    return Cache(ttls=config.cache_ttl.to_seconds())


def _parse_repo(repo: str) -> tuple[str, str]:
    """Parse an owner/repo string into (owner, name).

    Raises ValueError if the format is invalid.
    """
    parts = repo.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        msg = f"repo must be in owner/name format, got: {repo!r}"
        raise ValueError(msg)
    return parts[0], parts[1]


def _error_json(message: str) -> str:
    """Return a JSON error string."""
    return json.dumps({"error": message})


@asynccontextmanager
async def _scoring_resources(
    repo: str,
    scoring_model: str | None = None,
    force_score: bool = False,
) -> AsyncGenerator[tuple[GoodEggConfig, Cache, str, str]]:
    """Set up config, cache, and parsed repo for scoring tools.

    Yields (config, cache, repo_owner, repo_name) and ensures the cache
    is closed on exit.
    """
    config = _get_config()
    if scoring_model is not None and scoring_model in ("v1", "v2"):
        config = config.model_copy(update={"scoring_model": scoring_model})
    if force_score:
        config = config.model_copy(update={"skip_known_contributors": False})
    cache = _get_cache(config)
    try:
        repo_owner, repo_name = _parse_repo(repo)
        yield config, cache, repo_owner, repo_name
    finally:
        cache.close()


@asynccontextmanager
async def _cache_resource() -> AsyncGenerator[Cache]:
    """Set up config and cache for cache-only tools.

    Yields a Cache instance and ensures it is closed on exit.
    """
    config = _get_config()
    cache = _get_cache(config)
    try:
        yield cache
    finally:
        cache.close()


async def score_user(
    username: str,
    repo: str,
    scoring_model: str | None = None,
    force_score: bool = False,
) -> str:
    """Score a GitHub user's trustworthiness relative to a repository.

    Returns the full trust score with all metadata as JSON.

    Args:
        username: GitHub username to score.
        repo: Target repository in owner/repo format.
        scoring_model: Optional scoring model override (v1 or v2).
        force_score: Force full scoring even for known contributors.
    """
    try:
        async with _scoring_resources(repo, scoring_model, force_score) as (
            config, cache, repo_owner, repo_name,
        ):
            result = await score_pr_author(
                login=username,
                repo_owner=repo_owner,
                repo_name=repo_name,
                config=config,
                cache=cache,
            )
            return result.model_dump_json()
    except Exception as exc:
        return _error_json(str(exc))


async def check_pr_author(
    username: str,
    repo: str,
    scoring_model: str | None = None,
    force_score: bool = False,
) -> str:
    """Quick check of a PR author's trust level.

    Returns a compact summary with trust level, score, and PR count.

    Args:
        username: GitHub username to check.
        repo: Target repository in owner/repo format.
        scoring_model: Optional scoring model override (v1 or v2).
        force_score: Force full scoring even for known contributors.
    """
    try:
        async with _scoring_resources(repo, scoring_model, force_score) as (
            config, cache, repo_owner, repo_name,
        ):
            result = await score_pr_author(
                login=username,
                repo_owner=repo_owner,
                repo_name=repo_name,
                config=config,
                cache=cache,
            )
            summary: dict[str, Any] = {
                "user_login": result.user_login,
                "trust_level": result.trust_level.value,
                "normalized_score": result.normalized_score,
                "total_merged_prs": result.total_merged_prs,
                "scoring_model": result.scoring_model,
            }
            if result.component_scores:
                summary["component_scores"] = result.component_scores
            return json.dumps(summary)
    except Exception as exc:
        return _error_json(str(exc))


async def get_trust_details(
    username: str,
    repo: str,
    scoring_model: str | None = None,
    force_score: bool = False,
) -> str:
    """Get an expanded trust breakdown for a GitHub user.

    Returns detailed information including contributions, flags, and metadata.

    Args:
        username: GitHub username to analyse.
        repo: Target repository in owner/repo format.
        scoring_model: Optional scoring model override (v1 or v2).
        force_score: Force full scoring even for known contributors.
    """
    try:
        async with _scoring_resources(repo, scoring_model, force_score) as (
            config, cache, repo_owner, repo_name,
        ):
            result = await score_pr_author(
                login=username,
                repo_owner=repo_owner,
                repo_name=repo_name,
                config=config,
                cache=cache,
            )
            details: dict[str, Any] = {
                "user_login": result.user_login,
                "context_repo": result.context_repo,
                "trust_level": result.trust_level.value,
                "normalized_score": result.normalized_score,
                "raw_score": result.raw_score,
                "account_age_days": result.account_age_days,
                "total_merged_prs": result.total_merged_prs,
                "unique_repos_contributed": result.unique_repos_contributed,
                "language_match": result.language_match,
                "top_contributions": [
                    c.model_dump() for c in result.top_contributions
                ],
                "flags": result.flags,
                "scoring_metadata": result.scoring_metadata,
                "scoring_model": result.scoring_model,
                "component_scores": result.component_scores,
            }
            return json.dumps(details)
    except Exception as exc:
        return _error_json(str(exc))


async def cache_stats() -> str:
    """Show cache statistics.

    Returns cache entry counts, categories, and database size.
    """
    try:
        async with _cache_resource() as cache:
            stats = cache.stats()
            return json.dumps(stats)
    except Exception as exc:
        return _error_json(str(exc))


async def clear_cache(category: str | None = None) -> str:
    """Clear the cache.

    Optionally clear only a specific category. Without a category,
    removes all expired entries.

    Args:
        category: Optional cache category to clear (e.g. 'repo_metadata').
    """
    try:
        async with _cache_resource() as cache:
            if category:
                cache.invalidate_category(category)
                return json.dumps({"cleared_category": category})
            removed = cache.cleanup_expired()
            return json.dumps({"expired_entries_removed": removed})
    except Exception as exc:
        return _error_json(str(exc))


def main() -> None:
    """Run the Good Egg MCP server."""
    if FastMCP is None:
        print(
            "The MCP server requires the 'mcp' extra.\n"
            "Install it with: pip install good-egg[mcp]",
            file=sys.stderr,
        )
        sys.exit(1)
    server = FastMCP("good-egg")
    server.tool()(score_user)
    server.tool()(check_pr_author)
    server.tool()(get_trust_details)
    server.tool()(cache_stats)
    server.tool()(clear_cache)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
