"""MCP server for AI assistant integration."""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from good_egg.cache import Cache
from good_egg.config import GoodEggConfig, load_config
from good_egg.scorer import score_pr_author

mcp = FastMCP("good-egg")


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


@mcp.tool()
async def score_user(username: str, repo: str) -> str:
    """Score a GitHub user's trustworthiness relative to a repository.

    Returns the full trust score with all metadata as JSON.

    Args:
        username: GitHub username to score.
        repo: Target repository in owner/repo format.
    """
    try:
        repo_owner, repo_name = _parse_repo(repo)
    except ValueError as exc:
        return _error_json(str(exc))

    config = _get_config()
    cache = _get_cache(config)
    try:
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
    finally:
        cache.close()


@mcp.tool()
async def check_pr_author(username: str, repo: str) -> str:
    """Quick check of a PR author's trust level.

    Returns a compact summary with trust level, score, and PR count.

    Args:
        username: GitHub username to check.
        repo: Target repository in owner/repo format.
    """
    try:
        repo_owner, repo_name = _parse_repo(repo)
    except ValueError as exc:
        return _error_json(str(exc))

    config = _get_config()
    cache = _get_cache(config)
    try:
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
        }
        return json.dumps(summary)
    except Exception as exc:
        return _error_json(str(exc))
    finally:
        cache.close()


@mcp.tool()
async def get_trust_details(username: str, repo: str) -> str:
    """Get an expanded trust breakdown for a GitHub user.

    Returns detailed information including contributions, flags, and metadata.

    Args:
        username: GitHub username to analyse.
        repo: Target repository in owner/repo format.
    """
    try:
        repo_owner, repo_name = _parse_repo(repo)
    except ValueError as exc:
        return _error_json(str(exc))

    config = _get_config()
    cache = _get_cache(config)
    try:
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
        }
        return json.dumps(details)
    except Exception as exc:
        return _error_json(str(exc))
    finally:
        cache.close()


@mcp.tool()
async def cache_stats() -> str:
    """Show cache statistics.

    Returns cache entry counts, categories, and database size.
    """
    config = _get_config()
    cache = _get_cache(config)
    try:
        stats = cache.stats()
        return json.dumps(stats)
    except Exception as exc:
        return _error_json(str(exc))
    finally:
        cache.close()


@mcp.tool()
async def clear_cache(category: str | None = None) -> str:
    """Clear the cache.

    Optionally clear only a specific category. Without a category,
    removes all expired entries.

    Args:
        category: Optional cache category to clear (e.g. 'repo_metadata').
    """
    config = _get_config()
    cache = _get_cache(config)
    try:
        if category:
            cache.invalidate_category(category)
            return json.dumps({"cleared_category": category})
        removed = cache.cleanup_expired()
        return json.dumps({"expired_entries_removed": removed})
    except Exception as exc:
        return _error_json(str(exc))
    finally:
        cache.close()


def main() -> None:
    """Run the Good Egg MCP server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
