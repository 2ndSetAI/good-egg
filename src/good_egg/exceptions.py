"""Custom exception hierarchy for Good Egg."""

from __future__ import annotations

from datetime import datetime


class GoodEggError(Exception):
    """Base exception for Good Egg."""


class GitHubAPIError(GoodEggError):
    """Error from the GitHub API."""

    def __init__(
        self,
        message: str,
        status_code: int = 0,
        rate_limit_remaining: int | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.rate_limit_remaining = rate_limit_remaining


class RateLimitExhaustedError(GitHubAPIError):
    """GitHub API rate limit exhausted."""

    def __init__(self, reset_at: datetime, rate_limit_remaining: int = 0):
        self.reset_at = reset_at
        super().__init__(
            f"Rate limit exhausted. Resets at {reset_at.isoformat()}",
            status_code=403,
            rate_limit_remaining=rate_limit_remaining,
        )


class UserNotFoundError(GitHubAPIError):
    """GitHub user not found."""

    def __init__(self, login: str):
        self.login = login
        super().__init__(f"User not found: {login}", status_code=404)


class RepoNotFoundError(GitHubAPIError):
    """GitHub repository not found."""

    def __init__(self, repo: str):
        self.repo = repo
        super().__init__(f"Repository not found: {repo}", status_code=404)


class CacheError(GoodEggError):
    """Error with the cache layer."""


class ConfigError(GoodEggError):
    """Error with configuration."""


class InsufficientDataError(GoodEggError):
    """Not enough data to produce a meaningful score."""
