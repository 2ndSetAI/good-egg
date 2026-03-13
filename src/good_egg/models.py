"""Data models for Good Egg trust scoring."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, computed_field


class TrustLevel(StrEnum):
    """Trust classification levels."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"
    BOT = "BOT"
    EXISTING_CONTRIBUTOR = "EXISTING_CONTRIBUTOR"


class SuspicionLevel(StrEnum):
    """Suspension advisory risk levels."""
    HIGH = "HIGH"
    ELEVATED = "ELEVATED"
    NORMAL = "NORMAL"


class SuspicionScore(BaseModel):
    """Advisory suspension risk score."""
    raw_score: float = 0.0
    probability: float = 0.0
    suspicion_level: SuspicionLevel = SuspicionLevel.NORMAL
    component_scores: dict[str, float] = {}


class UserProfile(BaseModel):
    """GitHub user profile data."""
    login: str
    created_at: datetime
    followers_count: int = 0
    public_repos_count: int = 0
    is_bot: bool = False
    is_suspected_bot: bool = False

    @computed_field  # type: ignore[prop-decorator]
    @property
    def account_age_days(self) -> int:
        delta = datetime.now(UTC) - self.created_at
        return max(0, delta.days)


class RepoMetadata(BaseModel):
    """GitHub repository metadata."""
    name_with_owner: str
    stargazer_count: int = 0
    fork_count: int = 0
    primary_language: str | None = None
    is_archived: bool = False
    is_fork: bool = False


class MergedPR(BaseModel):
    """A merged pull request."""
    repo_name_with_owner: str
    title: str
    merged_at: datetime
    additions: int = 0
    deletions: int = 0
    changed_files: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def days_ago(self) -> int:
        delta = datetime.now(UTC) - self.merged_at
        return max(0, delta.days)


class UserContributionData(BaseModel):
    """Complete contribution data for a user."""
    profile: UserProfile
    merged_prs: list[MergedPR] = []
    contributed_repos: dict[str, RepoMetadata] = {}
    closed_pr_count: int = 0
    repo_contributors: dict[str, list[str]] = {}


class ContributionSummary(BaseModel):
    """Summary of contributions to a single repo."""
    repo_name: str
    pr_count: int
    language: str | None = None
    stars: int = 0


class FreshAccountAdvisory(BaseModel):
    """Advisory for accounts less than one year old."""
    is_fresh: bool
    account_age_days: int
    created_at: datetime | None = None
    threshold_days: int = 365


class TrustScore(BaseModel):
    """Complete trust score result."""
    user_login: str
    context_repo: str
    raw_score: float = 0.0
    normalized_score: float = 0.0
    trust_level: TrustLevel = TrustLevel.UNKNOWN

    account_age_days: int = 0
    total_merged_prs: int = 0
    unique_repos_contributed: int = 0
    top_contributions: list[ContributionSummary] = []
    language_match: bool = False
    flags: dict[str, bool] = {}
    scoring_metadata: dict[str, Any] = {}
    scoring_model: str = "v1"
    component_scores: dict[str, float] = {}
    fresh_account: FreshAccountAdvisory | None = None
    suspicion_score: SuspicionScore | None = None
