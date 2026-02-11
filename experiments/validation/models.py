from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PROutcome(enum.StrEnum):
    """Three-class PR outcome."""
    MERGED = "merged"
    REJECTED = "rejected"
    POCKET_VETO = "pocket_veto"


class CollectedPR(BaseModel):
    """A PR collected from GitHub in Stage 1."""
    repo: str                          # owner/repo
    number: int
    author_login: str
    title: str
    state: str                         # MERGED, CLOSED, OPEN
    created_at: datetime
    merged_at: datetime | None = None
    closed_at: datetime | None = None
    additions: int = 0
    deletions: int = 0
    changed_files: int = 0
    temporal_bin: str                  # e.g. "2024H1"


class ClassifiedPR(BaseModel):
    """A PR with outcome classification (Stage 2 output)."""
    repo: str
    number: int
    author_login: str
    title: str
    state: str
    created_at: datetime
    merged_at: datetime | None = None
    closed_at: datetime | None = None
    additions: int = 0
    deletions: int = 0
    changed_files: int = 0
    temporal_bin: str
    outcome: PROutcome
    stale_threshold_days: float


class AuthorRecord(BaseModel):
    """Unique author discovered in Stage 2, enriched in Stage 3."""
    login: str
    # Tier 1 unmerged PR stats (all authors)
    merged_count: int | None = None
    closed_count: int | None = None
    open_count: int | None = None
    # Tier 2 detailed unmerged data (subset)
    tier2_sampled: bool = False
    tier2_timeout_count: int | None = None
    tier2_explicit_rejection_count: int | None = None
    # Path to persisted UserContributionData JSON
    data_path: str | None = None


class ScoredPR(BaseModel):
    """A scored PR (Stage 4 output) - one row per PR per model variant."""
    repo: str
    pr_number: int
    author_login: str
    outcome: PROutcome
    temporal_bin: str
    created_at: datetime
    # Full model scores
    raw_score: float = 0.0
    normalized_score: float = 0.0
    trust_level: str = ""
    total_prs_at_time: int = 0      # PRs available after anti-lookahead
    unique_repos_at_time: int = 0
    # Ablation scores (normalized_score for each variant)
    ablation_scores: dict[str, float] = Field(default_factory=dict)


class FeatureRow(BaseModel):
    """Complete feature vector for analysis (Stage 5 output)."""
    repo: str
    pr_number: int
    author_login: str
    outcome: PROutcome
    temporal_bin: str
    created_at: datetime
    # GE scores
    ge_normalized_score: float = 0.0
    ge_trust_level: str = ""
    # Ablation scores
    ablation_scores: dict[str, float] = Field(default_factory=dict)
    # Candidate features
    log_account_age_days: float = 0.0
    log_followers: float = 0.0
    log_public_repos: float = 0.0
    # Author merge rate (H5) - Tier 1
    author_merge_rate: float | None = None
    author_open_pr_count: int | None = None
    # Author pocket veto rate (H5) - Tier 2
    author_pocket_veto_rate: float | None = None
    # Semantic similarity (H4)
    embedding_similarity: float | None = None
    # PR-level confounds
    log_pr_size: float = 0.0
    pr_changed_files: int = 0


class RepoEntry(BaseModel):
    """A repository entry from repo_list.yaml."""
    name: str
    language: str = ""
    stars: int = 0
    domain: str = ""


class TemporalBin(BaseModel):
    """A temporal bin definition from study_config.yaml."""
    label: str
    start: str  # ISO date string
    end: str


class StudyConfig(BaseModel):
    """Parsed study configuration from study_config.yaml."""
    temporal_bins: list[TemporalBin] = Field(default_factory=list)
    stale_threshold_bin: str = "2024H1"
    collection: dict[str, Any] = Field(default_factory=dict)
    classification: dict[str, Any] = Field(default_factory=dict)
    author_filtering: dict[str, Any] = Field(default_factory=dict)
    fetch: dict[str, Any] = Field(default_factory=dict)
    scoring: dict[str, Any] = Field(default_factory=dict)
    features: dict[str, Any] = Field(default_factory=dict)
    analysis: dict[str, Any] = Field(default_factory=dict)
    ablations: dict[str, Any] = Field(default_factory=dict)
    paths: dict[str, str] = Field(default_factory=dict)


class StatisticalResult(BaseModel):
    """Result of a statistical test."""
    test_name: str
    statistic: float | None = None
    p_value: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    effect_size: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)
