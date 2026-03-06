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


class BotDetectionPR(BaseModel):
    """A PR in the bot detection corpus."""

    repo: str
    number: int
    author: str
    title: str
    body: str = ""
    created_at: datetime
    merged_at: datetime | None = None
    closed_at: datetime | None = None
    state: str
    additions: int = 0
    deletions: int = 0
    files_changed: int = 0
    labels: list[str] = Field(default_factory=list)
    outcome: PROutcome | None = None
    stale_threshold_days: float | None = None


class ReviewRecord(BaseModel):
    """A review on a PR."""

    repo: str
    pr_number: int
    reviewer: str
    state: str
    body: str = ""
    submitted_at: datetime


class CommitRecord(BaseModel):
    """A commit on a PR."""

    repo: str
    pr_number: int
    sha: str
    author: str
    message: str = ""
    committed_at: datetime


class AuthorRecord(BaseModel):
    """Author metadata."""

    login: str
    account_created_at: datetime | None = None
    followers: int = 0
    public_repos: int = 0
    is_bot: bool = False
    ge_score: float | None = None
    ge_trust_level: str | None = None
    account_status: str | None = None


class BurstinessFeatures(BaseModel):
    """H1: Burstiness signal features for a single PR."""

    burst_count_1h: int = 0
    burst_repos_1h: int = 0
    burst_count_24h: int = 0
    burst_repos_24h: int = 0
    burst_max_rate: float = 0.0


class EngagementFeatures(BaseModel):
    """H2: Engagement lifecycle features for a single PR."""

    review_response_rate: float | None = None
    ci_failure_followup_rate: float | None = None
    avg_response_latency_hours: float | None = None
    abandoned_pr_rate: float | None = None


class CrossRepoFeatures(BaseModel):
    """H3: Cross-repo fingerprinting features for a single PR."""

    max_title_similarity: float = 0.0
    language_entropy: float = 0.0
    topic_coherence: float = 0.0
    duplicate_title_count: int = 0


class FeatureRow(BaseModel):
    """Complete feature vector for one PR, combining all hypotheses."""

    repo: str
    number: int
    author: str
    outcome: PROutcome
    created_at: datetime

    # H1: Burstiness
    burst_count_1h: int = 0
    burst_repos_1h: int = 0
    burst_count_24h: int = 0
    burst_repos_24h: int = 0
    burst_max_rate: float = 0.0

    # H2: Engagement
    review_response_rate: float | None = None
    ci_failure_followup_rate: float | None = None
    avg_response_latency_hours: float | None = None
    abandoned_pr_rate: float | None = None

    # H3: Cross-repo
    max_title_similarity: float = 0.0
    language_entropy: float = 0.0
    topic_coherence: float = 0.0
    duplicate_title_count: int = 0

    # GE scores (computed with anti-lookahead)
    ge_score_v1: float | None = None
    ge_score_v2: float | None = None

    # H6: Interaction features (burstiness x novelty)
    burst_no_prior_merge: int = 0
    burst_first_time_repo: int = 0
    burst_low_ge: int = 0
    burst_new_account: int = 0

    # H7: Burst content homogeneity
    burst_title_embedding_sim: float | None = None
    burst_body_embedding_sim: float | None = None
    burst_size_cv: float | None = None
    burst_file_pattern_entropy: float | None = None

    # Author metadata baselines
    account_age_days: float | None = None
    followers: int | None = None
    public_repos: int | None = None
    account_status: str | None = None


class StageCheckpoint(BaseModel):
    """Checkpoint written after each stage completes."""

    stage: str
    timestamp: datetime
    row_counts: dict[str, int] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)


class AuthorFeatureRow(BaseModel):
    """Complete feature vector for one author, combining H8-H13."""

    login: str

    # Ground truth
    account_status: str | None = None

    # H8: Author-level aggregates
    total_prs: int = 0
    total_repos: int = 0
    merge_rate: float = 0.0
    rejection_rate: float = 0.0
    pocket_veto_rate: float = 0.0
    median_additions: float = 0.0
    median_deletions: float = 0.0
    median_files_changed: float = 0.0
    mean_title_length: float = 0.0
    mean_body_length: float = 0.0
    empty_body_rate: float = 0.0
    active_days: int = 0
    career_span_days: float = 0.0
    prs_per_active_day: float = 0.0
    repos_per_active_day: float = 0.0
    account_age_days: float | None = None
    followers: int | None = None
    public_repos: int | None = None

    # H9: Time-series features
    inter_pr_cv: float | None = None
    inter_pr_median_hours: float | None = None
    max_dormancy_days: float | None = None
    burst_episode_count: int | None = None
    dormancy_burst_ratio: float | None = None
    weekend_ratio: float | None = None
    hour_entropy: float | None = None
    regularity_score: float | None = None

    # H10: Network features
    hub_score: float | None = None
    bipartite_clustering: float | None = None
    author_projection_degree: int | None = None
    repo_diversity_entropy: float | None = None
    connected_component_size: int | None = None
    mean_repo_popularity: float | None = None
    isolation_score: float | None = None

    # H11: LLM content analysis
    llm_spam_score: float | None = None

    # H13: Semi-supervised
    knn_distance_to_seed: float | None = None
    isolation_forest_score: float | None = None


class StudyConfig(BaseModel):
    """Parsed study configuration."""

    data_sources: dict[str, Any] = Field(default_factory=dict)
    classification: dict[str, Any] = Field(default_factory=dict)
    burstiness_sweep: dict[str, Any] = Field(default_factory=dict)
    analysis: dict[str, Any] = Field(default_factory=dict)
    scale: dict[str, Any] = Field(default_factory=dict)
    paths: dict[str, str] = Field(default_factory=dict)
    bot_patterns: list[str] = Field(default_factory=list)
    author_analysis: dict[str, Any] = Field(default_factory=dict)
