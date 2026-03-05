from __future__ import annotations

import logging
import math
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from experiments.bot_detection.cache import BotDetectionDB
from experiments.bot_detection.checkpoint import write_stage_checkpoint
from experiments.bot_detection.models import (
    BurstinessFeatures,
    CrossRepoFeatures,
    EngagementFeatures,
    FeatureRow,
    PROutcome,
    StudyConfig,
)
from good_egg.config import load_config
from good_egg.models import MergedPR, UserContributionData, UserProfile
from good_egg.scorer import TrustScorer

logger = logging.getLogger(__name__)


# ----------------------------------------------------------
# H1: Burstiness features
# ----------------------------------------------------------


def compute_burstiness(
    prior_prs: list[dict[str, Any]],
    test_time: datetime,
) -> BurstinessFeatures:
    """Compute burstiness features from author's prior cross-repo PRs.

    All PRs in prior_prs must already be filtered to:
    - repo != test_repo
    - created_at < test_time
    """
    if not prior_prs:
        return BurstinessFeatures()

    timestamps = []
    for pr in prior_prs:
        ca = pr.get("created_at")
        if ca is None:
            continue
        if isinstance(ca, str):
            continue
        timestamps.append(ca)

    if not timestamps:
        return BurstinessFeatures()

    timestamps.sort()

    # Count PRs and repos in windows before test_time
    t = test_time
    window_1h = t - timedelta(hours=1)
    window_24h = t - timedelta(hours=24)

    prs_1h = [pr for pr in prior_prs if pr.get("created_at") and pr["created_at"] >= window_1h]
    prs_24h = [
        pr for pr in prior_prs if pr.get("created_at") and pr["created_at"] >= window_24h
    ]

    burst_count_1h = len(prs_1h)
    burst_repos_1h = len({pr["repo"] for pr in prs_1h})
    burst_count_24h = len(prs_24h)
    burst_repos_24h = len({pr["repo"] for pr in prs_24h})

    # Peak rate: max PRs/hour in any 1h sliding window over prior 7d
    window_7d = t - timedelta(days=7)
    recent_ts = [ts for ts in timestamps if ts >= window_7d]
    burst_max_rate = 0.0
    if len(recent_ts) >= 2:
        for i, start in enumerate(recent_ts):
            end = start + timedelta(hours=1)
            count = sum(1 for ts in recent_ts[i:] if ts < end)
            burst_max_rate = max(burst_max_rate, float(count))

    return BurstinessFeatures(
        burst_count_1h=burst_count_1h,
        burst_repos_1h=burst_repos_1h,
        burst_count_24h=burst_count_24h,
        burst_repos_24h=burst_repos_24h,
        burst_max_rate=burst_max_rate,
    )


# ----------------------------------------------------------
# H2: Engagement lifecycle features
# ----------------------------------------------------------


def compute_engagement(
    prior_prs: list[dict[str, Any]],
    reviews_by_pr: dict[tuple[str, int], list[dict[str, Any]]],
    commits_by_pr: dict[tuple[str, int], list[dict[str, Any]]],
    author: str,
) -> EngagementFeatures:
    """Compute engagement features from author's prior cross-repo PRs.

    reviews_by_pr and commits_by_pr are pre-fetched for the author's
    prior PRs (already filtered by anti-lookahead).
    """
    if not prior_prs:
        return EngagementFeatures()

    response_counts = 0
    review_counts = 0
    followup_counts = 0
    changes_requested_counts = 0
    response_latencies: list[float] = []
    abandoned_count = 0
    total_closed = 0

    for pr in prior_prs:
        repo = pr["repo"]
        number = pr["number"]
        key = (repo, number)

        reviews = reviews_by_pr.get(key, [])
        commits = commits_by_pr.get(key, [])

        # Reviews from others (not self-reviews)
        other_reviews = [r for r in reviews if r.get("reviewer") != author]

        if other_reviews:
            # Check if author responded to reviews
            author_comments = [
                r for r in reviews
                if r.get("reviewer") == author and r.get("submitted_at")
            ]

            for review in other_reviews:
                review_counts += 1
                review_time = review.get("submitted_at")
                if review_time is None:
                    continue

                # Check for author response after this review
                responded = any(
                    ac.get("submitted_at") and ac["submitted_at"] > review_time
                    for ac in author_comments
                )
                # Also count commits after review as response
                commit_response = any(
                    c.get("committed_at") and c["committed_at"] > review_time
                    for c in commits
                )

                if responded or commit_response:
                    response_counts += 1
                    # Compute latency to first response
                    response_times = []
                    for ac in author_comments:
                        if ac.get("submitted_at") and ac["submitted_at"] > review_time:
                            response_times.append(ac["submitted_at"])
                    for c in commits:
                        if c.get("committed_at") and c["committed_at"] > review_time:
                            response_times.append(c["committed_at"])
                    if response_times:
                        first_response = min(response_times)
                        latency_hours = (
                            first_response - review_time
                        ).total_seconds() / 3600.0
                        response_latencies.append(latency_hours)

            # CHANGES_REQUESTED followed by new commit
            changes_reviews = [
                r for r in other_reviews if r.get("state") == "CHANGES_REQUESTED"
            ]
            for cr in changes_reviews:
                changes_requested_counts += 1
                cr_time = cr.get("submitted_at")
                if cr_time and any(
                    c.get("committed_at") and c["committed_at"] > cr_time
                    for c in commits
                ):
                    followup_counts += 1

        # Abandoned PR detection: closed without any author activity after first commit
        state = pr.get("state", "")
        if state == "CLOSED" and pr.get("merged_at") is None:
            total_closed += 1
            # Check if author had any activity (reviews or commits) after initial submission
            pr_created = pr.get("created_at")
            if pr_created:
                has_activity = any(
                    r.get("submitted_at") and r["submitted_at"] > pr_created
                    and r.get("reviewer") == author
                    for r in reviews
                )
                if not has_activity:
                    # Check for additional commits beyond the first
                    post_creation_commits = [
                        c for c in commits
                        if c.get("committed_at") and c["committed_at"] > pr_created
                    ]
                    if len(post_creation_commits) <= 1:  # Only the initial commit
                        abandoned_count += 1

    review_response_rate = (
        response_counts / review_counts if review_counts > 0 else None
    )
    ci_failure_followup_rate = (
        followup_counts / changes_requested_counts
        if changes_requested_counts > 0
        else None
    )
    avg_response_latency = (
        float(np.median(response_latencies)) if response_latencies else None
    )
    abandoned_pr_rate = (
        abandoned_count / total_closed if total_closed > 0 else None
    )

    return EngagementFeatures(
        review_response_rate=review_response_rate,
        ci_failure_followup_rate=ci_failure_followup_rate,
        avg_response_latency_hours=avg_response_latency,
        abandoned_pr_rate=abandoned_pr_rate,
    )


# ----------------------------------------------------------
# H3: Cross-repo fingerprinting (TF-IDF variant)
# ----------------------------------------------------------


def compute_cross_repo_tfidf(
    prior_prs: list[dict[str, Any]],
) -> CrossRepoFeatures:
    """Compute cross-repo fingerprinting features using TF-IDF similarity.

    Only uses titles from prior PRs across different repos.
    """
    if len(prior_prs) < 2:
        return CrossRepoFeatures()

    titles = [pr.get("title", "") for pr in prior_prs]
    repos = [pr.get("repo", "") for pr in prior_prs]

    # Filter out empty titles
    valid = [(t, r) for t, r in zip(titles, repos, strict=True) if t.strip()]
    if len(valid) < 2:
        return CrossRepoFeatures()

    titles_clean = [t for t, _ in valid]
    repos_clean = [r for _, r in valid]

    # TF-IDF similarity
    try:
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            min_df=1,
        )
        tfidf_matrix = vectorizer.fit_transform(titles_clean)
        sim_matrix = cosine_similarity(tfidf_matrix)
    except ValueError:
        return CrossRepoFeatures()

    # Max pairwise similarity across different repos
    max_sim = 0.0
    duplicate_count = 0
    n = len(titles_clean)
    for i in range(n):
        for j in range(i + 1, n):
            if repos_clean[i] != repos_clean[j]:
                s = sim_matrix[i, j]
                max_sim = max(max_sim, s)
                if s > 0.9:
                    duplicate_count += 1

    # Language/repo entropy
    repo_counts = Counter(repos_clean)
    total = sum(repo_counts.values())
    probs = [c / total for c in repo_counts.values()]
    language_entropy = -sum(p * math.log2(p) for p in probs if p > 0)

    # Topic coherence: inverse of entropy weighted by count
    topic_coherence = (1.0 / (1.0 + language_entropy)) * len(valid)

    return CrossRepoFeatures(
        max_title_similarity=float(max_sim),
        language_entropy=float(language_entropy),
        topic_coherence=float(topic_coherence),
        duplicate_title_count=duplicate_count,
    )


# ----------------------------------------------------------
# GE score computation (v1 and v2)
# ----------------------------------------------------------

# Pre-load scorer configs once at module level (lazy init in _get_scorers)
_scorers: dict[str, TrustScorer] | None = None


def _get_scorers() -> dict[str, TrustScorer]:
    """Lazily initialize v1 and v2 TrustScorer instances."""
    global _scorers  # noqa: PLW0603
    if _scorers is None:
        config_v1 = load_config()
        config_v1.scoring_model = "v1"
        config_v2 = load_config()
        config_v2.scoring_model = "v2"
        _scorers = {
            "v1": TrustScorer(config_v1),
            "v2": TrustScorer(config_v2),
        }
    return _scorers


def compute_ge_scores(
    prior_merged_prs: list[dict[str, Any]],
    closed_pr_count: int,
    author_info: dict[str, Any] | None,
    author: str,
    context_repo: str,
    test_time: datetime,
) -> dict[str, float | None]:
    """Compute GE v1 and v2 scores from the author's prior merged PRs.

    Anti-lookahead: only prior_merged_prs already filtered to
    merged_at < T and repo != context_repo should be passed in.
    """
    scorers = _get_scorers()

    # Build UserProfile
    if author_info and pd.notna(author_info.get("account_created_at")):
        created_at = author_info["account_created_at"]
        if not isinstance(created_at, datetime):
            created_at = datetime(2020, 1, 1, tzinfo=UTC)
        elif created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
    else:
        # Estimate: use earliest PR creation time minus 30 days, or fallback
        if prior_merged_prs:
            dates = [
                pr["merged_at"] for pr in prior_merged_prs
                if pd.notna(pr.get("merged_at"))
            ]
            if dates:
                earliest = min(dates)
                if earliest.tzinfo is None:
                    earliest = earliest.replace(tzinfo=UTC)
                created_at = earliest - timedelta(days=30)
            else:
                created_at = datetime(2020, 1, 1, tzinfo=UTC)
        else:
            created_at = datetime(2020, 1, 1, tzinfo=UTC)

    followers = 0
    public_repos = 0
    if author_info:
        followers = author_info.get("followers", 0) or 0
        public_repos = author_info.get("public_repos", 0) or 0

    profile = UserProfile(
        login=author,
        created_at=created_at,
        followers_count=followers,
        public_repos_count=public_repos,
        is_bot=False,
    )

    # Build MergedPR list from prior merged PRs
    merged_prs: list[MergedPR] = []
    for pr in prior_merged_prs:
        ma = pr.get("merged_at")
        if not pd.notna(ma):
            continue
        if ma.tzinfo is None:
            ma = ma.replace(tzinfo=UTC)
        merged_prs.append(MergedPR(
            repo_name_with_owner=pr["repo"],
            title=pr.get("title", ""),
            merged_at=ma,
            additions=pr.get("additions", 0) or 0,
            deletions=pr.get("deletions", 0) or 0,
            changed_files=pr.get("files_changed", 0) or 0,
        ))

    user_data = UserContributionData(
        profile=profile,
        merged_prs=merged_prs,
        contributed_repos={},
        closed_pr_count=closed_pr_count,
    )

    results: dict[str, float | None] = {}
    for model_name, scorer in scorers.items():
        try:
            score = scorer.score(user_data, context_repo)
            results[f"ge_score_{model_name}"] = score.normalized_score
        except Exception:
            logger.debug(
                "GE %s scoring failed for %s on %s", model_name, author, context_repo,
                exc_info=True,
            )
            results[f"ge_score_{model_name}"] = None

    return results


# ----------------------------------------------------------
# Main extraction loop
# ----------------------------------------------------------


def extract_features_for_pr(
    db: BotDetectionDB,
    repo: str,
    number: int,
    author: str,
    created_at: datetime,
    outcome: PROutcome,
) -> FeatureRow:
    """Extract all signal features for a single PR.

    Anti-lookahead: queries only author's PRs on OTHER repos
    with created_at strictly before this PR's created_at.
    """
    # Get author's prior PRs on other repos
    prior_prs = db.get_author_prs_before(author, repo, created_at)

    # Get reviews and commits for prior PRs
    reviews_by_pr: dict[tuple[str, int], list[dict[str, Any]]] = {}
    commits_by_pr: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for pr in prior_prs:
        key = (pr["repo"], pr["number"])
        reviews_by_pr[key] = db.get_pr_reviews(pr["repo"], pr["number"])
        commits_by_pr[key] = db.get_pr_commits(pr["repo"], pr["number"])

    # H1: Burstiness
    burst = compute_burstiness(prior_prs, created_at)

    # H2: Engagement
    engagement = compute_engagement(prior_prs, reviews_by_pr, commits_by_pr, author)

    # H3: Cross-repo (TF-IDF)
    cross_repo = compute_cross_repo_tfidf(prior_prs)

    # GE scores (v1 and v2) with anti-lookahead
    prior_merged = [pr for pr in prior_prs if pd.notna(pr.get("merged_at"))]
    closed_count = db.get_closed_pr_count_before(author, repo, created_at)
    author_info = db.get_author_info(author)
    ge_scores = compute_ge_scores(
        prior_merged, closed_count, author_info, author, repo, created_at,
    )

    # Author metadata for baselines
    account_age_days: float | None = None
    followers_val: int | None = None
    public_repos_val: int | None = None
    if author_info:
        ca = author_info.get("account_created_at")
        if pd.notna(ca):
            age = (created_at - ca).total_seconds() / 86400.0
            account_age_days = max(0.0, age)
        f = author_info.get("followers")
        if f is not None and f > 0:
            followers_val = f
        r = author_info.get("public_repos")
        if r is not None and r > 0:
            public_repos_val = r

    return FeatureRow(
        repo=repo,
        number=number,
        author=author,
        outcome=outcome,
        created_at=created_at,
        # H1
        burst_count_1h=burst.burst_count_1h,
        burst_repos_1h=burst.burst_repos_1h,
        burst_count_24h=burst.burst_count_24h,
        burst_repos_24h=burst.burst_repos_24h,
        burst_max_rate=burst.burst_max_rate,
        # H2
        review_response_rate=engagement.review_response_rate,
        ci_failure_followup_rate=engagement.ci_failure_followup_rate,
        avg_response_latency_hours=engagement.avg_response_latency_hours,
        abandoned_pr_rate=engagement.abandoned_pr_rate,
        # H3
        max_title_similarity=cross_repo.max_title_similarity,
        language_entropy=cross_repo.language_entropy,
        topic_coherence=cross_repo.topic_coherence,
        duplicate_title_count=cross_repo.duplicate_title_count,
        # GE scores
        ge_score_v1=ge_scores.get("ge_score_v1"),
        ge_score_v2=ge_scores.get("ge_score_v2"),
        # Author metadata
        account_age_days=account_age_days,
        followers=followers_val,
        public_repos=public_repos_val,
    )


def run_stage2(
    base_dir: Path,
    config: StudyConfig,
) -> None:
    """Extract signal features for all PRs in the corpus.

    For each PR at time T:
    1. Query author's other PRs before T
    2. Compute burstiness (H1), engagement (H2), cross-repo (H3) features
    3. Save as Parquet
    """
    db_path = base_dir / config.paths.get("local_db", "data/bot_detection.duckdb")
    features_dir = base_dir / config.paths.get("features", "data/features")
    features_dir.mkdir(parents=True, exist_ok=True)

    with BotDetectionDB(db_path) as db:
        # Get all classified PRs
        all_prs = db.con.execute(
            "SELECT repo, number, author, created_at, outcome FROM prs WHERE outcome IS NOT NULL"
        ).fetchdf()

        logger.info("Extracting features for %d PRs...", len(all_prs))

        rows: list[dict[str, Any]] = []
        for i, (_, pr) in enumerate(all_prs.iterrows()):
            if i > 0 and i % 500 == 0:
                logger.info("  Progress: %d / %d PRs", i, len(all_prs))

            try:
                outcome = PROutcome(pr["outcome"])
            except ValueError:
                continue

            feature_row = extract_features_for_pr(
                db=db,
                repo=pr["repo"],
                number=pr["number"],
                author=pr["author"],
                created_at=pr["created_at"],
                outcome=outcome,
            )
            rows.append(feature_row.model_dump())

        if not rows:
            logger.warning("No features extracted!")
            return

        # Save as Parquet
        df = pd.DataFrame(rows)
        output_path = features_dir / "features.parquet"
        df.to_parquet(output_path, index=False)
        logger.info("Wrote %d feature rows to %s", len(df), output_path)

    # Checkpoint
    write_stage_checkpoint(
        db_path.parent,
        "stage2",
        {"feature_rows": len(rows)},
        details={
            "output_path": str(output_path),
            "columns": list(df.columns),
        },
    )
