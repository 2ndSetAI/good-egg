from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from experiments.validation.ablations import (
    get_ablation_variants,
    make_scorer,
)
from experiments.validation.checkpoint import (
    load_scored_pr_keys,
    read_json,
    read_jsonl,
)
from experiments.validation.models import ClassifiedPR, ScoredPR, StudyConfig
from good_egg.config import GoodEggConfig
from good_egg.models import UserContributionData
from good_egg.scorer import TrustScorer

logger = logging.getLogger(__name__)


def _load_author_data(
    authors_dir: Path, login: str,
) -> UserContributionData | None:
    """Load author's UserContributionData from persisted JSON."""
    path = authors_dir / f"{login}.json"
    if not path.exists():
        return None

    raw = read_json(path)
    contrib = raw.get("contribution_data")
    if not contrib:
        return None

    return UserContributionData(**contrib)


def _apply_anti_lookahead(
    user_data: UserContributionData,
    cutoff: datetime,
) -> UserContributionData:
    """Filter author's merged PRs to only those merged before cutoff.

    Also filters contributed_repos to only repos referenced by
    remaining PRs.
    """
    filtered_prs = [
        pr for pr in user_data.merged_prs
        if pr.merged_at < cutoff
    ]

    # Keep only repos referenced by remaining PRs
    remaining_repos = {
        pr.repo_name_with_owner for pr in filtered_prs
    }
    filtered_repos = {
        k: v for k, v in user_data.contributed_repos.items()
        if k in remaining_repos
    }

    return UserContributionData(
        profile=user_data.profile,
        merged_prs=filtered_prs,
        contributed_repos=filtered_repos,
    )


def _score_pr(
    pr: ClassifiedPR,
    user_data: UserContributionData,
    full_scorer: TrustScorer,
    ablation_scorers: dict[str, TrustScorer],
) -> ScoredPR | None:
    """Score a single PR with full model and all ablations."""
    # Apply anti-lookahead
    filtered_data = _apply_anti_lookahead(user_data, pr.created_at)

    # Score with full model
    try:
        full_score = full_scorer.score(filtered_data, pr.repo)
    except Exception:
        logger.warning(
            "Failed to score %s PR #%d for %s",
            pr.repo, pr.number, pr.author_login,
        )
        return None

    # Score with ablation variants
    ablation_scores: dict[str, float] = {}
    for name, scorer in ablation_scorers.items():
        try:
            abl_score = scorer.score(filtered_data, pr.repo)
            ablation_scores[name] = abl_score.normalized_score
        except Exception:
            logger.debug(
                "Ablation %s failed for %s PR #%d",
                name, pr.repo, pr.number,
            )
            ablation_scores[name] = 0.0

    return ScoredPR(
        repo=pr.repo,
        pr_number=pr.number,
        author_login=pr.author_login,
        outcome=pr.outcome,
        temporal_bin=pr.temporal_bin,
        created_at=pr.created_at,
        raw_score=full_score.raw_score,
        normalized_score=full_score.normalized_score,
        trust_level=full_score.trust_level.value,
        total_prs_at_time=len(filtered_data.merged_prs),
        unique_repos_at_time=len(
            {p.repo_name_with_owner for p in filtered_data.merged_prs}
        ),
        ablation_scores=ablation_scores,
    )


def _append_parquet(scored_dir: Path, rows: list[dict]) -> None:
    """Append scored rows to the parquet file."""
    df = pd.DataFrame(rows)
    output_path = scored_dir / "full_model.parquet"

    if output_path.exists():
        existing = pd.read_parquet(output_path)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_parquet(output_path, index=False)
    logger.info("Wrote %d total rows to %s", len(df), output_path)


def run_stage4(base_dir: Path, config: StudyConfig) -> None:
    """Stage 4: Temporal scoring with anti-lookahead + ablations."""
    classified_dir = base_dir / config.paths.get(
        "classified_prs", "data/raw/prs_classified",
    )
    authors_dir = base_dir / config.paths.get(
        "raw_authors", "data/raw/authors",
    )
    scored_dir = base_dir / config.paths.get("scored", "data/scored")
    scored_dir.mkdir(parents=True, exist_ok=True)

    batch_size = config.scoring.get("batch_size", 100)

    # Load already-scored PRs for checkpoint
    scored_keys = load_scored_pr_keys(
        scored_dir, "full_model.parquet",
    )
    logger.info("Checkpoint: %d PRs already scored", len(scored_keys))

    # Initialize scorers
    ge_config = GoodEggConfig()
    full_scorer = TrustScorer(ge_config)

    ablation_variants = get_ablation_variants(ge_config)
    ablation_scorers = {
        name: make_scorer(variant)
        for name, variant in ablation_variants.items()
    }

    # Cache loaded author data
    author_cache: dict[str, UserContributionData | None] = {}

    scored_rows: list[dict] = []

    # Process all classified PRs
    for jsonl_file in sorted(classified_dir.glob("*.jsonl")):
        records = read_jsonl(jsonl_file)

        for record in records:
            pr = ClassifiedPR(**record)
            key = f"{pr.repo}::{pr.number}"

            if key in scored_keys:
                continue

            # Load author data (with caching)
            if pr.author_login not in author_cache:
                author_cache[pr.author_login] = _load_author_data(
                    authors_dir, pr.author_login,
                )

            user_data = author_cache[pr.author_login]
            if user_data is None:
                continue

            scored = _score_pr(
                pr, user_data, full_scorer, ablation_scorers,
            )
            if scored is not None:
                scored_rows.append(scored.model_dump(mode="json"))

            # Batch write
            if len(scored_rows) >= batch_size:
                _append_parquet(scored_dir, scored_rows)
                scored_rows.clear()

    # Final batch
    if scored_rows:
        _append_parquet(scored_dir, scored_rows)

    logger.info("Stage 4 complete")
