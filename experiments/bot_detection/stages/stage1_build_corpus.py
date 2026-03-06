from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bot_detection.cache import BotDetectionDB
from experiments.bot_detection.checkpoint import write_stage_checkpoint
from experiments.bot_detection.models import PROutcome, StudyConfig

logger = logging.getLogger(__name__)


def _compute_stale_threshold(
    merged_ttms: list[float],
    floor_days: float = 30.0,
    cap_days: float = 180.0,
    percentile: float = 90.0,
) -> float:
    """Compute per-repo stale threshold from merged time-to-merge values.

    threshold = max(floor, min(P{percentile} of TTM, cap))
    """
    if not merged_ttms:
        return floor_days
    p = float(np.percentile(merged_ttms, percentile))
    return max(floor_days, min(p, cap_days))


def _classify_outcome(
    state: str,
    merged_at: object,
    closed_at: object,
    created_at: datetime,
    stale_threshold_days: float,
) -> PROutcome:
    """Classify a PR into merged/rejected/pocket_veto."""
    # NB: DuckDB via pandas returns NaT for NULL timestamps, not None.
    # pd.NaT is truthy for `is not None`, so use pd.notna() instead.
    if pd.notna(merged_at) or state == "MERGED":
        return PROutcome.MERGED

    if pd.notna(closed_at):
        ttc = (closed_at - created_at).total_seconds() / 86400.0
        if ttc < stale_threshold_days:
            return PROutcome.REJECTED
        return PROutcome.POCKET_VETO

    # Still open: check if it's past the stale threshold
    age_days = (datetime.now() - created_at).total_seconds() / 86400.0
    if age_days >= stale_threshold_days:
        return PROutcome.POCKET_VETO

    # Still open and within threshold - treat as pocket veto for simplicity
    # (these are very old PRs that were never closed)
    return PROutcome.POCKET_VETO


def _classify_all_prs(
    db: BotDetectionDB,
    config: StudyConfig,
) -> dict[str, int]:
    """Classify all PRs in the database by outcome.

    Uses per-repo stale thresholds computed from 2024H1 merged PRs.
    Returns outcome counts.
    """
    clf = config.classification
    floor_days = clf.get("stale_threshold_floor_days", 30.0)
    cap_days = clf.get("stale_threshold_cap_days", 180.0)
    pct = clf.get("stale_threshold_percentile", 90.0)

    repos = db.get_distinct_repos()
    outcome_counts: dict[str, int] = {"merged": 0, "rejected": 0, "pocket_veto": 0}

    for repo in repos:
        # Compute stale threshold from merged PRs in this repo
        merged_prs = db.get_repo_prs(repo, state="MERGED")
        merged_ttms = []
        for pr in merged_prs:
            ma = pr.get("merged_at")
            ca = pr.get("created_at")
            if pd.notna(ma) and pd.notna(ca):
                ttm = (ma - ca).total_seconds() / 86400.0
                if ttm > 0:
                    merged_ttms.append(ttm)

        threshold = _compute_stale_threshold(merged_ttms, floor_days, cap_days, pct)
        logger.debug(
            "Repo %s: %d merged PRs, stale threshold = %.1f days",
            repo, len(merged_ttms), threshold,
        )

        # Classify all PRs in this repo
        all_prs = db.get_repo_prs(repo)
        for pr in all_prs:
            outcome = _classify_outcome(
                state=pr.get("state", ""),
                merged_at=pr.get("merged_at"),
                closed_at=pr.get("closed_at"),
                created_at=pr["created_at"],
                stale_threshold_days=threshold,
            )
            db.update_pr_outcome(repo, pr["number"], outcome.value, threshold)
            outcome_counts[outcome.value] += 1

    return outcome_counts


def run_stage1(
    base_dir: Path,
    config: StudyConfig,
    scale: str = "full",
) -> None:
    """Build the PR corpus from cached data sources.

    Steps:
    1. Import neoteny DuckDB data (primary, then secondary if available)
    2. Import PR 27 JSONL data (gap-fill)
    3. Filter known bots
    4. Classify outcomes
    5. Write checkpoint
    """
    db_path = base_dir / config.paths.get("local_db", "data/bot_detection.duckdb")

    # Determine repo filter based on scale
    repo_filter: list[str] | None = None
    if scale != "full":
        scale_repos = config.scale.get(scale, [])
        if scale_repos:
            repo_filter = scale_repos
            logger.info("Scale=%s: filtering to %d repos", scale, len(repo_filter))

    sources = config.data_sources
    total_counts: dict[str, int] = {}

    with BotDetectionDB(db_path) as db:
        # Step 0: Import OSS parquet data (primary source)
        oss_parquet_path = Path(sources.get("oss_parquet", ""))
        if oss_parquet_path.exists():
            logger.info("Importing OSS parquet data: %s", oss_parquet_path)
            counts = db.import_oss_parquet(oss_parquet_path, repo_filter=repo_filter)
            for k, v in counts.items():
                total_counts[k] = total_counts.get(k, 0) + v
        else:
            logger.warning("OSS parquet data not found: %s", oss_parquet_path)

        # Step 1: Import neoteny primary (gap-fill)
        primary_path = Path(sources.get("neoteny_primary", ""))
        if primary_path.exists():
            logger.info("Importing neoteny primary: %s", primary_path)
            counts = db.import_neoteny_cache(primary_path, repo_filter=repo_filter)
            for k, v in counts.items():
                total_counts[k] = total_counts.get(k, 0) + v
        else:
            logger.warning("Neoteny primary not found: %s", primary_path)

        # Step 1b: Import neoteny secondary (may be locked)
        secondary_path = Path(sources.get("neoteny_secondary", ""))
        if secondary_path.exists():
            logger.info("Attempting neoteny secondary: %s", secondary_path)
            counts = db.import_neoteny_cache(secondary_path, repo_filter=repo_filter)
            for k, v in counts.items():
                total_counts[k] = total_counts.get(k, 0) + v

        # Step 2: Import PR 27 data
        pr27_path = Path(sources.get("pr27_data", ""))
        if not pr27_path.is_absolute():
            # Resolve relative to project root
            project_root = base_dir.parent.parent
            pr27_path = project_root / pr27_path
        if pr27_path.exists():
            logger.info("Importing PR 27 data: %s", pr27_path)
            counts = db.import_pr27_data(pr27_path, repo_filter=repo_filter)
            for k, v in counts.items():
                total_counts[k] = total_counts.get(k, 0) + v
        else:
            logger.warning("PR 27 data not found: %s", pr27_path)

        # Step 3: Filter bots
        bot_count = db.filter_bots(config.bot_patterns)
        total_counts["bots_filtered"] = bot_count

        # Step 4: Classify outcomes
        logger.info("Classifying PR outcomes...")
        outcome_counts = _classify_all_prs(db, config)
        total_counts.update({f"outcome_{k}": v for k, v in outcome_counts.items()})

        # Summary
        final_pr_count = db.get_pr_count()
        n_repos = len(db.get_distinct_repos())
        n_authors = len(db.get_distinct_authors())
        total_counts["final_prs"] = final_pr_count
        total_counts["repos"] = n_repos
        total_counts["authors"] = n_authors

        logger.info(
            "Stage 1 complete: %d PRs, %d repos, %d authors",
            final_pr_count, n_repos, n_authors,
        )

    # Step 5: Write checkpoint
    data_parent = db_path.parent
    write_stage_checkpoint(
        data_parent,
        "stage1",
        total_counts,
        details={
            "scale": scale,
            "repo_filter": repo_filter,
        },
    )
