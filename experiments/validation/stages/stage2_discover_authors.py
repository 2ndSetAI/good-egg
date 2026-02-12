from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import median

from experiments.validation.checkpoint import (
    append_jsonl,
    read_jsonl,
    write_json,
)
from experiments.validation.models import (
    ClassifiedPR,
    CollectedPR,
    PROutcome,
    StudyConfig,
)

logger = logging.getLogger(__name__)

# GE bot patterns (mirrored from good_egg.github_client)
_BOT_SUFFIX_RE = re.compile(r"(\[bot\]|-bot|_bot|-app)$", re.IGNORECASE)
_BOT_PREFIX_RE = re.compile(
    r"^(dependabot|renovate|greenkeeper|snyk-|codecov|stale"
    r"|mergify|allcontributors|github-actions|pre-commit-ci)",
    re.IGNORECASE,
)


def _is_bot(login: str, extra_patterns: list[str]) -> bool:
    """Check if a login is a bot."""
    if _BOT_SUFFIX_RE.search(login) or _BOT_PREFIX_RE.search(login):
        return True
    return any(
        re.search(pattern, login, re.IGNORECASE)
        for pattern in extra_patterns
    )


# Known merge bot usernames and label patterns
_MERGE_BOT_CLOSERS = frozenset({
    "bors", "bors[bot]", "mergify[bot]", "mergify",
    "homu", "kodiak[bot]", "kodiak",
    "dependabot-preview[bot]",
})

_MERGE_BOT_LABELS = re.compile(
    r"(merged|auto-merge|automerge|bors|mergify)",
    re.IGNORECASE,
)


def _is_merge_bot_close(pr: CollectedPR) -> bool:
    """Detect if a closed (non-merged) PR was actually merged by a bot.

    Checks PR labels for merge bot indicators (e.g. "merged", "auto-merge")
    and known merge bot closer usernames.
    """
    if any(_MERGE_BOT_LABELS.search(label) for label in pr.labels):
        return True
    # Also check if the PR author is a known merge bot
    return pr.author_login.lower() in _MERGE_BOT_CLOSERS


def _is_bursty(
    author_prs: list[CollectedPR],
    window_hours: float = 1.0,
    min_repos: int = 3,
    min_prs: int = 5,
) -> bool:
    """Detect agentic spam via burst detection.

    Returns True if the author opened >= min_prs PRs across >= min_repos
    distinct repos within any sliding window of window_hours hours.
    """
    if len(author_prs) < min_prs:
        return False

    # Sort by creation time
    sorted_prs = sorted(author_prs, key=lambda p: p.created_at)
    window = timedelta(hours=window_hours)

    # Sliding window approach
    for i in range(len(sorted_prs)):
        window_end = sorted_prs[i].created_at + window
        # Collect PRs within the window
        window_prs = [
            p for p in sorted_prs[i:]
            if p.created_at <= window_end
        ]
        if len(window_prs) >= min_prs:
            repos_in_window = {p.repo for p in window_prs}
            if len(repos_in_window) >= min_repos:
                return True
    return False


def _compute_stale_threshold(
    merged_prs: list[CollectedPR],
    floor_days: int = 30,
    cap_days: int = 180,
    multiplier: int = 5,
) -> float:
    """Compute stale threshold from merged PRs.

    threshold = max(floor, min(multiplier * median_TTM, cap))
    """
    ttm_values: list[float] = []
    for pr in merged_prs:
        if pr.merged_at and pr.created_at:
            ttm = (pr.merged_at - pr.created_at).total_seconds() / 86400
            if ttm > 0:
                ttm_values.append(ttm)

    if not ttm_values:
        return float(floor_days)

    median_ttm = median(ttm_values)
    threshold = max(floor_days, min(multiplier * median_ttm, cap_days))
    return threshold


def _classify_pr(
    pr: CollectedPR,
    stale_threshold_days: float,
    buffer_days: int = 30,
    now: datetime | None = None,
) -> ClassifiedPR | None:
    """Classify a PR into outcome categories.

    Returns None for indeterminate PRs (open, within threshold).
    """
    if now is None:
        now = datetime.now(UTC)

    threshold = timedelta(days=stale_threshold_days)
    buffer = timedelta(days=buffer_days)

    # Merged PRs
    if pr.merged_at is not None:
        return ClassifiedPR(
            **pr.model_dump(),
            outcome=PROutcome.MERGED,
            stale_threshold_days=stale_threshold_days,
        )

    # Closed (non-merged) PRs
    if pr.closed_at is not None:
        time_open = pr.closed_at - pr.created_at
        outcome = (
            PROutcome.REJECTED if time_open <= threshold
            else PROutcome.POCKET_VETO
        )
        return ClassifiedPR(
            **pr.model_dump(),
            outcome=outcome,
            stale_threshold_days=stale_threshold_days,
        )

    # Still open PRs
    time_open = now - pr.created_at
    if time_open > threshold + buffer:
        # Open past threshold + buffer -> pocket veto
        return ClassifiedPR(
            **pr.model_dump(),
            outcome=PROutcome.POCKET_VETO,
            stale_threshold_days=stale_threshold_days,
        )

    # Indeterminate (still open, within threshold)
    return None


def run_stage2(base_dir: Path, config: StudyConfig) -> None:
    """Stage 2: Classify PRs and discover unique authors."""
    raw_dir = base_dir / config.paths.get("raw_prs", "data/raw/prs")
    classified_dir = base_dir / config.paths.get(
        "classified_prs", "data/raw/prs_classified",
    )
    authors_dir = base_dir / config.paths.get(
        "raw_authors", "data/raw/authors",
    )
    classified_dir.mkdir(parents=True, exist_ok=True)
    authors_dir.mkdir(parents=True, exist_ok=True)

    stale_cfg = config.classification
    floor_days = stale_cfg.get("stale_threshold_floor_days", 30)
    cap_days = stale_cfg.get("stale_threshold_cap_days", 180)
    multiplier = stale_cfg.get("stale_threshold_multiplier", 5)
    buffer_days = stale_cfg.get("pocket_veto_buffer_days", 30)

    extra_bot_patterns = config.author_filtering.get(
        "extra_bot_patterns", [],
    )

    # Burstiness detection config
    burst_window_hours = config.author_filtering.get(
        "burst_window_hours", 1.0,
    )
    burst_min_repos = config.author_filtering.get(
        "burst_min_repos", 3,
    )
    burst_min_prs = config.author_filtering.get(
        "burst_min_prs", 5,
    )

    stale_bin = config.stale_threshold_bin

    # Pre-compute bursty authors across all repos
    all_collected_prs: list[CollectedPR] = []
    for jsonl_file in sorted(raw_dir.glob("*.jsonl")):
        records = read_jsonl(jsonl_file)
        all_collected_prs.extend(CollectedPR(**r) for r in records)

    # Group by author for burstiness check
    prs_by_author: dict[str, list[CollectedPR]] = defaultdict(list)
    for pr in all_collected_prs:
        if pr.author_login:
            prs_by_author[pr.author_login].append(pr)

    bursty_authors: set[str] = set()
    for login, author_prs in prs_by_author.items():
        if _is_bursty(
            author_prs,
            window_hours=burst_window_hours,
            min_repos=burst_min_repos,
            min_prs=burst_min_prs,
        ):
            bursty_authors.add(login)
            logger.info("Bursty author detected: %s", login)

    if bursty_authors:
        logger.info(
            "Excluded %d bursty authors", len(bursty_authors),
        )

    all_authors: set[str] = set()
    total_classified = 0
    total_excluded = 0
    total_reclassified = 0

    for jsonl_file in sorted(raw_dir.glob("*.jsonl")):
        repo_key = jsonl_file.stem  # owner__repo
        records = read_jsonl(jsonl_file)

        # Parse all PRs
        prs = [CollectedPR(**r) for r in records]

        # Compute stale threshold from merged PRs in baseline bin
        baseline_merged = [
            p for p in prs
            if p.merged_at is not None and p.temporal_bin == stale_bin
        ]
        stale_threshold = _compute_stale_threshold(
            baseline_merged, floor_days, cap_days, multiplier,
        )
        logger.info(
            "%s: stale_threshold=%.1f days (from %d merged PRs)",
            repo_key, stale_threshold, len(baseline_merged),
        )

        # Classify each PR
        classified_records: list[dict] = []
        for pr in prs:
            # Skip non-merged PRs closed < 1 day (spam filter).
            # Merged PRs that close quickly are legitimate fast merges
            # and should NOT be excluded.
            if pr.merged_at is None and pr.created_at:
                end_time = pr.closed_at
                if end_time:
                    duration = (end_time - pr.created_at).total_seconds()
                    if duration < 86400:
                        total_excluded += 1
                        continue

            result = _classify_pr(
                pr, stale_threshold, buffer_days,
            )
            if result is None:
                total_excluded += 1
                continue

            # Rec 4: Detect merge-bot false negatives
            if (
                result.outcome != PROutcome.MERGED
                and _is_merge_bot_close(pr)
            ):
                result = ClassifiedPR(
                    **{
                        **pr.model_dump(),
                        "outcome": PROutcome.MERGED,
                        "stale_threshold_days": stale_threshold,
                    },
                )
                total_reclassified += 1

            # Skip bots and bursty authors
            if _is_bot(result.author_login, extra_bot_patterns):
                continue
            if result.author_login in bursty_authors:
                total_excluded += 1
                continue

            # Skip empty author logins
            if not result.author_login:
                continue

            # Skip PRs to self-owned repos (DOE Section 5)
            repo_owner = result.repo.split("/")[0].lower()
            if repo_owner == result.author_login.lower():
                total_excluded += 1
                continue

            classified_records.append(
                result.model_dump(mode="json"),
            )
            all_authors.add(result.author_login)

        # Write classified output
        output_path = classified_dir / f"{repo_key}.jsonl"
        if classified_records:
            append_jsonl(output_path, classified_records)
            total_classified += len(classified_records)

    # Write unique authors
    authors_list = sorted(all_authors)
    write_json(
        authors_dir / "unique_authors.json",
        {"authors": authors_list, "count": len(authors_list)},
    )

    logger.info(
        "Stage 2 complete: %d classified PRs, %d excluded, "
        "%d reclassified (merge-bot), %d unique authors",
        total_classified, total_excluded, total_reclassified,
        len(authors_list),
    )
