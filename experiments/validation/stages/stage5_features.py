from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd

from experiments.validation.checkpoint import read_json
from experiments.validation.embedding import (
    author_repo_similarity,
    embed_texts,
)
from experiments.validation.models import StudyConfig

logger = logging.getLogger(__name__)


async def run_stage5(base_dir: Path, config: StudyConfig) -> None:
    """Stage 5: Feature engineering from scores + author data."""
    scored_dir = base_dir / config.paths.get("scored", "data/scored")
    authors_dir = base_dir / config.paths.get(
        "raw_authors", "data/raw/authors",
    )
    features_dir = base_dir / config.paths.get(
        "features", "data/features",
    )
    features_dir.mkdir(parents=True, exist_ok=True)

    output_path = features_dir / "features.parquet"

    # Load scored data
    scored_path = scored_dir / "full_model.parquet"
    if not scored_path.exists():
        logger.error("Scored data not found. Run Stage 4 first.")
        return

    df = pd.read_parquet(scored_path)
    logger.info("Loaded %d scored PRs", len(df))

    # Build author data cache
    author_data_cache: dict[str, dict] = {}
    for login in df["author_login"].unique():
        author_file = authors_dir / f"{login}.json"
        if author_file.exists():
            author_data_cache[login] = read_json(author_file)

    # --- Candidate features ---

    # Account age, followers, public repos
    log_account_age = []
    log_followers = []
    log_public_repos = []
    author_merge_rates = []
    author_open_counts = []
    author_pv_rates = []

    for _, row in df.iterrows():
        login = row["author_login"]
        author = author_data_cache.get(login, {})
        contrib = author.get("contribution_data", {})
        profile = contrib.get("profile", {})

        # Account age from profile created_at
        created_str = profile.get("created_at")
        if created_str:
            from datetime import datetime
            try:
                created = datetime.fromisoformat(str(created_str))
                pr_created = pd.Timestamp(row["created_at"])
                if pr_created.tzinfo is None:
                    pr_created = pr_created.tz_localize("UTC")
                age_days = (
                    pr_created - pd.Timestamp(created)
                ).total_seconds() / 86400
                log_account_age.append(math.log1p(max(0, age_days)))
            except (ValueError, TypeError):
                log_account_age.append(0.0)
        else:
            log_account_age.append(0.0)

        log_followers.append(
            math.log1p(profile.get("followers_count", 0)),
        )
        log_public_repos.append(
            math.log1p(profile.get("public_repos_count", 0)),
        )

        # Author merge rate (Tier 1)
        merged_ct = author.get("merged_count")
        closed_ct = author.get("closed_count")
        if merged_ct is not None and closed_ct is not None:
            total = merged_ct + closed_ct
            rate = merged_ct / total if total > 0 else None
            author_merge_rates.append(rate)
        else:
            author_merge_rates.append(None)

        author_open_counts.append(author.get("open_count"))

        # Author pocket veto rate (Tier 2)
        if author.get("tier2_sampled"):
            t_count = author.get("tier2_timeout_count", 0)
            e_count = author.get(
                "tier2_explicit_rejection_count", 0,
            )
            total_closed = t_count + e_count
            pv_rate = (
                t_count / total_closed if total_closed > 0 else None
            )
            author_pv_rates.append(pv_rate)
        else:
            author_pv_rates.append(None)

    df["log_account_age_days"] = log_account_age
    df["log_followers"] = log_followers
    df["log_public_repos"] = log_public_repos
    df["author_merge_rate"] = author_merge_rates
    df["author_open_pr_count"] = author_open_counts
    df["author_pocket_veto_rate"] = author_pv_rates

    # PR-level confound controls
    df["log_pr_size"] = df.apply(
        lambda r: math.log1p(
            r.get("additions", 0) + r.get("deletions", 0)
        )
        if "additions" in r.index
        else 0.0,
        axis=1,
    )
    if "changed_files" not in df.columns:
        df["pr_changed_files"] = 0
    else:
        df["pr_changed_files"] = df["changed_files"].fillna(0).astype(int)

    # --- Newcomer cohort (Rec 3) ---
    # A newcomer has 0 merged PRs before this test PR
    df["is_newcomer"] = (df["total_prs_at_time"] == 0).astype(int)
    # First-time contributor to THIS repo's ecosystem
    df["is_first_repo_ecosystem"] = (
        df["unique_repos_at_time"] <= 1
    ).astype(int)

    # --- Semantic similarity (H4) ---
    embedding_model = config.features.get(
        "embedding_model", "text-embedding-004",
    )
    embed_batch = config.features.get("embedding_batch_size", 50)

    try:
        # Get unique repos and their descriptions
        unique_repos = df["repo"].unique().tolist()
        repo_descriptions = []
        for repo in unique_repos:
            # Use repo name as description fallback
            repo_descriptions.append(repo.replace("/", " "))

        repo_embeddings = await embed_texts(
            repo_descriptions,
            model=embedding_model,
            batch_size=embed_batch,
        )
        repo_emb_map = dict(
            zip(unique_repos, repo_embeddings, strict=True),
        )

        # Compute per-author, per-repo similarity
        similarities = []
        for _, row in df.iterrows():
            login = row["author_login"]
            target_repo = row["repo"]
            author = author_data_cache.get(login, {})
            contrib = author.get("contribution_data", {})
            contrib_repos = contrib.get("contributed_repos", {})

            # Get embeddings for author's contributed repos
            author_repo_names = [
                r for r in contrib_repos if r != target_repo
            ]
            author_embs = [
                repo_emb_map.get(r)
                for r in author_repo_names
                if r in repo_emb_map
            ]
            author_embs = [e for e in author_embs if e is not None]

            target_emb = repo_emb_map.get(target_repo)
            if target_emb is not None and author_embs:
                sim = author_repo_similarity(author_embs, target_emb)
            else:
                sim = None

            similarities.append(sim)

        df["embedding_similarity"] = similarities

    except Exception:
        logger.exception("Embedding computation failed")
        df["embedding_similarity"] = None

    # Save features
    df.to_parquet(output_path, index=False)
    logger.info(
        "Stage 5 complete: wrote %d rows to %s",
        len(df), output_path,
    )
