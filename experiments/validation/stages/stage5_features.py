from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd

from experiments.validation.checkpoint import read_json, read_jsonl
from experiments.validation.embedding import (
    cosine_similarity,
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

    # --- Semantic similarity (H4): PR body x repo README ---
    embedding_model = config.features.get(
        "embedding_model", "gemini-embedding-001",
    )
    embed_batch = config.features.get("embedding_batch_size", 50)

    try:
        # 1. Load repo READMEs from disk
        repos_dir = base_dir / "data" / "raw" / "repos"
        readme_texts: dict[str, str] = {}
        for repo_name in df["repo"].unique():
            owner, name = repo_name.split("/", 1)
            readme_path = (
                repos_dir / f"{owner}__{name}_readme.md"
            )
            if readme_path.exists():
                text = readme_path.read_text(errors="replace").strip()
                if text:
                    readme_texts[repo_name] = text

        # 2. Load PR bodies from classified JSONL files
        classified_dir = base_dir / config.paths.get(
            "classified_prs", "data/raw/prs_classified",
        )
        pr_body_map: dict[tuple[str, int], str] = {}
        for repo_name in df["repo"].unique():
            owner, name = repo_name.split("/", 1)
            cpath = classified_dir / f"{owner}__{name}.jsonl"
            if cpath.exists():
                for rec in read_jsonl(cpath):
                    body = rec.get("body", "")
                    pr_body_map[(repo_name, rec["number"])] = body

        # 3. Build list of unique texts to embed
        unique_texts: dict[str, int] = {}
        text_list: list[str] = []

        def _register_text(text: str) -> int:
            if text in unique_texts:
                return unique_texts[text]
            idx = len(text_list)
            unique_texts[text] = idx
            text_list.append(text)
            return idx

        # Register README texts
        readme_idx: dict[str, int] = {}
        for repo_name, text in readme_texts.items():
            # Truncate long READMEs to avoid token limits
            truncated = text[:4000]
            readme_idx[repo_name] = _register_text(truncated)

        # Register PR body/title texts
        pr_text_idx: dict[tuple[str, int], int] = {}
        for _, row in df.iterrows():
            key = (row["repo"], row["pr_number"])
            body = pr_body_map.get(key, "")
            text = body.strip() if body else ""
            if not text:
                # Fallback to PR title
                text = str(row.get("title", ""))
            if text:
                truncated = text[:2000]
                pr_text_idx[key] = _register_text(truncated)

        # 4. Batch embed all unique texts
        if text_list:
            all_embeddings = await embed_texts(
                text_list,
                model=embedding_model,
                batch_size=embed_batch,
            )
        else:
            all_embeddings = []

        # 5. Compute cosine similarity per row
        similarities = []
        for _, row in df.iterrows():
            repo_name = row["repo"]
            pr_key = (repo_name, row["pr_number"])

            r_idx = readme_idx.get(repo_name)
            p_idx = pr_text_idx.get(pr_key)

            if r_idx is not None and p_idx is not None:
                sim = cosine_similarity(
                    all_embeddings[r_idx],
                    all_embeddings[p_idx],
                )
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
