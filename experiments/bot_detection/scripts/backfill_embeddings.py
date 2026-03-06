"""Backfill burst_title_embedding_sim into the features parquet.

For each PR with burst_count_24h >= 2, finds the 24h burst window titles
from the DB, embeds them via Gemini, computes mean pairwise cosine similarity,
and updates the parquet column.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from pathlib import Path

import pandas as pd

from experiments.bot_detection.cache import BotDetectionDB
from experiments.bot_detection.embedding import embed_texts
from experiments.bot_detection.stages.stage2_extract_signals import (
    compute_embedding_similarity,
)

logger = logging.getLogger(__name__)

BASE_DIR = Path("experiments/bot_detection")
DB_PATH = BASE_DIR / "data" / "bot_detection.duckdb"
FEATURES_PATH = BASE_DIR / "data" / "features" / "features.parquet"


async def backfill() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = pd.read_parquet(FEATURES_PATH)
    bursty_mask = df.burst_count_24h >= 2
    bursty_df = df[bursty_mask].copy()
    logger.info("Found %d PRs with burst_count_24h >= 2", len(bursty_df))

    if bursty_df.empty:
        logger.info("Nothing to backfill.")
        return

    db = BotDetectionDB(DB_PATH)

    # Collect all unique titles we need to embed
    all_titles: set[str] = set()
    pr_burst_titles: dict[int, list[str]] = {}  # df index -> list of titles

    for idx, row in bursty_df.iterrows():
        author = row["author"]
        repo = row["repo"]
        created_at = row["created_at"]

        # Get prior PRs in 24h window on OTHER repos (same logic as stage2)
        prior_prs = db.get_author_prs_before(author, repo, created_at)
        prs_24h = [
            pr for pr in prior_prs
            if pr.get("created_at")
            and pr["created_at"] >= created_at - timedelta(hours=24)
        ]

        titles = [pr.get("title", "") for pr in prs_24h if pr.get("title")]
        pr_burst_titles[idx] = titles
        all_titles.update(titles)

    logger.info("Unique titles to embed: %d", len(all_titles))

    if not all_titles:
        logger.info("No titles found in burst windows.")
        db.close()
        return

    # Embed all unique titles at once
    title_list = sorted(all_titles)
    logger.info("Embedding %d titles via Gemini...", len(title_list))
    embeddings = await embed_texts(title_list)
    title_to_emb = dict(zip(title_list, embeddings, strict=True))
    logger.info("Embedding complete. Dim=%d", len(embeddings[0]) if embeddings else 0)

    # Compute similarity for each bursty PR
    updated = 0
    for idx, titles in pr_burst_titles.items():
        if len(titles) < 2:
            continue
        embs = [title_to_emb[t] for t in titles if t in title_to_emb]
        sim = compute_embedding_similarity(embs)
        if sim is not None:
            df.at[idx, "burst_title_embedding_sim"] = sim
            updated += 1

    logger.info("Updated %d rows with embedding similarity", updated)

    # Save
    df.to_parquet(FEATURES_PATH, index=False)
    logger.info("Saved updated parquet to %s", FEATURES_PATH)

    # Summary stats
    non_null = df.burst_title_embedding_sim.notna().sum()
    logger.info("burst_title_embedding_sim non-null: %d / %d", non_null, len(df))
    if non_null > 0:
        vals = df.burst_title_embedding_sim.dropna()
        logger.info("  mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
                     vals.mean(), vals.std(), vals.min(), vals.max())

    db.close()


if __name__ == "__main__":
    asyncio.run(backfill())
