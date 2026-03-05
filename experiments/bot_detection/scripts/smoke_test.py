"""End-to-end smoke test on micro scale (2 repos).

Usage:
    uv run python -m experiments.bot_detection.scripts.smoke_test
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from experiments.bot_detection.pipeline import _load_study_config

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = _load_study_config(BASE_DIR)
    data_dir = BASE_DIR / "data"
    db_path = BASE_DIR / config.paths.get("local_db", "data/bot_detection.duckdb")
    features_dir = BASE_DIR / config.paths.get("features", "data/features")
    results_dir = BASE_DIR / config.paths.get("results", "data/results")

    # Clean up previous smoke test data
    if db_path.exists():
        db_path.unlink()
        logger.info("Removed existing %s", db_path)

    # Stage 1: Build corpus (micro scale)
    logger.info("=== Stage 1: Build Corpus (micro) ===")
    from experiments.bot_detection.stages.stage1_build_corpus import run_stage1
    run_stage1(BASE_DIR, config, scale="micro")

    # Verify Stage 1
    checkpoint_path = data_dir / "stage1_complete.json"
    if not checkpoint_path.exists():
        logger.error("Stage 1 checkpoint not found!")
        sys.exit(1)

    from experiments.bot_detection.checkpoint import read_json
    checkpoint = read_json(checkpoint_path)
    logger.info("Stage 1 checkpoint: %s", checkpoint.get("row_counts", {}))

    # Verify DB has data
    from experiments.bot_detection.cache import BotDetectionDB
    with BotDetectionDB(db_path) as db:
        pr_count = db.get_pr_count()
        repos = db.get_distinct_repos()
        logger.info("PRs: %d, Repos: %s", pr_count, repos)
        if pr_count == 0:
            logger.error("No PRs imported!")
            sys.exit(1)

    # Stage 2: Extract signals
    logger.info("=== Stage 2: Extract Signals ===")
    from experiments.bot_detection.stages.stage2_extract_signals import run_stage2
    run_stage2(BASE_DIR, config)

    # Verify Stage 2
    features_path = features_dir / "features.parquet"
    if not features_path.exists():
        logger.error("Features parquet not found!")
        sys.exit(1)

    import pandas as pd
    df = pd.read_parquet(features_path)
    logger.info("Features: %d rows, %d columns", len(df), len(df.columns))
    logger.info("Feature columns: %s", list(df.columns))

    # Check for NaN/Inf
    numeric_cols = df.select_dtypes(include=["number"]).columns
    inf_counts = (df[numeric_cols] == float("inf")).sum()
    if inf_counts.sum() > 0:
        logger.warning("Inf values found: %s", inf_counts[inf_counts > 0].to_dict())

    # Stage 3: Evaluate
    logger.info("=== Stage 3: Evaluate ===")
    from experiments.bot_detection.stages.stage3_evaluate import run_stage3
    run_stage3(BASE_DIR, config)

    # Stage 4: Baselines
    logger.info("=== Stage 4: Baselines ===")
    from experiments.bot_detection.stages.stage4_baselines import run_stage4
    run_stage4(BASE_DIR, config)

    # Final summary
    logger.info("=== Smoke Test Complete ===")
    results_json = results_dir / "statistical_tests.json"
    if results_json.exists():
        results = read_json(results_json)
        keys = list(results.keys()) if isinstance(results, dict) else "N/A"
        logger.info("Results keys: %s", keys)
    else:
        logger.warning("No results JSON found (may be expected for small datasets)")

    logger.info("All stages completed successfully.")


if __name__ == "__main__":
    main()
