from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import yaml

from experiments.validation.models import RepoEntry, StudyConfig, TemporalBin

logger = logging.getLogger(__name__)


def _load_study_config(base_dir: Path) -> StudyConfig:
    """Load and parse study_config.yaml."""
    config_path = base_dir / "study_config.yaml"
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    bins = [TemporalBin(**b) for b in raw.get("temporal_bins", [])]
    return StudyConfig(
        temporal_bins=bins,
        stale_threshold_bin=raw.get("stale_threshold_bin", "2024H1"),
        collection=raw.get("collection", {}),
        classification=raw.get("classification", {}),
        author_filtering=raw.get("author_filtering", {}),
        fetch=raw.get("fetch", {}),
        scoring=raw.get("scoring", {}),
        features=raw.get("features", {}),
        analysis=raw.get("analysis", {}),
        ablations=raw.get("ablations", {}),
        paths=raw.get("paths", {}),
    )


def _load_repo_list(base_dir: Path) -> list[RepoEntry]:
    """Load repo_list.yaml and return list of RepoEntry."""
    repo_path = base_dir / "repo_list.yaml"
    with open(repo_path) as f:
        raw = yaml.safe_load(f)

    repos = raw.get("repos", [])
    if not repos:
        logger.warning("No repos found in %s", repo_path)
        return []
    return [RepoEntry(**r) for r in repos]


@click.group()
@click.option(
    "--base-dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path(__file__).parent,
    help="Base directory for experiment files.",
)
@click.option(
    "--limit",
    type=int,
    default=0,
    help="Limit PRs per repo (0 = no limit, for smoke testing).",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable debug logging.",
)
@click.pass_context
def cli(ctx: click.Context, base_dir: Path, limit: int, verbose: bool) -> None:
    """GE Validation Study Pipeline."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    ctx.ensure_object(dict)
    ctx.obj["base_dir"] = base_dir
    ctx.obj["limit"] = limit
    ctx.obj["config"] = _load_study_config(base_dir)
    ctx.obj["repos"] = _load_repo_list(base_dir)


def _run_stage(stage_num: int, ctx_obj: dict) -> None:
    """Run a single pipeline stage."""
    base_dir = ctx_obj["base_dir"]
    config = ctx_obj["config"]
    repos = ctx_obj["repos"]
    limit = ctx_obj["limit"]

    if stage_num == 1:
        from experiments.validation.stages.stage1_collect_prs import (
            run_stage1,
        )
        run_stage1(base_dir, config, repos, limit=limit)
    elif stage_num == 2:
        from experiments.validation.stages.stage2_discover_authors import (
            run_stage2,
        )
        run_stage2(base_dir, config)
    elif stage_num == 3:
        import asyncio

        from experiments.validation.stages.stage3_fetch_authors import (
            run_stage3,
        )
        asyncio.run(run_stage3(base_dir, config))
    elif stage_num == 4:
        from experiments.validation.stages.stage4_score import run_stage4
        run_stage4(base_dir, config)
    elif stage_num == 5:
        import asyncio

        from experiments.validation.stages.stage5_features import run_stage5
        asyncio.run(run_stage5(base_dir, config))
    elif stage_num == 6:
        from experiments.validation.stages.stage6_analyze import run_stage6
        run_stage6(base_dir, config)
    else:
        logger.error("Unknown stage: %d", stage_num)
        sys.exit(1)


@cli.command("run-all")
@click.pass_context
def run_all(ctx: click.Context) -> None:
    """Run all pipeline stages (1-6) sequentially."""
    for stage_num in range(1, 7):
        logger.info("=== Running Stage %d ===", stage_num)
        _run_stage(stage_num, ctx.obj)
    logger.info("=== Pipeline complete ===")


@cli.command("run-stage")
@click.argument("stage", type=int)
@click.pass_context
def run_stage(ctx: click.Context, stage: int) -> None:
    """Run a specific pipeline stage (1-6)."""
    logger.info("=== Running Stage %d ===", stage)
    _run_stage(stage, ctx.obj)
    logger.info("=== Stage %d complete ===", stage)


if __name__ == "__main__":
    cli()
