from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import yaml

from experiments.bot_detection.models import StudyConfig

logger = logging.getLogger(__name__)


def _load_study_config(base_dir: Path) -> StudyConfig:
    """Load and parse study_config.yaml."""
    config_path = base_dir / "study_config.yaml"
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    return StudyConfig(
        data_sources=raw.get("data_sources", {}),
        classification=raw.get("classification", {}),
        burstiness_sweep=raw.get("burstiness_sweep", {}),
        analysis=raw.get("analysis", {}),
        scale=raw.get("scale", {}),
        paths=raw.get("paths", {}),
        bot_patterns=raw.get("bot_patterns", []),
    )


@click.group()
@click.option(
    "--base-dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path(__file__).parent,
    help="Base directory for experiment files.",
)
@click.option(
    "--scale",
    type=click.Choice(["micro", "small", "full"]),
    default="full",
    help="Scale of the run (micro=2 repos, small=10, full=all).",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable debug logging.",
)
@click.pass_context
def cli(ctx: click.Context, base_dir: Path, scale: str, verbose: bool) -> None:
    """Bot Detection Experiment Pipeline."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    ctx.ensure_object(dict)
    ctx.obj["base_dir"] = base_dir
    ctx.obj["scale"] = scale
    ctx.obj["config"] = _load_study_config(base_dir)


def _run_stage(stage_num: int, ctx_obj: dict) -> None:
    """Run a single pipeline stage."""
    base_dir = ctx_obj["base_dir"]
    config = ctx_obj["config"]
    scale = ctx_obj["scale"]

    if stage_num == 1:
        from experiments.bot_detection.stages.stage1_build_corpus import run_stage1
        run_stage1(base_dir, config, scale=scale)
    elif stage_num == 2:
        from experiments.bot_detection.stages.stage2_extract_signals import run_stage2
        run_stage2(base_dir, config)
    elif stage_num == 3:
        from experiments.bot_detection.stages.stage3_evaluate import run_stage3
        run_stage3(base_dir, config)
    elif stage_num == 4:
        from experiments.bot_detection.stages.stage4_baselines import run_stage4
        run_stage4(base_dir, config)
    else:
        logger.error("Unknown stage: %d", stage_num)
        sys.exit(1)


@cli.command("run-all")
@click.pass_context
def run_all(ctx: click.Context) -> None:
    """Run all pipeline stages (1-4) sequentially."""
    for stage_num in range(1, 5):
        logger.info("=== Running Stage %d ===", stage_num)
        _run_stage(stage_num, ctx.obj)
    logger.info("=== Pipeline complete ===")


@cli.command("run-stage")
@click.argument("stage", type=int)
@click.pass_context
def run_stage(ctx: click.Context, stage: int) -> None:
    """Run a specific pipeline stage (1-4)."""
    logger.info("=== Running Stage %d ===", stage)
    _run_stage(stage, ctx.obj)
    logger.info("=== Stage %d complete ===", stage)


if __name__ == "__main__":
    cli()
