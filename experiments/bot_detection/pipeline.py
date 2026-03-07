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
        author_analysis=raw.get("author_analysis", {}),
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


_STAGE6_SUBSTAGES = {
    "6a": "time_series",
    "6b": "llm_content",
    "6c": "semi_supervised",
    "6d": "title_analysis",
}


def _run_stage(stage_num: int, ctx_obj: dict, substage: str | None = None) -> None:
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
    elif stage_num == 5:
        from experiments.bot_detection.stages.stage5_author_features import run_stage5
        run_stage5(base_dir, config)
    elif stage_num == 6:
        _run_stage6(base_dir, config, substage)
    elif stage_num == 7:
        from experiments.bot_detection.stages.stage7_author_evaluate import run_stage7
        run_stage7(base_dir, config)
    elif stage_num == 8:
        from experiments.bot_detection.stages.stage8_campaigns import run_stage8
        run_stage8(base_dir, config)
    else:
        logger.error("Unknown stage: %d", stage_num)
        sys.exit(1)


def _run_stage6(base_dir: Path, config: StudyConfig, substage: str | None = None) -> None:
    """Run stage 6 sub-stages.

    Sub-stages:
        6a (time_series) -- always runs, cheap
        6b (llm_content) -- Gemini-based, opt-in
        6c (semi_supervised) -- k-NN + Isolation Forest, opt-in
        6d (title_analysis) -- TF-IDF title features, opt-in

    If substage is None, only 6a runs (default for run-all / run-author-pipeline).
    """
    if substage is None or substage == "6a":
        from experiments.bot_detection.stages.stage6_time_series import run_stage6_time_series
        run_stage6_time_series(base_dir, config)
    if substage == "6b":
        from experiments.bot_detection.stages.stage6_llm_content import run_stage6_llm_content
        run_stage6_llm_content(base_dir, config)
    elif substage == "6c":
        from experiments.bot_detection.stages.stage6_semi_supervised import (
            run_stage6_semi_supervised,
        )
        run_stage6_semi_supervised(base_dir, config)
    elif substage == "6d":
        from experiments.bot_detection.stages.stage6_title_analysis import (
            run_stage6_title_analysis,
        )
        run_stage6_title_analysis(base_dir, config)


@cli.command("run-all")
@click.pass_context
def run_all(ctx: click.Context) -> None:
    """Run all pipeline stages (1-8) sequentially."""
    for stage_num in range(1, 9):
        logger.info("=== Running Stage %d ===", stage_num)
        _run_stage(stage_num, ctx.obj)
    logger.info("=== Pipeline complete ===")


@cli.command("run-stage")
@click.argument("stage", type=str)
@click.pass_context
def run_stage(ctx: click.Context, stage: str) -> None:
    """Run a specific pipeline stage.

    STAGE can be an integer (1-8) or a stage-6 sub-stage like "6a", "6b", "6c", "6d".

    Sub-stages:
        6a  Time-series features (H9) -- default when running stage 6
        6b  LLM content analysis (H11) -- requires Gemini API
        6c  Semi-supervised k-NN + Isolation Forest (H13)
        6d  TF-IDF title analysis (H11 alternative) -- local, deterministic
    """
    substage = None
    if stage in _STAGE6_SUBSTAGES:
        substage = stage
        stage_num = 6
    else:
        try:
            stage_num = int(stage)
        except ValueError:
            logger.error("Unknown stage: %s", stage)
            sys.exit(1)

    logger.info("=== Running Stage %s ===", stage)
    _run_stage(stage_num, ctx.obj, substage=substage)
    logger.info("=== Stage %s complete ===", stage)


@cli.command("run-author-pipeline")
@click.pass_context
def run_author_pipeline(ctx: click.Context) -> None:
    """Run author-level pipeline stages (5-8) sequentially."""
    for stage_num in range(5, 9):
        logger.info("=== Running Stage %d ===", stage_num)
        _run_stage(stage_num, ctx.obj)
    logger.info("=== Author pipeline complete ===")


@cli.command("run-temporal-holdout")
@click.option(
    "--cutoffs",
    type=str,
    default=None,
    help="Comma-separated cutoff dates (YYYY-MM-DD) to override config.",
)
@click.pass_context
def run_temporal_holdout(ctx: click.Context, cutoffs: str | None) -> None:
    """Run temporal holdout experiment (Iteration 6)."""
    from experiments.bot_detection.stages.stage9_temporal_holdout import (
        run_temporal_holdout as _run_holdout,
    )

    config = ctx.obj["config"]

    # Allow CLI override of cutoff dates
    if cutoffs:
        if "temporal_holdout" not in config.author_analysis:
            config.author_analysis["temporal_holdout"] = {}
        config.author_analysis["temporal_holdout"]["cutoffs"] = cutoffs.split(",")

    _run_holdout(ctx.obj["base_dir"], config)


@cli.command("run-merge-rate-experiment")
@click.option(
    "--cutoffs",
    type=str,
    default=None,
    help="Comma-separated cutoff dates (YYYY-MM-DD) to override config.",
)
@click.pass_context
def run_merge_rate_experiment(ctx: click.Context, cutoffs: str | None) -> None:
    """Run merge rate non-monotonicity experiment (Iteration 7)."""
    from experiments.bot_detection.stages.stage10_merge_rate_models import (
        run_stage10,
    )

    config = ctx.obj["config"]
    cutoff_list = cutoffs.split(",") if cutoffs else None
    run_stage10(ctx.obj["base_dir"], config, cutoffs=cutoff_list)


@cli.command("run-two-model-pipeline")
@click.option(
    "--cutoffs",
    type=str,
    default=None,
    help="Comma-separated cutoff dates (YYYY-MM-DD) to override config.",
)
@click.pass_context
def run_two_model_pipeline(ctx: click.Context, cutoffs: str | None) -> None:
    """Run two-model pipeline experiment (Iteration 8)."""
    from experiments.bot_detection.stages.stage11_two_model_pipeline import (
        run_stage11,
    )

    config = ctx.obj["config"]
    cutoff_list = cutoffs.split(",") if cutoffs else None
    run_stage11(ctx.obj["base_dir"], config, cutoffs=cutoff_list)


@cli.command("run-knn-holdout")
@click.option(
    "--cutoffs",
    type=str,
    default=None,
    help="Comma-separated cutoff dates (YYYY-MM-DD) to override config.",
)
@click.pass_context
def run_knn_holdout(ctx: click.Context, cutoffs: str | None) -> None:
    """Run k-NN holdout experiment (Iteration 9)."""
    from experiments.bot_detection.stages.stage12_knn_holdout import run_stage12

    config = ctx.obj["config"]
    cutoff_list = cutoffs.split(",") if cutoffs else None
    run_stage12(ctx.obj["base_dir"], config, cutoffs=cutoff_list)


@cli.command("run-merge-prediction")
@click.option(
    "--cutoffs",
    type=str,
    default=None,
    help="Comma-separated cutoff dates (YYYY-MM-DD) to override config.",
)
@click.pass_context
def run_merge_prediction(ctx: click.Context, cutoffs: str | None) -> None:
    """Run merge prediction experiment (Iteration 10, Experiments A/C/D)."""
    from experiments.bot_detection.stages.stage13_merge_prediction import run_stage13

    config = ctx.obj["config"]
    cutoff_list = cutoffs.split(",") if cutoffs else None
    run_stage13(ctx.obj["base_dir"], config, cutoffs=cutoff_list)


@cli.command("run-advisory-score")
@click.option(
    "--cutoffs",
    type=str,
    default=None,
    help="Comma-separated cutoff dates (YYYY-MM-DD) to override config.",
)
@click.pass_context
def run_advisory_score(ctx: click.Context, cutoffs: str | None) -> None:
    """Run advisory suspension score experiment (Iteration 10, Experiment B)."""
    from experiments.bot_detection.stages.stage14_advisory_score import run_stage14

    config = ctx.obj["config"]
    cutoff_list = cutoffs.split(",") if cutoffs else None
    run_stage14(ctx.obj["base_dir"], config, cutoffs=cutoff_list)


@cli.command("run-feature-ablation")
@click.option(
    "--cutoffs",
    type=str,
    default=None,
    help="Comma-separated cutoff dates (YYYY-MM-DD) to override config.",
)
@click.pass_context
def run_feature_ablation(ctx: click.Context, cutoffs: str | None) -> None:
    """Run Bad Egg feature ablation experiment (Iteration 11)."""
    from experiments.bot_detection.stages.stage15_feature_ablation import (
        run_stage15,
    )

    config = ctx.obj["config"]
    cutoff_list = cutoffs.split(",") if cutoffs else None
    run_stage15(ctx.obj["base_dir"], config, cutoffs=cutoff_list)


@cli.command("run-hub-score-repo")
@click.option(
    "--cutoffs",
    type=str,
    default=None,
    help="Comma-separated cutoff dates (YYYY-MM-DD) to override config.",
)
@click.pass_context
def run_hub_score_repo(ctx: click.Context, cutoffs: str | None) -> None:
    """Run EBE hub_score repo-specific experiment (Iteration 11)."""
    from experiments.bot_detection.stages.stage16_hub_score_repo import (
        run_stage16,
    )

    config = ctx.obj["config"]
    cutoff_list = cutoffs.split(",") if cutoffs else None
    run_stage16(ctx.obj["base_dir"], config, cutoffs=cutoff_list)


@cli.command("run-hub-score-unknown")
@click.option(
    "--cutoffs",
    type=str,
    default=None,
    help="Comma-separated cutoff dates (YYYY-MM-DD) to override config.",
)
@click.pass_context
def run_hub_score_unknown(ctx: click.Context, cutoffs: str | None) -> None:
    """Run hub_score experiment on unknown contributors (Iteration 11b)."""
    from experiments.bot_detection.stages.stage17_hub_score_unknown import (
        run_stage17,
    )

    config = ctx.obj["config"]
    cutoff_list = cutoffs.split(",") if cutoffs else None
    run_stage17(ctx.obj["base_dir"], config, cutoffs=cutoff_list)


@cli.command("run-recency-window")
@click.option(
    "--cutoffs",
    type=str,
    default=None,
    help="Comma-separated cutoff dates (YYYY-MM-DD) to override config.",
)
@click.pass_context
def run_recency_window(ctx: click.Context, cutoffs: str | None) -> None:
    """Run recency window experiment on unknown contributors (Iteration 12)."""
    from experiments.bot_detection.stages.stage18_recency_window import (
        run_stage18,
    )

    config = ctx.obj["config"]
    cutoff_list = cutoffs.split(",") if cutoffs else None
    run_stage18(ctx.obj["base_dir"], config, cutoffs=cutoff_list)


if __name__ == "__main__":
    cli()
