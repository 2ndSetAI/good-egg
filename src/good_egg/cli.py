"""Click-based CLI for Good Egg trust scoring."""

from __future__ import annotations

import asyncio
import sys

import click

from good_egg.cache import Cache
from good_egg.config import load_config
from good_egg.formatter import format_cli_output, format_json
from good_egg.scorer import score_pr_author


@click.group()
@click.version_option(package_name="good-egg")
def main() -> None:
    """Good Egg - GitHub contributor trust scoring."""


@main.command()
@click.argument("username")
@click.option("--repo", required=True, help="Context repository (owner/name)")
@click.option("--token", envvar="GITHUB_TOKEN", help="GitHub token")
@click.option("--config", "config_path", default=None, help="Config file path")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option(
    "--scoring-model",
    type=click.Choice(["v1", "v2"]),
    default=None,
    help="Scoring model (v1 or v2)",
)
@click.option(
    "--force-score",
    is_flag=True,
    default=False,
    help="Force full scoring even for known contributors",
)
def score(
    username: str,
    repo: str,
    token: str | None,
    config_path: str | None,
    verbose: bool,
    output_json: bool,
    scoring_model: str | None,
    force_score: bool,
) -> None:
    """Score a GitHub user's trustworthiness relative to a repository."""
    if not token:
        click.echo("Error: GitHub token required. Set GITHUB_TOKEN or use --token.", err=True)
        sys.exit(1)

    parts = repo.split("/")
    if len(parts) != 2:
        click.echo("Error: --repo must be in owner/name format.", err=True)
        sys.exit(1)

    repo_owner, repo_name = parts
    config = load_config(config_path)
    if scoring_model is not None:
        config = config.model_copy(update={"scoring_model": scoring_model})
    if force_score:
        config = config.model_copy(update={"skip_known_contributors": False})
    cache = Cache(ttls=config.cache_ttl.to_seconds())

    result = asyncio.run(
        score_pr_author(
            login=username,
            repo_owner=repo_owner,
            repo_name=repo_name,
            config=config,
            token=token,
            cache=cache,
        )
    )
    cache.close()

    if output_json:
        click.echo(format_json(result))
    else:
        click.echo(format_cli_output(result, verbose=verbose))


@main.command("cache-stats")
def cache_stats() -> None:
    """Show cache statistics."""
    cache = Cache()
    stats = cache.stats()
    click.echo(f"Total entries: {stats['total_entries']}")
    click.echo(f"Active entries: {stats['active_entries']}")
    click.echo(f"Expired entries: {stats['expired_entries']}")
    click.echo(f"Database size: {stats['db_size_bytes']:,} bytes")
    if stats["categories"]:
        click.echo("\nBy category:")
        for cat, count in stats["categories"].items():
            click.echo(f"  {cat}: {count}")
    cache.close()


@main.command("cache-clear")
@click.option("--category", default=None, help="Clear specific category only")
def cache_clear(category: str | None) -> None:
    """Clear the cache."""
    cache = Cache()
    if category:
        cache.invalidate_category(category)
        click.echo(f"Cleared cache category: {category}")
    else:
        removed = cache.cleanup_expired()
        click.echo(f"Removed {removed} expired entries")
    cache.close()
