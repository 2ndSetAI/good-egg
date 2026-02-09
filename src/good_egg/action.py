"""GitHub Action entry point for Good Egg."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys

from good_egg.cache import Cache
from good_egg.config import load_config
from good_egg.exceptions import GoodEggError, RateLimitExhaustedError, UserNotFoundError
from good_egg.formatter import format_check_run_summary, format_markdown_comment
from good_egg.github_client import GitHubClient
from good_egg.models import TrustLevel
from good_egg.scorer import TrustScorer


async def run_action() -> None:
    """Main action logic."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(name)s: %(message)s",
    )
    logger = logging.getLogger("good_egg.action")

    # Read environment
    token = os.environ.get("GITHUB_TOKEN", "")
    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    repository = os.environ.get("GITHUB_REPOSITORY", "")

    # Read action inputs (GitHub Actions sets INPUT_ env vars)
    config_path = os.environ.get("INPUT_CONFIG-PATH") or os.environ.get("INPUT_CONFIG_PATH")
    if config_path and not os.path.exists(config_path):
        logger.warning("Config file %s not found, using defaults", config_path)
        config_path = None
    should_comment = os.environ.get("INPUT_COMMENT", "true").lower() == "true"
    should_check_run = os.environ.get("INPUT_CHECK-RUN", "false").lower() == "true"
    fail_on_low = os.environ.get("INPUT_FAIL-ON-LOW", "false").lower() == "true"

    if not token:
        print("::error::GITHUB_TOKEN is required")
        sys.exit(1)

    if not event_path or not os.path.exists(event_path):
        print("::error::GITHUB_EVENT_PATH is not set or file does not exist")
        sys.exit(1)

    # Parse event
    with open(event_path) as f:
        event = json.load(f)

    pr_data = event.get("pull_request", {})
    pr_number = pr_data.get("number") or event.get("number")
    pr_author = pr_data.get("user", {}).get("login", "")
    head_sha = pr_data.get("head", {}).get("sha", "")

    if not pr_number or not pr_author:
        print("::error::Could not extract PR number or author from event")
        sys.exit(1)

    repo_parts = repository.split("/")
    if len(repo_parts) != 2:
        print(f"::error::Invalid GITHUB_REPOSITORY: {repository}")
        sys.exit(1)

    repo_owner, repo_name = repo_parts

    # Load config and score
    config = load_config(config_path)
    cache = Cache(ttls=config.cache_ttl.to_seconds())

    async with GitHubClient(token=token, config=config, cache=cache) as client:
        user_data = await client.get_user_contribution_data(
            pr_author, context_repo=repository
        )

        scorer = TrustScorer(config)
        score = scorer.score(user_data, repository)

        # Post/update PR comment
        if should_comment:
            comment_body = format_markdown_comment(score)
            existing_comment_id = await client.find_existing_comment(
                repo_owner, repo_name, pr_number
            )
            if existing_comment_id:
                await client.update_pr_comment(
                    repo_owner, repo_name, existing_comment_id, comment_body
                )
            else:
                await client.post_pr_comment(
                    repo_owner, repo_name, pr_number, comment_body
                )

        # Create check run
        if should_check_run and head_sha:
            title, summary = format_check_run_summary(score)
            await client.create_check_run(
                repo_owner, repo_name, head_sha, title, summary
            )

    cache.close()

    # Set outputs
    _set_output("score", f"{score.normalized_score:.2f}")
    _set_output("trust-level", score.trust_level.value)
    _set_output("user", score.user_login)

    # Summary
    pct = score.normalized_score * 100
    logger.info("Good Egg: %s (%.0f%%) for %s", score.trust_level.value, pct, pr_author)

    # Fail if configured and trust is low
    if fail_on_low and score.trust_level == TrustLevel.LOW:
        print(f"::error::Trust level is LOW for {pr_author}")
        sys.exit(1)


def _set_output(name: str, value: str) -> None:
    """Set a GitHub Actions output variable."""
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"{name}={value}\n")


def main() -> None:
    """Entry point."""
    try:
        asyncio.run(run_action())
    except RateLimitExhaustedError as exc:
        print(f"::error::Rate limit exhausted. Resets at {exc.reset_at.isoformat()}. "
              "Consider using a GitHub App token for higher limits.")
        sys.exit(1)
    except UserNotFoundError as exc:
        print(f"::warning::User not found: {exc.login}. Setting outputs to UNKNOWN.")
        _set_output("score", "0.00")
        _set_output("trust-level", "UNKNOWN")
        _set_output("user", exc.login)
    except GoodEggError as exc:
        print(f"::error::{exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
