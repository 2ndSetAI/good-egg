"""Example: Score a GitHub user with Good Egg (v3 / Diet Egg)."""

from __future__ import annotations

import asyncio
import os

from good_egg import score_pr_author


async def main() -> None:
    result = await score_pr_author(
        login="octocat",
        repo_owner="octocat",
        repo_name="Hello-World",
        token=os.environ["GITHUB_TOKEN"],
    )
    print(f"User: {result.user_login}")
    print(f"Trust level: {result.trust_level}")
    print(f"Scoring model: {result.scoring_model}")

    if result.flags.get("scoring_skipped"):
        pr_count = result.scoring_metadata.get("context_repo_merged_pr_count", 0)
        print(f"Scoring skipped -- {pr_count} merged PRs in repo")
    else:
        print(f"Score: {result.normalized_score:.2f}")
        print(f"Merged PRs: {result.total_merged_prs}")
        print(f"Unique repos: {result.unique_repos_contributed}")

        # v3 component scores
        if result.component_scores:
            merge_rate = result.component_scores.get("merge_rate")
            if merge_rate is not None:
                print(f"Merge rate: {merge_rate:.0%}")

        # Fresh Egg advisory
        if result.fresh_account and result.fresh_account.is_fresh:
            print(
                f"Fresh account: {result.fresh_account.account_age_days} days old"
                f" (< {result.fresh_account.threshold_days} days)"
            )


if __name__ == "__main__":
    asyncio.run(main())
