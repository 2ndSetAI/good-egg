"""Output formatting for Good Egg trust scores."""

from __future__ import annotations

import click

from good_egg.models import TrustLevel, TrustScore

COMMENT_MARKER = "<!-- good-egg-trust-score -->"

_TRUST_LEVEL_COLORS: dict[TrustLevel, str] = {
    TrustLevel.HIGH: "green",
    TrustLevel.MEDIUM: "yellow",
    TrustLevel.LOW: "red",
    TrustLevel.UNKNOWN: "white",
    TrustLevel.BOT: "blue",
    TrustLevel.EXISTING_CONTRIBUTOR: "green",
}

_TRUST_LEVEL_EMOJI: dict[TrustLevel, str] = {
    TrustLevel.HIGH: "\U0001f95a",       # egg
    TrustLevel.MEDIUM: "\U0001f95a",     # egg
    TrustLevel.LOW: "\u26a0\ufe0f",      # warning
    TrustLevel.UNKNOWN: "\u2753",        # question mark
    TrustLevel.BOT: "\U0001f916",        # robot
    TrustLevel.EXISTING_CONTRIBUTOR: "\u2705",  # checkmark
}


def _brand_name(score: TrustScore) -> str:
    """Return 'Better Egg' for v2, 'Good Egg' for v1."""
    return "Better Egg" if score.scoring_model == "v2" else "Good Egg"


def format_markdown_comment(score: TrustScore) -> str:
    """Format a trust score as a GitHub PR comment in Markdown."""
    if score.trust_level == TrustLevel.EXISTING_CONTRIBUTOR:
        pr_count = score.scoring_metadata.get("context_repo_merged_pr_count", 0)
        return "\n".join([
            COMMENT_MARKER,
            "\u2705 **Existing Contributor**",
            "",
            f"**{score.user_login}** has {pr_count} merged"
            f" PR{'s' if pr_count != 1 else ''} in `{score.context_repo}`.",
            "",
            "> Scoring skipped for known contributors.",
            "",
        ])

    emoji = _TRUST_LEVEL_EMOJI.get(score.trust_level, "\U0001f95a")
    pct = score.normalized_score * 100
    brand = _brand_name(score)

    lines: list[str] = [
        COMMENT_MARKER,
        f"## {emoji} {brand}: **{score.trust_level.value}** Trust",
        "",
        f"**Score:** {pct:.0f}%",
        "",
    ]

    # v2 component score breakdown
    if score.scoring_model == "v2" and score.component_scores:
        lines.append("### Score Breakdown")
        lines.append("")
        lines.append("| Component | Value |")
        lines.append("|-----------|-------|")
        if "graph_score" in score.component_scores:
            gs = score.component_scores["graph_score"] * 100
            lines.append(f"| Graph Score | {gs:.0f}% |")
        if "merge_rate" in score.component_scores:
            mr = score.component_scores["merge_rate"] * 100
            merged = score.total_merged_prs
            total = merged + (
                score.scoring_metadata.get("closed_pr_count", 0)
                if isinstance(score.scoring_metadata.get("closed_pr_count"), int)
                else 0
            )
            if total > 0:
                lines.append(f"| Merge Rate | {mr:.0f}% ({merged}/{total} PRs) |")
            else:
                lines.append(f"| Merge Rate | {mr:.0f}% |")
        if "log_account_age" in score.component_scores:
            lines.append(
                f"| Account Age | {score.account_age_days:,} days |"
            )
        lines.append("")

    # Top contributions table
    if score.top_contributions:
        lines.append("### Top Contributions")
        lines.append("")
        lines.append("| Repository | PRs | Language | Stars |")
        lines.append("|------------|-----|----------|-------|")
        for c in score.top_contributions:
            lang = c.language or "N/A"
            lines.append(f"| {c.repo_name} | {c.pr_count} | {lang} | {c.stars} |")
        lines.append("")

    # Flags section
    flags_to_show: list[str] = []
    if score.flags.get("is_new_account"):
        flags_to_show.append("\u26a0\ufe0f New account (< 30 days)")
    if score.flags.get("has_insufficient_data"):
        flags_to_show.append("\u2139\ufe0f Insufficient data for confident scoring")
    if score.flags.get("is_bot"):
        flags_to_show.append("\U0001f916 Bot account detected")
    if score.flags.get("used_cached_data"):
        flags_to_show.append("\U0001f4be Using cached data")

    if flags_to_show:
        lines.append("### Flags")
        lines.append("")
        for flag in flags_to_show:
            lines.append(f"- {flag}")
        lines.append("")

    # Low trust note
    if score.trust_level == TrustLevel.LOW:
        lines.append("> **First-time contributor -- review manually**")
        lines.append("")

    return "\n".join(lines)


def format_cli_output(score: TrustScore, verbose: bool = False) -> str:
    """Format a trust score for terminal display with color."""
    if score.trust_level == TrustLevel.EXISTING_CONTRIBUTOR:
        pr_count = score.scoring_metadata.get("context_repo_merged_pr_count", 0)
        label = click.style("EXISTING CONTRIBUTOR", fg="green", bold=True)
        return "\n".join([
            label,
            f"User: {score.user_login}",
            f"Context: {score.context_repo}",
            f"Merged PRs in repo: {pr_count}",
        ])

    color = _TRUST_LEVEL_COLORS.get(score.trust_level, "white")
    pct = score.normalized_score * 100
    brand = _brand_name(score)

    trust_styled = click.style(score.trust_level.value, fg=color, bold=True)
    pct_styled = click.style(f"{pct:.0f}%", bold=True)

    lines: list[str] = [
        f"{brand}: {trust_styled} ({pct_styled})",
        f"User: {score.user_login}",
        f"Context: {score.context_repo}",
    ]

    if verbose:
        lines.append("")
        lines.append(
            f"Account age: {score.account_age_days} days | "
            f"Merged PRs: {score.total_merged_prs} | "
            f"Repos: {score.unique_repos_contributed}"
        )

        if score.scoring_model == "v2" and score.component_scores:
            lines.append("")
            lines.append("Component scores:")
            if "graph_score" in score.component_scores:
                gs = score.component_scores["graph_score"] * 100
                lines.append(f"  Graph Score: {gs:.0f}%")
            if "merge_rate" in score.component_scores:
                mr = score.component_scores["merge_rate"] * 100
                lines.append(f"  Merge Rate: {mr:.0f}%")
            if "log_account_age" in score.component_scores:
                lines.append(
                    f"  Account Age: {score.account_age_days:,} days"
                )

        if score.top_contributions:
            lines.append("")
            lines.append("Top contributions:")
            for c in score.top_contributions:
                lang = c.language or "N/A"
                lines.append(f"  {c.repo_name}: {c.pr_count} PRs ({lang}, {c.stars} stars)")

        if score.flags:
            lines.append("")
            lines.append("Flags:")
            for flag, value in score.flags.items():
                if value:
                    lines.append(f"  - {flag}")

        if score.scoring_metadata:
            lines.append("")
            lines.append("Metadata:")
            for key, value in score.scoring_metadata.items():
                lines.append(f"  {key}: {value}")

    return "\n".join(lines)


def format_json(score: TrustScore) -> str:
    """Format a trust score as JSON."""
    return score.model_dump_json(indent=2)


def format_check_run_summary(score: TrustScore) -> tuple[str, str]:
    """Format a trust score for a GitHub Check Run.

    Returns:
        A (title, summary) tuple for the Check Run API.
    """
    if score.trust_level == TrustLevel.EXISTING_CONTRIBUTOR:
        pr_count = score.scoring_metadata.get("context_repo_merged_pr_count", 0)
        title = f"Existing Contributor: {score.user_login}"
        summary = (
            f"**{score.user_login}** has {pr_count} merged"
            f" PR{'s' if pr_count != 1 else ''}"
            f" in `{score.context_repo}`. Scoring skipped."
        )
        return title, summary

    pct = score.normalized_score * 100
    brand = _brand_name(score)
    title = f"{brand}: {score.trust_level.value} ({pct:.0f}%)"

    summary_lines: list[str] = [
        f"**Trust Level:** {score.trust_level.value}",
        f"**Score:** {pct:.0f}%",
        f"**User:** {score.user_login}",
        "",
        f"Account age: {score.account_age_days} days | "
        f"Merged PRs: {score.total_merged_prs} | "
        f"Repos contributed to: {score.unique_repos_contributed}",
    ]

    if score.scoring_model == "v2" and score.component_scores:
        summary_lines.append("")
        summary_lines.append("**Score Breakdown:**")
        if "graph_score" in score.component_scores:
            gs = score.component_scores["graph_score"] * 100
            summary_lines.append(f"- Graph Score: {gs:.0f}%")
        if "merge_rate" in score.component_scores:
            mr = score.component_scores["merge_rate"] * 100
            summary_lines.append(f"- Merge Rate: {mr:.0f}%")
        if "log_account_age" in score.component_scores:
            summary_lines.append(
                f"- Account Age: {score.account_age_days:,} days"
            )

    if score.top_contributions:
        summary_lines.append("")
        summary_lines.append("**Top Contributions:**")
        for c in score.top_contributions:
            lang = c.language or "N/A"
            summary_lines.append(f"- {c.repo_name}: {c.pr_count} PRs ({lang}, {c.stars} stars)")

    if score.flags:
        active_flags = [k for k, v in score.flags.items() if v]
        if active_flags:
            summary_lines.append("")
            summary_lines.append(f"**Flags:** {', '.join(active_flags)}")

    summary = "\n".join(summary_lines)
    return title, summary
