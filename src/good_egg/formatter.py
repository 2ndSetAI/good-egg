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
}

_TRUST_LEVEL_EMOJI: dict[TrustLevel, str] = {
    TrustLevel.HIGH: "\U0001f95a",       # egg
    TrustLevel.MEDIUM: "\U0001f95a",     # egg
    TrustLevel.LOW: "\u26a0\ufe0f",      # warning
    TrustLevel.UNKNOWN: "\u2753",        # question mark
    TrustLevel.BOT: "\U0001f916",        # robot
}


def format_markdown_comment(score: TrustScore) -> str:
    """Format a trust score as a GitHub PR comment in Markdown."""
    emoji = _TRUST_LEVEL_EMOJI.get(score.trust_level, "\U0001f95a")
    pct = score.normalized_score * 100

    lines: list[str] = [
        COMMENT_MARKER,
        f"## {emoji} Good Egg: **{score.trust_level.value}** Trust",
        "",
        f"**Score:** {pct:.0f}%",
        "",
    ]

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
    if score.flags.get("new_account"):
        flags_to_show.append("\u26a0\ufe0f New account (< 30 days)")
    if score.flags.get("insufficient_data"):
        flags_to_show.append("\u2139\ufe0f Insufficient data for confident scoring")
    if score.flags.get("bot_detected"):
        flags_to_show.append("\U0001f916 Bot account detected")
    if score.flags.get("cached_data"):
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
    color = _TRUST_LEVEL_COLORS.get(score.trust_level, "white")
    pct = score.normalized_score * 100

    trust_styled = click.style(score.trust_level.value, fg=color, bold=True)
    pct_styled = click.style(f"{pct:.0f}%", bold=True)

    lines: list[str] = [
        f"Good Egg: {trust_styled} ({pct_styled})",
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
    pct = score.normalized_score * 100
    title = f"Good Egg: {score.trust_level.value} ({pct:.0f}%)"

    summary_lines: list[str] = [
        f"**Trust Level:** {score.trust_level.value}",
        f"**Score:** {pct:.0f}%",
        f"**User:** {score.user_login}",
        "",
        f"Account age: {score.account_age_days} days | "
        f"Merged PRs: {score.total_merged_prs} | "
        f"Repos contributed to: {score.unique_repos_contributed}",
    ]

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
