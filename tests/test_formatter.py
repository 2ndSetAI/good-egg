"""Tests for output formatting."""

from __future__ import annotations

import json

from good_egg.formatter import (
    COMMENT_MARKER,
    format_check_run_summary,
    format_cli_output,
    format_json,
    format_markdown_comment,
)
from good_egg.models import ContributionSummary, TrustLevel, TrustScore


def _make_score(**kwargs) -> TrustScore:
    """Helper to build a TrustScore with sensible defaults."""
    defaults = {
        "user_login": "testuser",
        "context_repo": "owner/repo",
        "raw_score": 0.65,
        "normalized_score": 0.72,
        "trust_level": TrustLevel.HIGH,
        "percentile": 85.0,
        "account_age_days": 365,
        "total_merged_prs": 42,
        "unique_repos_contributed": 10,
        "top_contributions": [
            ContributionSummary(
                repo_name="cool/project",
                pr_count=15,
                language="Python",
                stars=1200,
            ),
            ContributionSummary(
                repo_name="another/repo",
                pr_count=8,
                language="Rust",
                stars=500,
            ),
        ],
        "language_match": True,
        "flags": {},
        "scoring_metadata": {},
    }
    defaults.update(kwargs)
    return TrustScore(**defaults)


class TestFormatMarkdownComment:
    def test_contains_marker(self) -> None:
        score = _make_score()
        md = format_markdown_comment(score)
        assert COMMENT_MARKER in md

    def test_high_trust_header(self) -> None:
        score = _make_score(trust_level=TrustLevel.HIGH)
        md = format_markdown_comment(score)
        assert "HIGH" in md
        assert "72%" in md

    def test_contributions_table(self) -> None:
        score = _make_score()
        md = format_markdown_comment(score)
        assert "cool/project" in md
        assert "| 15 |" in md
        assert "Python" in md

    def test_no_contributions_table_when_empty(self) -> None:
        score = _make_score(top_contributions=[])
        md = format_markdown_comment(score)
        assert "Top Contributions" not in md

    def test_flags_shown(self) -> None:
        score = _make_score(flags={"is_new_account": True, "is_bot": False})
        md = format_markdown_comment(score)
        assert "New account" in md
        assert "Bot account" not in md

    def test_all_flags_render(self) -> None:
        score = _make_score(flags={
            "is_new_account": True,
            "has_insufficient_data": True,
            "is_bot": True,
            "used_cached_data": True,
        })
        md = format_markdown_comment(score)
        assert "New account" in md
        assert "Insufficient data" in md
        assert "Bot account detected" in md
        assert "Using cached data" in md

    def test_low_trust_manual_review_note(self) -> None:
        score = _make_score(trust_level=TrustLevel.LOW, normalized_score=0.15)
        md = format_markdown_comment(score)
        assert "First-time contributor -- review manually" in md

    def test_high_trust_no_manual_review_note(self) -> None:
        score = _make_score(trust_level=TrustLevel.HIGH)
        md = format_markdown_comment(score)
        assert "review manually" not in md

    def test_null_language_in_contributions(self) -> None:
        score = _make_score(top_contributions=[
            ContributionSummary(repo_name="x/y", pr_count=1, language=None, stars=0),
        ])
        md = format_markdown_comment(score)
        assert "N/A" in md


class TestFormatCliOutput:
    def test_basic_output(self) -> None:
        score = _make_score()
        out = format_cli_output(score)
        assert "testuser" in out
        assert "owner/repo" in out

    def test_verbose_shows_contributions(self) -> None:
        score = _make_score()
        out = format_cli_output(score, verbose=True)
        assert "cool/project" in out
        assert "15 PRs" in out

    def test_non_verbose_hides_contributions(self) -> None:
        score = _make_score()
        out = format_cli_output(score, verbose=False)
        assert "cool/project" not in out

    def test_verbose_shows_flags(self) -> None:
        score = _make_score(flags={"is_new_account": True})
        out = format_cli_output(score, verbose=True)
        assert "is_new_account" in out

    def test_verbose_shows_metadata(self) -> None:
        score = _make_score(scoring_metadata={"graph_nodes": 42})
        out = format_cli_output(score, verbose=True)
        assert "graph_nodes" in out
        assert "42" in out

    def test_contains_ansi_color_codes(self) -> None:
        score = _make_score(trust_level=TrustLevel.HIGH)
        out = format_cli_output(score)
        # click.style adds ANSI escape codes
        assert "\x1b[" in out


class TestFormatJson:
    def test_valid_json(self) -> None:
        score = _make_score()
        result = format_json(score)
        parsed = json.loads(result)
        assert parsed["user_login"] == "testuser"
        assert parsed["trust_level"] == "HIGH"

    def test_roundtrip(self) -> None:
        score = _make_score()
        result = format_json(score)
        parsed = json.loads(result)
        restored = TrustScore(**parsed)
        assert restored.user_login == score.user_login
        assert restored.trust_level == score.trust_level
        assert restored.normalized_score == score.normalized_score


class TestFormatCheckRunSummary:
    def test_title_format(self) -> None:
        score = _make_score()
        title, _ = format_check_run_summary(score)
        assert title == "Good Egg: HIGH (72%)"

    def test_summary_contains_user(self) -> None:
        score = _make_score()
        _, summary = format_check_run_summary(score)
        assert "testuser" in summary

    def test_summary_contains_contributions(self) -> None:
        score = _make_score()
        _, summary = format_check_run_summary(score)
        assert "cool/project" in summary

    def test_summary_contains_flags(self) -> None:
        score = _make_score(flags={"is_new_account": True, "used_cached_data": True})
        _, summary = format_check_run_summary(score)
        assert "is_new_account" in summary
        assert "used_cached_data" in summary

    def test_no_flags_section_when_empty(self) -> None:
        score = _make_score(flags={})
        _, summary = format_check_run_summary(score)
        assert "Flags" not in summary
