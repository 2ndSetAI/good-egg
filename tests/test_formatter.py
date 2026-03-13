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
from good_egg.models import (
    ContributionSummary,
    FreshAccountAdvisory,
    SuspicionLevel,
    SuspicionScore,
    TrustLevel,
    TrustScore,
)


def _make_score(**kwargs) -> TrustScore:
    """Helper to build a TrustScore with sensible defaults."""
    defaults = {
        "user_login": "testuser",
        "context_repo": "owner/repo",
        "raw_score": 0.65,
        "normalized_score": 0.72,
        "trust_level": TrustLevel.HIGH,

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


class TestBetterEggFormatting:
    def _make_v2_score(self, **kwargs) -> TrustScore:
        defaults = {
            "user_login": "testuser",
            "context_repo": "owner/repo",
            "raw_score": 0.65,
            "normalized_score": 0.72,
            "trust_level": TrustLevel.HIGH,

            "account_age_days": 1825,
            "total_merged_prs": 42,
            "unique_repos_contributed": 10,
            "top_contributions": [
                ContributionSummary(
                    repo_name="cool/project",
                    pr_count=15,
                    language="Python",
                    stars=1200,
                ),
            ],
            "language_match": True,
            "flags": {},
            "scoring_metadata": {"closed_pr_count": 8},
            "scoring_model": "v2",
            "component_scores": {
                "graph_score": 0.65,
                "merge_rate": 0.85,
                "log_account_age": 7.5,
            },
        }
        defaults.update(kwargs)
        return TrustScore(**defaults)

    def test_markdown_better_egg_header(self) -> None:
        score = self._make_v2_score()
        md = format_markdown_comment(score)
        assert "Better Egg" in md
        assert "Good Egg" not in md.split("Better Egg")[0]  # No "Good Egg" before Better Egg

    def test_markdown_component_breakdown(self) -> None:
        score = self._make_v2_score()
        md = format_markdown_comment(score)
        assert "Score Breakdown" in md
        assert "Graph Score" in md
        assert "Merge Rate" in md
        assert "Account Age" in md
        assert "65%" in md  # graph_score * 100
        assert "85% (42/50 PRs)" in md  # merge_rate with closed_pr_count=8

    def test_cli_better_egg_header(self) -> None:
        score = self._make_v2_score()
        out = format_cli_output(score)
        assert "Better Egg" in out

    def test_cli_verbose_component_scores(self) -> None:
        score = self._make_v2_score()
        out = format_cli_output(score, verbose=True)
        assert "Component scores" in out
        assert "Graph Score" in out
        assert "Merge Rate" in out

    def test_check_run_better_egg_title(self) -> None:
        score = self._make_v2_score()
        title, summary = format_check_run_summary(score)
        assert "Better Egg" in title
        assert "Score Breakdown" in summary

    def test_json_includes_v2_fields(self) -> None:
        score = self._make_v2_score()
        result = format_json(score)
        parsed = json.loads(result)
        assert parsed["scoring_model"] == "v2"
        assert "graph_score" in parsed["component_scores"]

    def test_v1_format_unchanged(self) -> None:
        score = _make_score()
        md = format_markdown_comment(score)
        assert "Good Egg" in md
        assert "Better Egg" not in md
        assert "Score Breakdown" not in md

    def test_v1_cli_unchanged(self) -> None:
        score = _make_score()
        out = format_cli_output(score)
        assert "Good Egg" in out
        assert "Better Egg" not in out

    def test_v1_check_run_unchanged(self) -> None:
        score = _make_score()
        title, _ = format_check_run_summary(score)
        assert title == "Good Egg: HIGH (72%)"


class TestExistingContributorFormatting:
    def _make_existing_contributor_score(self, pr_count: int = 5) -> TrustScore:
        return TrustScore(
            user_login="knownuser",
            context_repo="my-org/my-repo",
            trust_level=TrustLevel.EXISTING_CONTRIBUTOR,
            flags={"is_existing_contributor": True, "scoring_skipped": True},
            scoring_metadata={"context_repo_merged_pr_count": pr_count},
        )

    def test_markdown_existing_contributor(self) -> None:
        score = self._make_existing_contributor_score()
        md = format_markdown_comment(score)
        assert COMMENT_MARKER in md
        assert "Existing Contributor" in md
        assert "knownuser" in md
        assert "5 merged PRs" in md
        assert "my-org/my-repo" in md
        assert "Scoring skipped" in md

    def test_markdown_existing_contributor_singular(self) -> None:
        score = self._make_existing_contributor_score(pr_count=1)
        md = format_markdown_comment(score)
        assert "1 merged PR " in md
        assert "1 merged PRs" not in md

    def test_cli_existing_contributor(self) -> None:
        score = self._make_existing_contributor_score()
        out = format_cli_output(score)
        assert "EXISTING CONTRIBUTOR" in out
        assert "knownuser" in out
        assert "my-org/my-repo" in out
        assert "5" in out

    def test_check_run_existing_contributor(self) -> None:
        score = self._make_existing_contributor_score()
        title, summary = format_check_run_summary(score)
        assert "Existing Contributor" in title
        assert "knownuser" in title
        assert "5 merged PRs" in summary
        assert "Scoring skipped" in summary

    def test_json_existing_contributor(self) -> None:
        score = self._make_existing_contributor_score()
        result = format_json(score)
        parsed = json.loads(result)
        assert parsed["trust_level"] == "EXISTING_CONTRIBUTOR"
        assert parsed["flags"]["is_existing_contributor"] is True


class TestDietEggFormatting:
    def _make_v3_score(self, **kwargs) -> TrustScore:
        defaults = {
            "user_login": "testuser",
            "context_repo": "owner/repo",
            "raw_score": 0.75,
            "normalized_score": 0.75,
            "trust_level": TrustLevel.HIGH,
            "account_age_days": 500,
            "total_merged_prs": 30,
            "unique_repos_contributed": 8,
            "top_contributions": [
                ContributionSummary(
                    repo_name="cool/project",
                    pr_count=15,
                    language="Python",
                    stars=1200,
                ),
            ],
            "language_match": True,
            "flags": {},
            "scoring_metadata": {"closed_pr_count": 10},
            "scoring_model": "v3",
            "component_scores": {"merge_rate": 0.75},
        }
        defaults.update(kwargs)
        return TrustScore(**defaults)

    def test_markdown_diet_egg_header(self) -> None:
        score = self._make_v3_score()
        md = format_markdown_comment(score)
        assert "Diet Egg" in md
        assert "Better Egg" not in md
        assert "Good Egg" not in md

    def test_markdown_component_breakdown(self) -> None:
        score = self._make_v3_score()
        md = format_markdown_comment(score)
        assert "Score Breakdown" in md
        assert "Merge Rate" in md
        # graph_score and log_account_age should not appear
        assert "Graph Score" not in md
        assert "Account Age" not in md

    def test_cli_diet_egg_header(self) -> None:
        score = self._make_v3_score()
        out = format_cli_output(score)
        assert "Diet Egg" in out

    def test_cli_verbose_component_scores(self) -> None:
        score = self._make_v3_score()
        out = format_cli_output(score, verbose=True)
        assert "Component scores" in out
        assert "Merge Rate" in out
        assert "Graph Score" not in out

    def test_check_run_diet_egg_title(self) -> None:
        score = self._make_v3_score()
        title, summary = format_check_run_summary(score)
        assert "Diet Egg" in title
        assert "Score Breakdown" in summary
        assert "Merge Rate" in summary

    def test_json_includes_v3_fields(self) -> None:
        score = self._make_v3_score()
        result = format_json(score)
        parsed = json.loads(result)
        assert parsed["scoring_model"] == "v3"
        assert "merge_rate" in parsed["component_scores"]
        assert "graph_score" not in parsed["component_scores"]


class TestFreshAccountFormatting:
    def _make_fresh_score(self, is_fresh: bool = True, **kwargs) -> TrustScore:
        fresh = FreshAccountAdvisory(
            is_fresh=is_fresh,
            account_age_days=200 if is_fresh else 500,
        )
        defaults = {
            "user_login": "newbie",
            "context_repo": "owner/repo",
            "raw_score": 0.6,
            "normalized_score": 0.6,
            "trust_level": TrustLevel.MEDIUM,
            "account_age_days": 200 if is_fresh else 500,
            "total_merged_prs": 5,
            "unique_repos_contributed": 2,
            "flags": {},
            "scoring_model": "v3",
            "component_scores": {"merge_rate": 0.6},
            "fresh_account": fresh,
        }
        defaults.update(kwargs)
        return TrustScore(**defaults)

    def test_markdown_fresh_account_shown(self) -> None:
        score = self._make_fresh_score(is_fresh=True)
        md = format_markdown_comment(score)
        assert "Fresh Account" in md
        assert "200 days old" in md

    def test_markdown_fresh_account_hidden(self) -> None:
        score = self._make_fresh_score(is_fresh=False)
        md = format_markdown_comment(score)
        assert "Fresh Account" not in md

    def test_cli_fresh_account_shown(self) -> None:
        score = self._make_fresh_score(is_fresh=True)
        out = format_cli_output(score, verbose=True)
        assert "Fresh account" in out
        assert "200 days old" in out

    def test_cli_fresh_account_hidden_non_verbose(self) -> None:
        score = self._make_fresh_score(is_fresh=True)
        out = format_cli_output(score, verbose=False)
        assert "Fresh account" not in out

    def test_cli_fresh_account_hidden_not_fresh(self) -> None:
        score = self._make_fresh_score(is_fresh=False)
        out = format_cli_output(score, verbose=True)
        assert "Fresh account" not in out

    def test_check_run_fresh_account_shown(self) -> None:
        score = self._make_fresh_score(is_fresh=True)
        _, summary = format_check_run_summary(score)
        assert "Fresh Account" in summary
        assert "200 days old" in summary

    def test_check_run_fresh_account_hidden(self) -> None:
        score = self._make_fresh_score(is_fresh=False)
        _, summary = format_check_run_summary(score)
        assert "Fresh Account" not in summary

    def test_json_fresh_account_serialized(self) -> None:
        score = self._make_fresh_score(is_fresh=True)
        result = format_json(score)
        parsed = json.loads(result)
        assert parsed["fresh_account"]["is_fresh"] is True
        assert parsed["fresh_account"]["account_age_days"] == 200

    def test_json_fresh_account_none(self) -> None:
        score = TrustScore(
            user_login="u", context_repo="o/r", fresh_account=None
        )
        result = format_json(score)
        parsed = json.loads(result)
        assert parsed["fresh_account"] is None


class TestSuspicionAdvisoryFormatting:
    def _make_score_with_suspicion(
        self, level: SuspicionLevel, probability: float = 0.12
    ) -> TrustScore:
        ss = SuspicionScore(
            raw_score=1.0,
            probability=probability,
            suspicion_level=level,
            component_scores={
                "merge_rate": 0.5,
                "log_median_additions": 3.0,
                "isolation_score": 0.8,
            },
        )
        return _make_score(suspicion_score=ss)

    def test_markdown_shows_advisory_for_high(self) -> None:
        score = self._make_score_with_suspicion(SuspicionLevel.HIGH, 0.12)
        md = format_markdown_comment(score)
        assert "Suspension Advisory" in md
        assert "**HIGH**" in md
        assert "12%" in md
        assert "Does not confirm malicious intent" in md

    def test_markdown_shows_advisory_for_elevated(self) -> None:
        score = self._make_score_with_suspicion(SuspicionLevel.ELEVATED, 0.07)
        md = format_markdown_comment(score)
        assert "Suspension Advisory" in md
        assert "**ELEVATED**" in md
        assert "7%" in md

    def test_markdown_omits_advisory_for_normal(self) -> None:
        score = self._make_score_with_suspicion(SuspicionLevel.NORMAL, 0.02)
        md = format_markdown_comment(score)
        assert "Suspension Advisory" not in md

    def test_markdown_omits_advisory_when_none(self) -> None:
        score = _make_score()
        md = format_markdown_comment(score)
        assert "Suspension Advisory" not in md

    def test_cli_shows_suspicion_high(self) -> None:
        score = self._make_score_with_suspicion(SuspicionLevel.HIGH, 0.12)
        out = format_cli_output(score)
        assert "Suspicion: HIGH (12%)" in out

    def test_cli_shows_suspicion_elevated(self) -> None:
        score = self._make_score_with_suspicion(SuspicionLevel.ELEVATED, 0.07)
        out = format_cli_output(score)
        assert "Suspicion: ELEVATED (7%)" in out

    def test_cli_hides_normal_non_verbose(self) -> None:
        score = self._make_score_with_suspicion(SuspicionLevel.NORMAL, 0.02)
        out = format_cli_output(score, verbose=False)
        assert "Suspicion" not in out

    def test_cli_shows_normal_verbose(self) -> None:
        score = self._make_score_with_suspicion(SuspicionLevel.NORMAL, 0.02)
        out = format_cli_output(score, verbose=True)
        assert "Suspicion: NORMAL (2%)" in out

    def test_check_run_includes_advisory_for_high(self) -> None:
        score = self._make_score_with_suspicion(SuspicionLevel.HIGH, 0.12)
        _, summary = format_check_run_summary(score)
        assert "Suspension Advisory" in summary
        assert "HIGH" in summary

    def test_check_run_omits_advisory_for_normal(self) -> None:
        score = self._make_score_with_suspicion(SuspicionLevel.NORMAL, 0.02)
        _, summary = format_check_run_summary(score)
        assert "Suspension Advisory" not in summary

    def test_json_includes_suspicion_score(self) -> None:
        score = self._make_score_with_suspicion(SuspicionLevel.HIGH, 0.12)
        result = format_json(score)
        parsed = json.loads(result)
        assert parsed["suspicion_score"] is not None
        assert parsed["suspicion_score"]["suspicion_level"] == "HIGH"
        assert parsed["suspicion_score"]["probability"] == 0.12
