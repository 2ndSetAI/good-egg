from __future__ import annotations

from collections import Counter

import pytest

from experiments.bot_detection.stages.stage6_title_analysis import (
    compute_cross_author_commonality,
    compute_lexical_poverty,
    compute_template_match_rate,
    compute_title_features,
    compute_title_shortness,
    compute_within_author_homogeneity,
)


class TestTitleShortness:
    def test_empty(self):
        assert compute_title_shortness([]) == 0.0

    def test_short_titles_score_higher(self):
        short = compute_title_shortness(["fix typo", "update"])
        long = compute_title_shortness(
            ["Refactor the authentication middleware to support OAuth2"],
        )
        assert short > long

    def test_single_char(self):
        score = compute_title_shortness(["x"])
        assert score == pytest.approx(0.5)  # 1/(1+1)


class TestLexicalPoverty:
    def test_empty(self):
        assert compute_lexical_poverty([]) == 0.0

    def test_all_unique(self):
        score = compute_lexical_poverty(["alpha beta gamma"])
        assert score == pytest.approx(0.0)

    def test_all_repeated(self):
        score = compute_lexical_poverty(["fix fix fix fix"])
        assert score == pytest.approx(0.75)  # 1 - 1/4

    def test_repetitive_titles_score_higher(self):
        repetitive = compute_lexical_poverty(
            ["fix typo", "fix typo", "fix typo"]
        )
        varied = compute_lexical_poverty(
            ["implement auth system", "add database migration", "refactor test suite"]
        )
        assert repetitive > varied


class TestWithinAuthorHomogeneity:
    def test_empty(self):
        assert compute_within_author_homogeneity([]) == 0.0

    def test_single_title(self):
        assert compute_within_author_homogeneity(["just one"]) == 0.0

    def test_identical_titles(self):
        score = compute_within_author_homogeneity(
            ["fix typo in readme", "fix typo in readme", "fix typo in readme"]
        )
        assert score == pytest.approx(1.0)

    def test_diverse_titles_score_lower(self):
        similar = compute_within_author_homogeneity(
            ["fix typo in readme", "fix typo in docs", "fix typo in changelog"]
        )
        diverse = compute_within_author_homogeneity(
            ["implement OAuth2 flow", "add database index", "refactor CSS grid layout"]
        )
        assert similar > diverse


class TestTemplateMatchRate:
    def test_empty(self):
        assert compute_template_match_rate([]) == 0.0

    def test_all_template(self):
        titles = ["fix typo", "update readme", "fix grammar"]
        assert compute_template_match_rate(titles) == pytest.approx(1.0)

    def test_no_template(self):
        titles = [
            "Implement distributed caching layer with Redis",
            "Refactor authentication middleware for multi-tenant support",
        ]
        assert compute_template_match_rate(titles) == pytest.approx(0.0)

    def test_two_word_titles_match(self):
        assert compute_template_match_rate(["Hello World"]) == pytest.approx(1.0)


class TestCrossAuthorCommonality:
    def test_empty(self):
        assert compute_cross_author_commonality([], Counter(), min_count=2) == 0.0

    def test_unique_titles(self):
        counts = Counter({"fix typo": 1, "unique title": 1})
        assert compute_cross_author_commonality(
            ["fix typo"], counts, min_count=2,
        ) == 0.0

    def test_common_titles(self):
        counts = Counter({"fix typo": 5, "update readme": 3})
        score = compute_cross_author_commonality(
            ["fix typo", "update readme", "my unique pr"],
            counts,
            min_count=2,
        )
        assert score == pytest.approx(2.0 / 3.0)


class TestComputeTitleFeatures:
    def test_empty_input(self):
        df = compute_title_features({})
        assert len(df) == 0

    def test_spammy_vs_legitimate(self):
        """Authors with spammy titles should get higher title_spam_score."""
        all_titles = {
            "spammer": ["fix typo", "fix typo", "update readme", "fix typo", "fix bug"],
            "legit": [
                "Implement JWT refresh token rotation",
                "Add PostgreSQL connection pooling with pgbouncer",
                "Refactor CI pipeline to use matrix builds",
                "Fix race condition in WebSocket handler",
                "Add OpenTelemetry distributed tracing",
            ],
        }
        df = compute_title_features(all_titles)
        spammer_score = df.loc[df["login"] == "spammer", "title_spam_score"].iloc[0]
        legit_score = df.loc[df["login"] == "legit", "title_spam_score"].iloc[0]
        assert spammer_score > legit_score

    def test_all_authors_have_scores(self):
        all_titles = {
            "alice": ["one title"],
            "bob": ["another title", "second pr"],
        }
        df = compute_title_features(all_titles)
        assert len(df) == 2
        assert "title_spam_score" in df.columns
        assert df["title_spam_score"].notna().all()

    def test_scores_between_0_and_1(self):
        all_titles = {
            f"user{i}": [f"title {i} for pr {j}" for j in range(5)]
            for i in range(10)
        }
        df = compute_title_features(all_titles)
        assert (df["title_spam_score"] >= 0.0).all()
        assert (df["title_spam_score"] <= 1.0).all()
