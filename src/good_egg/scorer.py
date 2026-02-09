"""Trust scoring engine using graph-based ranking over contribution graphs."""

from __future__ import annotations

import os
from collections import defaultdict

import networkx as nx

from good_egg.config import GoodEggConfig, load_config
from good_egg.graph_builder import TrustGraphBuilder
from good_egg.models import (
    ContributionSummary,
    TrustLevel,
    TrustScore,
    UserContributionData,
)


class TrustScorer:
    """Compute trust scores for GitHub users via personalised graph scoring."""

    def __init__(self, config: GoodEggConfig) -> None:
        self.config = config
        self._graph_builder = TrustGraphBuilder(config)

    def score(
        self, user_data: UserContributionData, context_repo: str
    ) -> TrustScore:
        """Score a user based on their contribution data relative to *context_repo*."""
        login = user_data.profile.login
        flags: dict[str, bool] = {
            "is_bot": user_data.profile.is_bot,
            "is_new_account": (
                user_data.profile.account_age_days
                < self.config.thresholds.new_account_days
            ),
            "has_insufficient_data": False,
            "used_cached_data": False,
        }

        # ---- Bot short-circuit ----
        if user_data.profile.is_bot:
            return TrustScore(
                user_login=login,
                context_repo=context_repo,
                raw_score=0.0,
                normalized_score=0.0,
                trust_level=TrustLevel.BOT,
                account_age_days=user_data.profile.account_age_days,
                flags=flags,
            )

        # ---- Insufficient data short-circuit ----
        if not user_data.merged_prs:
            flags["has_insufficient_data"] = True
            return TrustScore(
                user_login=login,
                context_repo=context_repo,
                raw_score=0.0,
                normalized_score=0.0,
                trust_level=TrustLevel.UNKNOWN,
                account_age_days=user_data.profile.account_age_days,
                flags=flags,
            )

        # ---- Suspected bot flag ----
        if user_data.profile.is_suspected_bot:
            flags["is_suspected_bot"] = True

        # ---- Contribution stats ----
        total_prs = len(user_data.merged_prs)
        unique_repos = len({pr.repo_name_with_owner for pr in user_data.merged_prs})

        # ---- Build graph ----
        graph = self._graph_builder.build_graph(user_data, context_repo)

        # ---- Determine context language ----
        context_language = self._resolve_context_language(
            user_data, context_repo
        )

        # ---- Build personalization vector ----
        personalization = self._graph_builder.build_personalization_vector(
            graph, context_repo, context_language,
            total_prs=total_prs, unique_repos=unique_repos,
        )

        # ---- Run graph scoring ----
        pr_scores = nx.pagerank(
            graph,
            alpha=self.config.graph_scoring.alpha,
            personalization=personalization if personalization else None,
            weight="weight",
            max_iter=self.config.graph_scoring.max_iterations,
            tol=self.config.graph_scoring.tolerance,
        )

        user_node = f"user:{login}"
        raw_score = pr_scores.get(user_node, 0.0)

        # ---- Normalize ----
        normalized = self._normalize(raw_score, graph)

        # ---- Classify ----
        trust_level = self._classify(normalized, flags)

        # ---- Build top contributions ----
        top_contributions = self._build_top_contributions(user_data)

        # ---- Language match ----
        language_match = self._check_language_match(
            user_data, context_language
        )

        return TrustScore(
            user_login=login,
            context_repo=context_repo,
            raw_score=raw_score,
            normalized_score=normalized,
            trust_level=trust_level,
            account_age_days=user_data.profile.account_age_days,
            total_merged_prs=len(user_data.merged_prs),
            unique_repos_contributed=unique_repos,
            top_contributions=top_contributions,
            language_match=language_match,
            flags=flags,
            scoring_metadata={
                "graph_nodes": graph.number_of_nodes(),
                "graph_edges": graph.number_of_edges(),
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_context_language(
        user_data: UserContributionData, context_repo: str
    ) -> str | None:
        """Return the primary language of the context repo if known."""
        meta = user_data.contributed_repos.get(context_repo)
        if meta:
            return meta.primary_language
        return None

    def _normalize(self, raw_score: float, graph: nx.DiGraph) -> float:
        """Heuristic normalization of a raw graph score to [0, 1].

        Uses the number of nodes in the graph as a scaling reference:
        a uniform distribution would give 1/N per node; we scale relative
        to that baseline.
        """
        n = graph.number_of_nodes()
        if n == 0:
            return 0.0
        baseline = 1.0 / n
        if baseline == 0:
            return 0.0
        ratio = raw_score / baseline
        # Sigmoid-like mapping: ratio of 1.0 (uniform) maps to ~0.5
        normalized = ratio / (ratio + 1.0)
        return min(1.0, max(0.0, normalized))

    def _classify(
        self, normalized_score: float, flags: dict[str, bool]
    ) -> TrustLevel:
        """Map a normalized score to a trust level."""
        if flags.get("is_bot"):
            return TrustLevel.BOT
        if normalized_score >= self.config.thresholds.high_trust:
            return TrustLevel.HIGH
        if normalized_score >= self.config.thresholds.medium_trust:
            return TrustLevel.MEDIUM
        return TrustLevel.LOW

    @staticmethod
    def _build_top_contributions(
        user_data: UserContributionData,
    ) -> list[ContributionSummary]:
        """Aggregate PRs by repo and return sorted by PR count descending."""
        counts: dict[str, int] = defaultdict(int)
        for pr in user_data.merged_prs:
            counts[pr.repo_name_with_owner] += 1

        summaries: list[ContributionSummary] = []
        for repo_name, pr_count in counts.items():
            meta = user_data.contributed_repos.get(repo_name)
            summaries.append(
                ContributionSummary(
                    repo_name=repo_name,
                    pr_count=pr_count,
                    language=meta.primary_language if meta else None,
                    stars=meta.stargazer_count if meta else 0,
                )
            )

        summaries.sort(key=lambda s: s.pr_count, reverse=True)
        return summaries[:10]

    @staticmethod
    def _check_language_match(
        user_data: UserContributionData,
        context_language: str | None,
    ) -> bool:
        """Check if any contributed repo shares the context language."""
        if context_language is None:
            return False
        return any(
            meta.primary_language == context_language
            for meta in user_data.contributed_repos.values()
        )


async def score_pr_author(
    login: str,
    repo_owner: str,
    repo_name: str,
    config: GoodEggConfig | None = None,
    token: str | None = None,
    cache: object | None = None,
) -> TrustScore:
    """Convenience function: fetch data and score a PR author.

    Parameters
    ----------
    login:
        GitHub username to score.
    repo_owner:
        Owner of the context repository.
    repo_name:
        Name of the context repository.
    config:
        Optional configuration; defaults are used when *None*.
    token:
        GitHub token; falls back to the ``GITHUB_TOKEN`` env var.
    cache:
        Optional :class:`~good_egg.cache.Cache` instance for caching API
        responses.  When *None*, no caching is performed.
    """
    from good_egg.github_client import GitHubClient

    if config is None:
        config = load_config()

    if token is None:
        token = os.environ.get("GITHUB_TOKEN", "")

    context_repo = f"{repo_owner}/{repo_name}"

    async with GitHubClient(token=token, config=config, cache=cache) as client:  # type: ignore[arg-type]
        user_data = await client.get_user_contribution_data(
            login, context_repo=context_repo
        )

    scorer = TrustScorer(config)
    return scorer.score(user_data, context_repo)
