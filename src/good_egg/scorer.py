"""Trust scoring engine using graph-based ranking over contribution graphs."""

from __future__ import annotations

import math
import os
import statistics
from collections import defaultdict

import networkx as nx

from good_egg.config import GoodEggConfig, load_config
from good_egg.graph_builder import TrustGraphBuilder
from good_egg.models import (
    ContributionSummary,
    FreshAccountAdvisory,
    SuspicionLevel,
    SuspicionScore,
    TrustLevel,
    TrustScore,
    UserContributionData,
)


class TrustScorer:
    """Compute trust scores for GitHub users via personalised graph scoring."""

    def __init__(self, config: GoodEggConfig) -> None:
        self.config = config

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
        model_name = self.config.scoring_model

        fresh_account = self._build_fresh_account_advisory(user_data)

        # ---- Bot short-circuit ----
        # fresh_account is intentionally omitted: bot profiles have
        # unreliable age data so the advisory is not meaningful.
        if user_data.profile.is_bot:
            return TrustScore(
                user_login=login,
                context_repo=context_repo,
                raw_score=0.0,
                normalized_score=0.0,
                trust_level=TrustLevel.BOT,
                account_age_days=user_data.profile.account_age_days,
                flags=flags,
                scoring_model=model_name,
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
                scoring_model=model_name,
                fresh_account=fresh_account,
            )

        # ---- Suspected bot flag ----
        if user_data.profile.is_suspected_bot:
            flags["is_suspected_bot"] = True

        if model_name == "v3":
            return self._score_v3(user_data, context_repo, flags, fresh_account)
        if model_name == "v2":
            return self._score_v2(user_data, context_repo, flags, fresh_account)
        return self._score_v1(user_data, context_repo, flags, fresh_account)

    # ------------------------------------------------------------------
    # v1 scoring path
    # ------------------------------------------------------------------

    def _score_v1(
        self,
        user_data: UserContributionData,
        context_repo: str,
        flags: dict[str, bool],
        fresh_account: FreshAccountAdvisory | None = None,
    ) -> TrustScore:
        """Original graph-based scoring pipeline."""
        login = user_data.profile.login
        graph_builder = TrustGraphBuilder(self.config)
        total_prs = len(user_data.merged_prs)
        unique_repos = len({pr.repo_name_with_owner for pr in user_data.merged_prs})

        graph = graph_builder.build_graph(user_data, context_repo)
        context_language = self._resolve_context_language(user_data, context_repo)
        personalization = graph_builder.build_personalization_vector(
            graph, context_repo, context_language,
            total_prs=total_prs, unique_repos=unique_repos,
        )

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
        normalized = self._normalize(raw_score, graph)
        trust_level = self._classify(normalized, flags)
        top_contributions = self._build_top_contributions(user_data)
        language_match = self._check_language_match(user_data, context_language)

        result = TrustScore(
            user_login=login,
            context_repo=context_repo,
            raw_score=raw_score,
            normalized_score=normalized,
            trust_level=trust_level,
            account_age_days=user_data.profile.account_age_days,
            total_merged_prs=total_prs,
            unique_repos_contributed=unique_repos,
            top_contributions=top_contributions,
            language_match=language_match,
            flags=flags,
            scoring_metadata={
                "graph_nodes": graph.number_of_nodes(),
                "graph_edges": graph.number_of_edges(),
            },
            scoring_model="v1",
            fresh_account=fresh_account,
        )

        if self.config.bad_egg.enabled and user_data.merged_prs:
            result.suspicion_score = self._compute_suspicion_score(user_data)

        return result

    # ------------------------------------------------------------------
    # v2 scoring path (Better Egg)
    # ------------------------------------------------------------------

    def _score_v2(
        self,
        user_data: UserContributionData,
        context_repo: str,
        flags: dict[str, bool],
        fresh_account: FreshAccountAdvisory | None = None,
    ) -> TrustScore:
        """v2 scoring: simplified graph + logistic regression combined model."""
        login = user_data.profile.login
        graph_builder = TrustGraphBuilder(self.config, simplified=True)
        total_prs = len(user_data.merged_prs)
        unique_repos = len({pr.repo_name_with_owner for pr in user_data.merged_prs})

        # Step 1-2: build simplified graph
        graph = graph_builder.build_graph(user_data, context_repo)
        context_language = self._resolve_context_language(user_data, context_repo)
        personalization = graph_builder.build_personalization_vector(
            graph, context_repo, context_language,
            total_prs=total_prs, unique_repos=unique_repos,
        )

        # Step 3: run graph scoring
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
        graph_score = self._normalize(raw_score, graph)

        # Step 4: compute external features
        merged_count = len(user_data.merged_prs)
        closed_count = user_data.closed_pr_count
        total_author_prs = merged_count + closed_count
        merge_rate: float | None = (
            merged_count / total_author_prs if total_author_prs > 0 else None
        )
        log_account_age = math.log(user_data.profile.account_age_days + 1)

        # Step 5: combined model
        v2_cfg = self.config.v2
        cm = v2_cfg.combined_model
        logit = cm.intercept + cm.graph_score_weight * graph_score

        if v2_cfg.features.merge_rate and merge_rate is not None:
            logit += cm.merge_rate_weight * merge_rate
        if v2_cfg.features.account_age:
            logit += cm.account_age_weight * log_account_age

        # Sigmoid: P(merge)
        normalized = 1.0 / (1.0 + math.exp(-logit))

        # Step 6-7: classify and build result
        trust_level = self._classify(normalized, flags)
        top_contributions = self._build_top_contributions(user_data)
        language_match = self._check_language_match(user_data, context_language)

        component_scores: dict[str, float] = {
            "graph_score": graph_score,
        }
        if merge_rate is not None:
            component_scores["merge_rate"] = merge_rate
        component_scores["log_account_age"] = log_account_age

        result = TrustScore(
            user_login=login,
            context_repo=context_repo,
            raw_score=logit,
            normalized_score=normalized,
            trust_level=trust_level,
            account_age_days=user_data.profile.account_age_days,
            total_merged_prs=total_prs,
            unique_repos_contributed=unique_repos,
            top_contributions=top_contributions,
            language_match=language_match,
            flags=flags,
            scoring_metadata={
                "graph_nodes": graph.number_of_nodes(),
                "graph_edges": graph.number_of_edges(),
                "closed_pr_count": closed_count,
            },
            scoring_model="v2",
            component_scores=component_scores,
            fresh_account=fresh_account,
        )

        if self.config.bad_egg.enabled and user_data.merged_prs:
            result.suspicion_score = self._compute_suspicion_score(user_data)

        return result

    # ------------------------------------------------------------------
    # v3 scoring path (Diet Egg)
    # ------------------------------------------------------------------

    def _score_v3(
        self,
        user_data: UserContributionData,
        context_repo: str,
        flags: dict[str, bool],
        fresh_account: FreshAccountAdvisory | None = None,
    ) -> TrustScore:
        """v3 scoring: merge rate as sole signal."""
        login = user_data.profile.login
        total_prs = len(user_data.merged_prs)
        unique_repos = len({pr.repo_name_with_owner for pr in user_data.merged_prs})

        merged_count = len(user_data.merged_prs)
        closed_count = user_data.closed_pr_count
        total_author_prs = merged_count + closed_count
        merge_rate = (
            merged_count / total_author_prs if total_author_prs > 0 else 0.0
        )

        trust_level = self._classify(merge_rate, flags)
        context_language = self._resolve_context_language(user_data, context_repo)
        top_contributions = self._build_top_contributions(user_data)
        language_match = self._check_language_match(user_data, context_language)

        result = TrustScore(
            user_login=login,
            context_repo=context_repo,
            raw_score=merge_rate,
            normalized_score=merge_rate,
            trust_level=trust_level,
            account_age_days=user_data.profile.account_age_days,
            total_merged_prs=total_prs,
            unique_repos_contributed=unique_repos,
            top_contributions=top_contributions,
            language_match=language_match,
            flags=flags,
            scoring_metadata={
                "closed_pr_count": closed_count,
            },
            scoring_model="v3",
            component_scores={"merge_rate": merge_rate},
            fresh_account=fresh_account,
        )

        if self.config.bad_egg.enabled and user_data.merged_prs:
            result.suspicion_score = self._compute_suspicion_score(user_data)

        return result

    # ------------------------------------------------------------------
    # Fresh account advisory
    # ------------------------------------------------------------------

    @staticmethod
    def _build_fresh_account_advisory(
        user_data: UserContributionData,
    ) -> FreshAccountAdvisory:
        """Build a fresh account advisory from user profile data."""
        threshold = FreshAccountAdvisory.model_fields["threshold_days"].default
        age_days = user_data.profile.account_age_days
        return FreshAccountAdvisory(
            is_fresh=age_days < threshold,
            account_age_days=age_days,
            created_at=user_data.profile.created_at,
        )

    # ------------------------------------------------------------------
    # Bad Egg: suspension advisory score
    # ------------------------------------------------------------------

    def _compute_suspicion_score(
        self, user_data: UserContributionData,
    ) -> SuspicionScore:
        """Compute advisory suspension risk score using 8-feature LR model."""
        cfg = self.config.bad_egg

        # Feature 1: merge_rate
        merged_count = len(user_data.merged_prs)
        closed_count = user_data.closed_pr_count
        total_prs = merged_count + closed_count
        merge_rate = merged_count / total_prs if total_prs > 0 else 0.0

        # Feature 2: total_prs (log-transformed)
        log_total_prs = math.log1p(total_prs)

        # Feature 3: career_span_days (log-transformed)
        if len(user_data.merged_prs) >= 2:
            dates = [pr.merged_at for pr in user_data.merged_prs]
            span = (max(dates) - min(dates)).total_seconds() / 86400.0
        else:
            span = 0.0
        log_career_span = math.log1p(span)

        # Feature 4: mean_title_length
        if user_data.merged_prs:
            mean_title_len = sum(
                len(pr.title) for pr in user_data.merged_prs
            ) / len(user_data.merged_prs)
        else:
            mean_title_len = 0.0

        # Feature 5: isolation_score (from bipartite contributor graph)
        isolation_score = self._compute_isolation_score(user_data)

        # Feature 6: total_repos
        total_repos = len({pr.repo_name_with_owner for pr in user_data.merged_prs})

        # Feature 7: median_additions (log-transformed)
        additions = [pr.additions for pr in user_data.merged_prs]
        median_adds = statistics.median(additions) if additions else 0.0
        log_median_additions = math.log1p(median_adds)

        # Feature 8: median_files_changed (log-transformed)
        files = [pr.changed_files for pr in user_data.merged_prs]
        median_files = statistics.median(files) if files else 0.0
        log_median_files = math.log1p(median_files)

        # Logistic regression
        m = cfg.model
        logit = (
            m.intercept
            + m.merge_rate_weight * merge_rate
            + m.total_prs_weight * log_total_prs
            + m.career_span_days_weight * log_career_span
            + m.mean_title_length_weight * mean_title_len
            + m.isolation_score_weight * isolation_score
            + m.total_repos_weight * total_repos
            + m.median_additions_weight * log_median_additions
            + m.median_files_changed_weight * log_median_files
        )
        probability = 1.0 / (1.0 + math.exp(-logit))

        # Classify tier
        t = cfg.thresholds
        if probability >= t.high:
            level = SuspicionLevel.HIGH
        elif probability >= t.elevated:
            level = SuspicionLevel.ELEVATED
        else:
            level = SuspicionLevel.NORMAL

        return SuspicionScore(
            raw_score=logit,
            probability=probability,
            suspicion_level=level,
            component_scores={
                "merge_rate": merge_rate,
                "log_total_prs": log_total_prs,
                "log_career_span_days": log_career_span,
                "mean_title_length": mean_title_len,
                "isolation_score": isolation_score,
                "total_repos": float(total_repos),
                "log_median_additions": log_median_additions,
                "log_median_files_changed": log_median_files,
            },
        )

    @staticmethod
    def _compute_isolation_score(user_data: UserContributionData) -> float:
        """Fraction of author's repos where no other multi-repo contributor works."""
        login = user_data.profile.login.lower()
        user_repos = set(user_data.repo_contributors.keys())

        if not user_repos:
            return 1.0

        # Build lookup: contributor -> set of repos they appear in
        contributor_repos: dict[str, set[str]] = defaultdict(set)
        for repo, contributors in user_data.repo_contributors.items():
            for c in contributors:
                if c.lower() != login:
                    contributor_repos[c.lower()].add(repo)

        # Multi-repo contributors: appear in 2+ of this author's repos
        multi_repo_contributors = {
            c for c, repos in contributor_repos.items() if len(repos) >= 2
        }

        # A repo is isolated if none of its contributors are multi-repo
        isolated = 0
        for repo in user_repos:
            repo_contribs = {
                c.lower() for c in user_data.repo_contributors.get(repo, [])
                if c.lower() != login
            }
            if not repo_contribs & multi_repo_contributors:
                isolated += 1

        # Also count repos where we skipped fetching (popular repos) as non-isolated
        all_contributed = set(user_data.contributed_repos.keys())
        skipped_repos = all_contributed - user_repos
        total = len(user_repos) + len(skipped_repos)

        return isolated / total if total > 0 else 1.0

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
        if config.skip_known_contributors:
            merged_count = await client.check_existing_contributor(
                login, repo_owner, repo_name,
            )
            if merged_count > 0:
                return TrustScore(
                    user_login=login,
                    context_repo=context_repo,
                    trust_level=TrustLevel.EXISTING_CONTRIBUTOR,
                    flags={
                        "is_existing_contributor": True,
                        "scoring_skipped": True,
                    },
                    scoring_metadata={
                        "context_repo_merged_pr_count": merged_count,
                    },
                    scoring_model=config.scoring_model,
                )

        user_data = await client.get_user_contribution_data(
            login, context_repo=context_repo
        )

    scorer = TrustScorer(config)
    return scorer.score(user_data, context_repo)
