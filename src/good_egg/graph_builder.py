"""Trust graph construction from GitHub contribution data."""

from __future__ import annotations

import math
from collections import defaultdict

import networkx as nx

from good_egg.config import GoodEggConfig, PageRankConfig
from good_egg.models import RepoMetadata, UserContributionData


class TrustGraphBuilder:
    """Build a bipartite trust graph from user contribution data."""

    MAX_PRS_PER_REPO = 20

    def __init__(self, config: GoodEggConfig) -> None:
        self.config = config

    def build_graph(
        self, user_data: UserContributionData, context_repo: str
    ) -> nx.DiGraph:
        """Build a bipartite directed graph from user contribution data.

        Nodes are prefixed: ``user:{login}`` and ``repo:{owner/name}``.
        Edges carry a ``weight`` attribute computed from recency decay,
        repository quality, and edge-type multipliers.
        """
        graph = nx.DiGraph()
        login = user_data.profile.login
        user_node = f"user:{login}"

        graph.add_node(user_node, kind="user", login=login)

        # Group PRs by repo and cap at MAX_PRS_PER_REPO
        prs_by_repo: dict[str, list] = defaultdict(list)
        for pr in user_data.merged_prs:
            prs_by_repo[pr.repo_name_with_owner].append(pr)

        for repo_name, prs in prs_by_repo.items():
            # Anti-gaming: cap PRs per repo
            prs = sorted(prs, key=lambda p: p.merged_at, reverse=True)[
                : self.MAX_PRS_PER_REPO
            ]

            repo_node = f"repo:{repo_name}"
            repo_meta = user_data.contributed_repos.get(repo_name)

            if repo_node not in graph:
                graph.add_node(
                    repo_node,
                    kind="repo",
                    name=repo_name,
                    language=repo_meta.primary_language if repo_meta else None,
                    stars=repo_meta.stargazer_count if repo_meta else 0,
                )

            quality = self._repo_quality(repo_meta)
            is_self_contrib = self._is_self_contribution(login, repo_name)

            for pr in prs:
                decay = self._recency_decay(pr.days_ago)
                weight = (
                    decay
                    * quality
                    * self.config.edge_weights.merged_pr
                )
                if is_self_contrib:
                    weight *= 0.3

                # Forward edge: user -> repo
                if graph.has_edge(user_node, repo_node):
                    graph[user_node][repo_node]["weight"] += weight
                else:
                    graph.add_edge(user_node, repo_node, weight=weight)

            # Reverse edge: repo -> user (0.3x of total forward weight)
            forward_weight = graph[user_node][repo_node]["weight"]
            graph.add_edge(repo_node, user_node, weight=forward_weight * 0.3)

        return graph

    def build_personalization_vector(
        self,
        graph: nx.DiGraph,
        context_repo: str,
        context_language: str | None,
        total_prs: int = 0,
        unique_repos: int = 0,
    ) -> dict[str, float]:
        """Build a personalization (restart) vector for PageRank.

        The context repo gets the highest weight, same-language repos get
        medium weight, everything else gets a low weight, and user nodes
        get zero.
        """
        pr_config = self.config.pagerank
        adjusted_other_weight = self._compute_adjusted_other_weight(
            pr_config, total_prs, unique_repos
        )
        context_node = f"repo:{context_repo}"
        personalization: dict[str, float] = {}

        for node in graph.nodes:
            kind = graph.nodes[node].get("kind")
            if kind == "user":
                personalization[node] = 0.0
            elif node == context_node:
                personalization[node] = pr_config.context_repo_weight
            elif (
                context_language
                and graph.nodes[node].get("language") == context_language
            ):
                personalization[node] = pr_config.same_language_weight
            else:
                personalization[node] = adjusted_other_weight

        # Normalize to sum=1
        total = sum(personalization.values())
        if total > 0:
            personalization = {k: v / total for k, v in personalization.items()}

        return personalization

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recency_decay(self, days_ago: int) -> float:
        """Exponential decay based on half-life."""
        half_life = self.config.recency.half_life_days
        if days_ago > self.config.recency.max_age_days:
            return 0.0
        return math.exp(-0.693 * days_ago / half_life)

    def _repo_quality(self, meta: RepoMetadata | None) -> float:
        """Compute a quality score for a repository."""
        if meta is None:
            return 1.0

        language_mult = self._get_language_multiplier(meta.primary_language)
        quality = math.log1p(meta.stargazer_count * language_mult)

        if meta.is_archived:
            quality *= 0.5
        if meta.is_fork:
            quality *= 0.3

        return quality

    def _get_language_multiplier(self, language: str | None) -> float:
        """Get the ecosystem size normalization multiplier for a language."""
        return self.config.language_normalization.get_multiplier(language)

    @staticmethod
    def _compute_adjusted_other_weight(
        pr_config: PageRankConfig,
        total_prs: int,
        unique_repos: int,
    ) -> float:
        """Compute adjusted other_weight based on contribution diversity and volume.

        Prolific cross-ecosystem contributors get a higher other_weight so their
        contributions outside the context language are not excessively penalized.
        """
        base = pr_config.other_weight
        if total_prs <= 0 or unique_repos <= 0:
            return base
        diversity_factor = 1.0 + pr_config.diversity_scale * min(1.0, unique_repos / 20.0)
        volume_factor = 1.0 + pr_config.volume_scale * min(
            1.0, math.log(total_prs) / math.log(100)
        )
        return base * diversity_factor * volume_factor

    @staticmethod
    def _is_self_contribution(login: str, repo_name: str) -> bool:
        """Check if a repo belongs to the user (owner/repo pattern)."""
        owner = repo_name.split("/")[0] if "/" in repo_name else ""
        return owner.lower() == login.lower()
