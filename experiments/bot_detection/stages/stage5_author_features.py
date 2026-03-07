from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from experiments.bot_detection.cache import BotDetectionDB
from experiments.bot_detection.checkpoint import write_stage_checkpoint
from experiments.bot_detection.models import StudyConfig

logger = logging.getLogger(__name__)


# ----------------------------------------------------------
# H10: Network features from bipartite author-repo graph
# ----------------------------------------------------------


def _build_bipartite_graph(
    pairs: list[tuple[str, str, int]],
) -> tuple[nx.Graph, set[str], set[str]]:
    """Build a bipartite graph from (author, repo, pr_count) triples.

    Returns (graph, author_nodes, repo_nodes).
    """
    graph = nx.Graph()
    author_nodes: set[str] = set()
    repo_nodes: set[str] = set()

    for author, repo, pr_count in pairs:
        a_node = f"a:{author}"
        r_node = f"r:{repo}"
        author_nodes.add(a_node)
        repo_nodes.add(r_node)
        graph.add_node(a_node, bipartite=0)
        graph.add_node(r_node, bipartite=1)
        graph.add_edge(a_node, r_node, weight=pr_count)

    return graph, author_nodes, repo_nodes


def _compute_network_features(
    pairs: list[tuple[str, str, int]],
) -> pd.DataFrame:
    """Compute H10 network features for all authors.

    Features:
    - hub_score: HITS hub score from bipartite graph
    - bipartite_clustering: clustering coefficient in bipartite graph
    - author_projection_degree: degree in author-author projection
    - repo_diversity_entropy: Shannon entropy of PR distribution across repos
    - connected_component_size: size of author's component in projection
    - mean_repo_popularity: mean total PR count of repos the author touched
    - isolation_score: fraction of author's repos where no other multi-repo author contributes
    """
    if not pairs:
        return pd.DataFrame(columns=[
            "login", "hub_score", "bipartite_clustering",
            "author_projection_degree", "repo_diversity_entropy",
            "connected_component_size", "mean_repo_popularity",
            "isolation_score",
        ])

    graph, author_nodes, repo_nodes = _build_bipartite_graph(pairs)

    # Hub scores: use degree centrality on the bipartite graph.
    # HITS is numerically unstable on bipartite graphs (produces negative/inf
    # values on small or highly symmetric subgraphs). Degree centrality captures
    # the same signal (authors touching many repos score higher) and is stable.
    hubs = nx.degree_centrality(graph)

    # Bipartite clustering
    try:
        bip_clustering = nx.bipartite.clustering(graph, author_nodes)
    except Exception:
        logger.warning("Bipartite clustering failed, using zeros")
        bip_clustering = {n: 0.0 for n in author_nodes}

    # Author projection (two authors share an edge if they contributed to the same repo)
    try:
        proj = nx.bipartite.projected_graph(graph, author_nodes, multigraph=False)
    except Exception:
        logger.warning("Bipartite projection failed, using empty graph")
        proj = nx.Graph()

    # Connected components in projection
    component_map: dict[str, int] = {}
    for component in nx.connected_components(proj):
        size = len(component)
        for node in component:
            component_map[node] = size

    # Build per-author lookup: author -> [(repo, pr_count), ...]
    author_repos: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for author, repo, pr_count in pairs:
        author_repos[author].append((repo, pr_count))

    # Repo total PR counts (for mean_repo_popularity)
    repo_total_prs: dict[str, int] = defaultdict(int)
    for _author, repo, pr_count in pairs:
        repo_total_prs[repo] += pr_count

    # Multi-repo authors (for isolation_score)
    author_repo_sets: dict[str, set[str]] = defaultdict(set)
    for author, repo, _count in pairs:
        author_repo_sets[author].add(repo)
    multi_repo_authors = {a for a, repos in author_repo_sets.items() if len(repos) >= 2}

    # Repos that have at least one multi-repo author OTHER than a given author
    repo_multi_authors: dict[str, set[str]] = defaultdict(set)
    for author in multi_repo_authors:
        for repo in author_repo_sets[author]:
            repo_multi_authors[repo].add(author)

    rows: list[dict[str, object]] = []
    for author, repo_list in author_repos.items():
        a_node = f"a:{author}"

        # Hub score
        hub_score = hubs.get(a_node, 0.0)

        # Bipartite clustering
        bip_clust = bip_clustering.get(a_node, 0.0)

        # Projection degree
        proj_degree = proj.degree(a_node) if proj.has_node(a_node) else 0

        # Repo diversity entropy: Shannon entropy of PR count distribution
        counts = [c for _, c in repo_list]
        total = sum(counts)
        if total > 0 and len(counts) > 1:
            probs = [c / total for c in counts]
            entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        else:
            entropy = 0.0

        # Connected component size
        comp_size = component_map.get(a_node, 1)

        # Mean repo popularity
        popularities = [repo_total_prs[repo] for repo, _ in repo_list]
        mean_pop = float(np.mean(popularities)) if popularities else 0.0

        # Isolation score: fraction of author's repos where no OTHER multi-repo
        # author contributes
        author_repo_names = {repo for repo, _ in repo_list}
        if author_repo_names:
            isolated_count = 0
            for repo in author_repo_names:
                other_multirepo = repo_multi_authors.get(repo, set()) - {author}
                if not other_multirepo:
                    isolated_count += 1
            isolation = isolated_count / len(author_repo_names)
        else:
            isolation = 1.0

        rows.append({
            "login": author,
            "hub_score": float(hub_score),
            "bipartite_clustering": float(bip_clust),
            "author_projection_degree": int(proj_degree),
            "repo_diversity_entropy": float(entropy),
            "connected_component_size": int(comp_size),
            "mean_repo_popularity": float(mean_pop),
            "isolation_score": float(isolation),
        })

    return pd.DataFrame(rows)


# ----------------------------------------------------------
# Main entry point
# ----------------------------------------------------------


def run_stage5(
    base_dir: Path,
    config: StudyConfig,
    cutoff: datetime | None = None,
    output_dir: Path | None = None,
) -> None:
    """Compute author-level features (H8 aggregates + H10 network).

    Steps:
    1. Query aggregate stats per author (H8)
    2. Query author profile fields (account_age, followers, etc.)
    3. Build bipartite author-repo graph and compute network features (H10)
    4. Merge all features and write to parquet
    """
    db_path = base_dir / config.paths.get("local_db", "data/bot_detection.duckdb")
    features_dir = base_dir / config.paths.get("features", "data/features")
    features_dir.mkdir(parents=True, exist_ok=True)

    with BotDetectionDB(db_path) as db:
        # H8: Author aggregate stats
        logger.info("Computing H8 author aggregate stats...")
        if cutoff is not None:
            agg_df = db.get_author_aggregate_stats_before(cutoff)
        else:
            agg_df = db.get_author_aggregate_stats()
        logger.info("  %d authors with aggregate stats", len(agg_df))

        # Author profile fields
        logger.info("Fetching author profile fields...")
        profile_df = db.get_author_profile_fields()

        # Compute account_age_days from account_created_at
        if not profile_df.empty and "account_created_at" in profile_df.columns:
            ref_time = pd.Timestamp(cutoff) if cutoff is not None else pd.Timestamp.now()
            profile_df["account_age_days"] = profile_df["account_created_at"].apply(
                lambda x: (ref_time - x).total_seconds() / 86400.0 if pd.notna(x) else None,
            )
        else:
            profile_df["account_age_days"] = None

        profile_df = profile_df[["login", "account_age_days", "followers",
                                  "public_repos", "account_status"]]

        # H10: Network features
        logger.info("Computing H10 network features...")
        if cutoff is not None:
            pairs = db.get_all_author_repo_pairs_before(cutoff)
        else:
            pairs = db.get_all_author_repo_pairs()
        logger.info("  %d author-repo pairs", len(pairs))

    network_df = _compute_network_features(pairs)
    logger.info("  %d authors with network features", len(network_df))

    # Merge: agg_df LEFT JOIN profile_df LEFT JOIN network_df
    merged = agg_df.merge(profile_df, on="login", how="left")
    merged = merged.merge(network_df, on="login", how="left")

    # Write parquet
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "author_features.parquet"
    else:
        output_path = features_dir / "author_features.parquet"
    merged.to_parquet(output_path, index=False)
    logger.info("Wrote %d author feature rows to %s", len(merged), output_path)

    # Checkpoint
    write_stage_checkpoint(
        db_path.parent,
        "stage5",
        {"authors": len(merged)},
        details={
            "output_path": str(output_path),
            "columns": list(merged.columns),
            "h8_columns": list(agg_df.columns),
            "h10_columns": list(network_df.columns),
        },
    )
