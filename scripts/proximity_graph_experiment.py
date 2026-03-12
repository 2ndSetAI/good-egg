"""Graph-based proximity experiment for suspension detection.

Tests H2 (graph-based proximity captures structural signal) via:
  - Shared-repo Jaccard proximity (max and mean-k variants)
  - Personalized PageRank from suspended seeds

Strategies A, B, C as in the k-NN experiment.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import duckdb
import networkx as nx
import numpy as np
import pandas as pd
from proximity_common import (
    CUTOFFS,
    DB_PATH,
    RESULTS_DIR,
    compute_metrics,
    load_all_time_features,
    load_temporal_features,
    merge_rate_baseline,
    run_suspended_cv,
)

# ---------------------------------------------------------------------------
# Graph proximity methods
# ---------------------------------------------------------------------------

def build_author_repo_data(
    con: duckdb.DuckDBPyConnection,
    cutoff: str | None = None,
) -> tuple[dict[str, set[str]], nx.Graph]:
    """Build author-repo mappings and bipartite graph.

    Returns (author_repos dict, bipartite Graph).
    """
    if cutoff:
        rows = con.execute("""
            SELECT author, repo, COUNT(*) as pr_count
            FROM prs
            WHERE created_at < ?::TIMESTAMP AND author IS NOT NULL
            GROUP BY author, repo
        """, [cutoff]).fetchall()
    else:
        rows = con.execute("""
            SELECT author, repo, COUNT(*) as pr_count
            FROM prs WHERE author IS NOT NULL
            GROUP BY author, repo
        """).fetchall()

    author_repos: dict[str, set[str]] = defaultdict(set)
    g = nx.Graph()

    for author, repo, count in rows:
        author_repos[author].add(repo)
        a_node = f"a:{author}"
        r_node = f"r:{repo}"
        g.add_edge(a_node, r_node, weight=count)

    return author_repos, g


def jaccard_max_proximity(
    seed_repos: dict[str, set[str]],
    eval_repos: dict[str, set[str]],
) -> dict[str, float]:
    """Max Jaccard similarity between eval author's repos and any seed's repos."""
    scores: dict[str, float] = {}
    seed_list = list(seed_repos.values())

    for author, repos in eval_repos.items():
        if not repos:
            scores[author] = 0.0
            continue
        max_j = 0.0
        for s_repos in seed_list:
            if not s_repos:
                continue
            inter = len(repos & s_repos)
            union = len(repos | s_repos)
            if union > 0:
                max_j = max(max_j, inter / union)
        scores[author] = max_j

    return scores


def jaccard_mean_k_proximity(
    seed_repos: dict[str, set[str]],
    eval_repos: dict[str, set[str]],
    k: int = 5,
) -> dict[str, float]:
    """Mean of top-k Jaccard similarities to seeds."""
    scores: dict[str, float] = {}
    seed_list = list(seed_repos.values())

    for author, repos in eval_repos.items():
        if not repos:
            scores[author] = 0.0
            continue
        jaccards = []
        for s_repos in seed_list:
            if not s_repos:
                continue
            inter = len(repos & s_repos)
            union = len(repos | s_repos)
            jaccards.append(inter / union if union > 0 else 0.0)

        if not jaccards:
            scores[author] = 0.0
            continue

        jaccards.sort(reverse=True)
        top_k = jaccards[:min(k, len(jaccards))]
        scores[author] = float(np.mean(top_k))

    return scores


def ppr_proximity(
    graph: nx.Graph,
    seed_authors: list[str],
    alpha: float = 0.85,
) -> dict[str, float]:
    """Personalized PageRank with restart on suspended seed nodes.

    Returns PPR value at each author node (higher = closer to seeds).
    """
    seed_nodes = [f"a:{a}" for a in seed_authors if f"a:{a}" in graph]
    if not seed_nodes:
        return {}

    personalization = {node: 1.0 / len(seed_nodes) for node in seed_nodes}

    try:
        ppr = nx.pagerank(graph, alpha=alpha, personalization=personalization)
    except nx.PowerIterationFailedConvergence:
        ppr = nx.pagerank(
            graph, alpha=alpha, personalization=personalization, max_iter=200,
        )

    # Extract author scores
    scores: dict[str, float] = {}
    for node, val in ppr.items():
        if node.startswith("a:"):
            scores[node.removeprefix("a:")] = float(val)

    return scores


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def run_strategy_a(
    df: pd.DataFrame,
    author_repos: dict[str, set[str]],
    graph: nx.Graph,
) -> dict[str, Any]:
    """Strategy A: Discovery-order holdout for graph methods."""
    print("\n=== Strategy A: Discovery-Order Holdout (Graph) ===")
    results: dict[str, Any] = {}

    susp_mask = df["account_status"] == "suspended"
    orig_mask = df["is_original_discovery"]

    seed_authors = df.loc[orig_mask, "author" if "author" in df.columns
                          else "login"].tolist()
    test_df = df[~orig_mask | ~susp_mask].copy()

    author_col = "author" if "author" in test_df.columns else "login"
    y_test = (test_df["account_status"] == "suspended").astype(int).values
    test_authors = test_df[author_col].tolist()

    n_seeds = len(seed_authors)
    print(f"  Seeds: {n_seeds}, Test: {y_test.sum()} suspended, "
          f"{(1 - y_test).sum()} active")

    # Jaccard max
    seed_repo_map = {a: author_repos.get(a, set()) for a in seed_authors}
    test_repo_map = {a: author_repos.get(a, set()) for a in test_authors}

    jm_scores = jaccard_max_proximity(seed_repo_map, test_repo_map)
    scores_arr = np.array([jm_scores.get(a, 0.0) for a in test_authors])
    m = compute_metrics(y_test, scores_arr)
    results["jaccard_max"] = m
    print(f"  jaccard_max: AUC={m.get('auc_roc', float('nan')):.4f}")

    # Jaccard mean-k5
    jmk_scores = jaccard_mean_k_proximity(seed_repo_map, test_repo_map, k=5)
    scores_arr = np.array([jmk_scores.get(a, 0.0) for a in test_authors])
    m = compute_metrics(y_test, scores_arr)
    results["jaccard_mean_k5"] = m
    print(f"  jaccard_mean_k5: AUC={m.get('auc_roc', float('nan')):.4f}")

    # PPR
    ppr_scores = ppr_proximity(graph, seed_authors)
    scores_arr = np.array([ppr_scores.get(a, 0.0) for a in test_authors])
    m = compute_metrics(y_test, scores_arr)
    results["ppr"] = m
    print(f"  ppr: AUC={m.get('auc_roc', float('nan')):.4f}")

    # Baselines
    y_mr, s_mr = merge_rate_baseline(test_df)
    results["baseline_merge_rate"] = compute_metrics(y_mr, s_mr)

    return results


def run_strategy_b(
    df: pd.DataFrame,
    author_repos: dict[str, set[str]],
    graph: nx.Graph,
    label: str = "merged-PR",
) -> dict[str, Any]:
    """Strategy B: Suspended-only CV for graph methods."""
    print(f"\n=== Strategy B: CV ({label}) (Graph) ===")
    results: dict[str, Any] = {}

    author_col = "author" if "author" in df.columns else "login"

    # Jaccard max via CV
    def jaccard_max_fn(
        seed_df: pd.DataFrame,
        eval_df: pd.DataFrame,
    ) -> np.ndarray:
        s_authors = seed_df[author_col].tolist()
        e_authors = eval_df[author_col].tolist()
        s_repos = {a: author_repos.get(a, set()) for a in s_authors}
        e_repos = {a: author_repos.get(a, set()) for a in e_authors}
        jm = jaccard_max_proximity(s_repos, e_repos)
        return np.array([jm.get(a, 0.0) for a in e_authors])

    y, scores = run_suspended_cv(df, jaccard_max_fn)
    m = compute_metrics(y, scores)
    results["jaccard_max"] = m
    print(f"  jaccard_max: AUC={m.get('auc_roc', float('nan')):.4f}")

    # Jaccard mean-k5 via CV
    def jaccard_mean_k5_fn(
        seed_df: pd.DataFrame,
        eval_df: pd.DataFrame,
    ) -> np.ndarray:
        s_authors = seed_df[author_col].tolist()
        e_authors = eval_df[author_col].tolist()
        s_repos = {a: author_repos.get(a, set()) for a in s_authors}
        e_repos = {a: author_repos.get(a, set()) for a in e_authors}
        jm = jaccard_mean_k_proximity(s_repos, e_repos, k=5)
        return np.array([jm.get(a, 0.0) for a in e_authors])

    y, scores = run_suspended_cv(df, jaccard_mean_k5_fn)
    m = compute_metrics(y, scores)
    results["jaccard_mean_k5"] = m
    print(f"  jaccard_mean_k5: AUC={m.get('auc_roc', float('nan')):.4f}")

    # PPR via CV
    def ppr_fn(
        seed_df: pd.DataFrame,
        eval_df: pd.DataFrame,
    ) -> np.ndarray:
        s_authors = seed_df[author_col].tolist()
        e_authors = eval_df[author_col].tolist()
        ppr_s = ppr_proximity(graph, s_authors)
        return np.array([ppr_s.get(a, 0.0) for a in e_authors])

    y, scores = run_suspended_cv(df, ppr_fn)
    m = compute_metrics(y, scores)
    results["ppr"] = m
    print(f"  ppr: AUC={m.get('auc_roc', float('nan')):.4f}")

    # Baselines
    y_mr, s_mr = merge_rate_baseline(df)
    results["baseline_merge_rate"] = compute_metrics(y_mr, s_mr)

    return results


def run_strategy_c(
    cutoffs: list[str],
) -> dict[str, Any]:
    """Strategy C: Temporal holdout for graph methods."""
    print("\n=== Strategy C: Temporal Holdout (Graph) ===")
    results: dict[str, Any] = {}

    for cutoff in cutoffs:
        df = load_temporal_features(cutoff)
        n_susp = (df["account_status"] == "suspended").sum()
        if n_susp < 5:
            print(f"  Skipping {cutoff}: only {n_susp} suspended")
            continue

        con = duckdb.connect(str(DB_PATH), read_only=True)
        author_repos, graph = build_author_repo_data(con, cutoff)
        con.close()

        author_col = "author" if "author" in df.columns else "login"
        cutoff_results: dict[str, Any] = {"n_suspended": int(n_susp)}

        # Jaccard max via CV
        def jaccard_max_fn(
            seed_df: pd.DataFrame,
            eval_df: pd.DataFrame,
            _ar: dict[str, set[str]] = author_repos,
            _ac: str = author_col,
        ) -> np.ndarray:
            s_authors = seed_df[_ac].tolist()
            e_authors = eval_df[_ac].tolist()
            s_repos = {a: _ar.get(a, set()) for a in s_authors}
            e_repos = {a: _ar.get(a, set()) for a in e_authors}
            jm = jaccard_max_proximity(s_repos, e_repos)
            return np.array([jm.get(a, 0.0) for a in e_authors])

        y, scores = run_suspended_cv(df, jaccard_max_fn)
        m = compute_metrics(y, scores)
        cutoff_results["jaccard_max"] = m
        print(f"  {cutoff} jaccard_max: "
              f"AUC={m.get('auc_roc', float('nan')):.4f}")

        # PPR via CV
        def ppr_fn(
            seed_df: pd.DataFrame,
            eval_df: pd.DataFrame,
            _g: nx.Graph = graph,
            _ac: str = author_col,
        ) -> np.ndarray:
            s_authors = seed_df[_ac].tolist()
            e_authors = eval_df[_ac].tolist()
            ppr_s = ppr_proximity(_g, s_authors)
            return np.array([ppr_s.get(a, 0.0) for a in e_authors])

        y, scores = run_suspended_cv(df, ppr_fn)
        m = compute_metrics(y, scores)
        cutoff_results["ppr"] = m
        print(f"  {cutoff} ppr: AUC={m.get('auc_roc', float('nan')):.4f}")

        # Baseline
        y_mr, s_mr = merge_rate_baseline(df)
        cutoff_results["baseline_merge_rate"] = compute_metrics(y_mr, s_mr)

        results[cutoff] = cutoff_results

    return results


def main() -> None:
    all_results: dict[str, Any] = {}

    # Load data
    df_merged = load_all_time_features(merged_pr_only=True)

    # The parquet uses 'login', temporal uses 'author' — normalize
    if "login" in df_merged.columns and "author" not in df_merged.columns:
        df_merged = df_merged.rename(columns={"login": "author"})

    con = duckdb.connect(str(DB_PATH), read_only=True)
    author_repos, graph = build_author_repo_data(con)
    con.close()

    print(f"Graph: {graph.number_of_nodes()} nodes, "
          f"{graph.number_of_edges()} edges")

    # Strategy A
    all_results["strategy_a"] = run_strategy_a(
        df_merged, author_repos, graph,
    )

    # Strategy B: merged-PR
    all_results["strategy_b_merged"] = run_strategy_b(
        df_merged, author_repos, graph, label="merged-PR",
    )

    # Strategy C: temporal
    all_results["strategy_c"] = run_strategy_c(CUTOFFS)

    # Save results
    output_path = RESULTS_DIR / "graph_results.json"

    def _default(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_default)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
