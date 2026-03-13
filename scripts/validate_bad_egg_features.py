"""Validate Bad Egg feature importance with correct population and temporal holdout.

Replicates PR 44 (bot-detection branch) methodology:
- Population: only authors with ≥1 merged PR before each cutoff (production-relevant)
- Temporal holdout: features computed from pre-cutoff PRs only
- LOO ablation with DeLong tests + Holm-Bonferroni correction
- Forward selection with DeLong stopping

10 candidate features (all GH-action compatible, no account_age):
  merge_rate, total_prs, career_span_days, mean_title_length,
  median_additions, median_files_changed, total_repos,
  isolation_score, hub_score, bipartite_clustering.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import duckdb
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent / "experiments" / "bot_detection"
DB_PATH = BASE / "data" / "bot_detection.duckdb"

FEATURES = [
    "merge_rate", "total_prs", "career_span_days", "mean_title_length",
    "median_additions", "median_files_changed", "total_repos",
    "isolation_score", "hub_score", "bipartite_clustering",
]
LOG_TRANSFORM = {"total_prs", "career_span_days", "median_additions", "median_files_changed"}

CUTOFFS = ["2022-07-01", "2023-01-01", "2024-01-01"]


# ---------------------------------------------------------------------------
# Data loading with temporal filtering
# ---------------------------------------------------------------------------

def get_features_before_cutoff(
    con: duckdb.DuckDBPyConnection, cutoff: str,
) -> pd.DataFrame:
    """Compute aggregate features per author from PRs before cutoff.

    Only includes authors with at least 1 merged PR before cutoff.
    """
    rows = con.execute("""
        WITH author_prs AS (
            SELECT
                author,
                state,
                created_at,
                title,
                additions,
                files_changed,
                repo
            FROM prs
            WHERE created_at < ?::TIMESTAMP
              AND author IS NOT NULL
        ),
        merged_authors AS (
            SELECT DISTINCT author
            FROM author_prs
            WHERE state = 'MERGED'
        )
        SELECT
            ma.author,
            -- merge_rate
            SUM(CASE WHEN ap.state = 'MERGED' THEN 1 ELSE 0 END)::DOUBLE
                / COUNT(*)::DOUBLE AS merge_rate,
            -- total_prs
            COUNT(*)::DOUBLE AS total_prs,
            -- career_span_days
            COALESCE(
                EXTRACT(EPOCH FROM (MAX(ap.created_at) - MIN(ap.created_at))) / 86400.0,
                0.0
            ) AS career_span_days,
            -- mean_title_length
            AVG(LENGTH(ap.title))::DOUBLE AS mean_title_length,
            -- median_additions (only from merged PRs)
            MEDIAN(CASE WHEN ap.state = 'MERGED' THEN ap.additions END)::DOUBLE
                AS median_additions,
            -- median_files_changed (only from merged PRs)
            MEDIAN(CASE WHEN ap.state = 'MERGED' THEN ap.files_changed END)::DOUBLE
                AS median_files_changed,
            -- total_repos
            COUNT(DISTINCT ap.repo)::DOUBLE AS total_repos
        FROM merged_authors ma
        JOIN author_prs ap ON ap.author = ma.author
        GROUP BY ma.author
    """, [cutoff]).fetchdf()

    return rows


def compute_per_user_isolation_before(
    con: duckdb.DuckDBPyConnection, cutoff: str,
) -> dict[str, float]:
    """Compute isolation_score per author from pre-cutoff author-repo pairs."""
    pairs = con.execute("""
        SELECT author, repo, COUNT(*) as pr_count
        FROM prs
        WHERE created_at < ?::TIMESTAMP AND author IS NOT NULL
        GROUP BY author, repo
    """, [cutoff]).fetchall()

    repo_contributors: dict[str, set[str]] = defaultdict(set)
    author_repos: dict[str, set[str]] = defaultdict(set)
    for author, repo, _ in pairs:
        repo_contributors[repo].add(author)
        author_repos[author].add(repo)

    isolation_scores: dict[str, float] = {}
    for author, repos in author_repos.items():
        if not repos:
            isolation_scores[author] = 1.0
            continue

        contributor_repo_count: dict[str, int] = defaultdict(int)
        for repo in repos:
            for c in repo_contributors[repo]:
                if c != author:
                    contributor_repo_count[c] += 1

        multi_repo = {c for c, count in contributor_repo_count.items() if count >= 2}

        isolated = 0
        for repo in repos:
            other_contribs = repo_contributors[repo] - {author}
            if not (other_contribs & multi_repo):
                isolated += 1

        isolation_scores[author] = isolated / len(repos)

    return isolation_scores


def compute_graph_features_before(
    con: duckdb.DuckDBPyConnection, cutoff: str,
) -> dict[str, dict[str, float]]:
    """Compute hub_score and bipartite_clustering from pre-cutoff bipartite graph.

    Returns {"hub_score": {author: val}, "bipartite_clustering": {author: val}}.
    """
    triples = con.execute("""
        SELECT author, repo, COUNT(*) as pr_count
        FROM prs
        WHERE created_at < ?::TIMESTAMP AND author IS NOT NULL
        GROUP BY author, repo
    """, [cutoff]).fetchall()

    if not triples:
        return {"hub_score": {}, "bipartite_clustering": {}}

    g = nx.Graph()
    authors_set = set()
    repos_set = set()
    for author, repo, count in triples:
        a_node = f"a:{author}"
        r_node = f"r:{repo}"
        g.add_edge(a_node, r_node, weight=count)
        authors_set.add(a_node)
        repos_set.add(r_node)

    # Hub score = degree centrality for author nodes
    centrality = nx.degree_centrality(g)
    hub_scores = {
        a.removeprefix("a:"): centrality.get(a, 0.0) for a in authors_set
    }

    # Bipartite clustering
    try:
        clustering = nx.bipartite.clustering(g, authors_set)
        bip_clustering = {
            a.removeprefix("a:"): clustering.get(a, 0.0) for a in authors_set
        }
    except Exception:
        bip_clustering = {a.removeprefix("a:"): 0.0 for a in authors_set}

    return {"hub_score": hub_scores, "bipartite_clustering": bip_clustering}


def load_cutoff_data(
    con: duckdb.DuckDBPyConnection, cutoff: str,
) -> pd.DataFrame:
    """Load features + labels for authors with merged PRs before cutoff."""
    print(f"\n--- Loading data for cutoff {cutoff} ---")

    df = get_features_before_cutoff(con, cutoff)
    print(f"  Authors with merged PRs before {cutoff}: {len(df)}")

    # Add isolation_score
    print("  Computing isolation_score...")
    iso = compute_per_user_isolation_before(con, cutoff)
    df["isolation_score"] = df["author"].map(iso).fillna(1.0)

    # Add graph features
    print("  Computing graph features (hub_score, bipartite_clustering)...")
    graph_feats = compute_graph_features_before(con, cutoff)
    df["hub_score"] = df["author"].map(graph_feats["hub_score"]).fillna(0.0)
    df["bipartite_clustering"] = (
        df["author"].map(graph_feats["bipartite_clustering"]).fillna(0.0)
    )

    # Join labels
    labels = con.execute(
        "SELECT login, account_status FROM authors WHERE account_status IS NOT NULL"
    ).fetchdf()
    df = df.merge(labels, left_on="author", right_on="login", how="inner")

    # Filter to labeled only
    df = df[df["account_status"].isin(["active", "suspended"])].copy()

    n_susp = (df["account_status"] == "suspended").sum()
    n_active = (df["account_status"] == "active").sum()
    print(f"  Labeled: {len(df)} ({n_susp} suspended, {n_active} active)")

    return df


# ---------------------------------------------------------------------------
# DeLong test + Holm-Bonferroni (from bot-detection stats.py)
# ---------------------------------------------------------------------------

def delong_auc_test(
    y_true: np.ndarray,
    y_scores_a: np.ndarray,
    y_scores_b: np.ndarray,
) -> dict[str, Any]:
    """Paired DeLong test for comparing two AUC-ROC values."""
    n1 = np.sum(y_true == 1)
    n0 = np.sum(y_true == 0)

    if n1 == 0 or n0 == 0:
        return {
            "auc_a": float("nan"), "auc_b": float("nan"),
            "z_statistic": float("nan"), "p_value": float("nan"),
        }

    auc_a = roc_auc_score(y_true, y_scores_a)
    auc_b = roc_auc_score(y_true, y_scores_b)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    def placement_values(
        scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        pos_scores = scores[pos_idx]
        neg_scores = scores[neg_idx]
        v10 = np.array([
            np.mean(ps > neg_scores) + 0.5 * np.mean(ps == neg_scores)
            for ps in pos_scores
        ])
        v01 = np.array([
            np.mean(pos_scores > ns) + 0.5 * np.mean(pos_scores == ns)
            for ns in neg_scores
        ])
        return v10, v01

    v10_a, v01_a = placement_values(y_scores_a)
    v10_b, v01_b = placement_values(y_scores_b)

    s10 = np.cov(np.stack([v10_a, v10_b]))
    s01 = np.cov(np.stack([v01_a, v01_b]))

    if s10.ndim == 0:
        s10 = np.array([[s10]])
    if s01.ndim == 0:
        s01 = np.array([[s01]])

    s = s10 / n1 + s01 / n0
    contrast = np.array([1, -1])
    var_diff = contrast @ s @ contrast

    if var_diff <= 0:
        return {"auc_a": auc_a, "auc_b": auc_b, "z_statistic": 0.0, "p_value": 1.0}

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p_value = 2.0 * sp_stats.norm.sf(abs(z))

    return {
        "auc_a": float(auc_a), "auc_b": float(auc_b),
        "z_statistic": float(z), "p_value": float(p_value),
    }


def holm_bonferroni(
    p_values: dict[str, float], alpha: float = 0.05,
) -> dict[str, dict[str, Any]]:
    """Apply Holm-Bonferroni correction to a set of p-values."""
    sorted_tests = sorted(p_values.items(), key=lambda x: x[1])
    m = len(sorted_tests)
    results: dict[str, dict[str, Any]] = {}

    prev_adj = 0.0
    for rank, (name, p) in enumerate(sorted_tests, start=1):
        adjusted_p = min(1.0, p * (m - rank + 1))
        adjusted_p = max(adjusted_p, prev_adj)
        prev_adj = adjusted_p
        results[name] = {
            "p_value": p, "adjusted_p": adjusted_p,
            "reject": adjusted_p <= alpha, "rank": rank,
        }

    return results


# ---------------------------------------------------------------------------
# CV helper with LOO/stratified switching
# ---------------------------------------------------------------------------

def get_oof_probabilities(
    x: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """Get out-of-fold probabilities using LOO or 5-fold stratified CV."""
    n_pos = y.sum()
    oof = np.full(len(y), np.nan)

    if n_pos < 30:
        # LOO-CV
        loo = LeaveOneOut()
        for train_idx, test_idx in loo.split(x, y):
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x[train_idx])
            x_test = scaler.transform(x[test_idx])
            model = LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=seed,
            )
            model.fit(x_train, y[train_idx])
            oof[test_idx] = model.predict_proba(x_test)[:, 1]
    else:
        # 5-fold stratified CV
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(x, y):
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x[train_idx])
            x_test = scaler.transform(x[test_idx])
            model = LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=seed,
            )
            model.fit(x_train, y[train_idx])
            oof[test_idx] = model.predict_proba(x_test)[:, 1]

    return oof


def prepare_features(df: pd.DataFrame, feature_list: list[str]) -> np.ndarray:
    """Extract and log-transform feature matrix."""
    arrays = []
    for col in feature_list:
        vals = df[col].fillna(0).values.astype(float)
        if col in LOG_TRANSFORM:
            vals = np.log1p(np.abs(vals)) * np.sign(vals)
        arrays.append(vals)
    return np.column_stack(arrays)


# ---------------------------------------------------------------------------
# LOO ablation per cutoff
# ---------------------------------------------------------------------------

def run_ablation(
    df: pd.DataFrame, features: list[str],
) -> list[dict[str, Any]]:
    """Drop each feature one at a time, compare AUC via DeLong test."""
    y = (df["account_status"] == "suspended").astype(int).values
    x_full = prepare_features(df, features)
    oof_full = get_oof_probabilities(x_full, y)
    auc_full = roc_auc_score(y, oof_full)

    results = []
    for feat in features:
        ablated = [f for f in features if f != feat]
        x_abl = prepare_features(df, ablated)
        oof_abl = get_oof_probabilities(x_abl, y)
        auc_abl = roc_auc_score(y, oof_abl)

        delong = delong_auc_test(y, oof_full, oof_abl)
        delta = auc_full - auc_abl  # positive = removing hurts

        results.append({
            "feature": feat,
            "full_auc": auc_full,
            "ablated_auc": auc_abl,
            "delta": delta,
            "p_value": delong["p_value"],
            "z_stat": delong["z_statistic"],
        })

    # Holm-Bonferroni correction
    p_vals = {r["feature"]: r["p_value"] for r in results}
    hb = holm_bonferroni(p_vals)

    for r in results:
        feat = r["feature"]
        r["adjusted_p"] = hb[feat]["adjusted_p"]
        # KEEP only if removing significantly HURTS AUC (delta > 0 and reject)
        r["verdict"] = (
            "KEEP" if hb[feat]["reject"] and r["delta"] > 0
            else "DISPENSABLE"
        )

    return results


# ---------------------------------------------------------------------------
# Forward selection per cutoff
# ---------------------------------------------------------------------------

def run_forward_selection(
    df: pd.DataFrame, features: list[str],
) -> list[dict[str, Any]]:
    """Greedy forward selection with DeLong stopping."""
    y = (df["account_status"] == "suspended").astype(int).values
    selected: list[str] = []
    trajectory: list[dict[str, Any]] = []
    prev_oof: np.ndarray | None = None

    for step in range(len(features)):
        best_feat = None
        best_auc = -1.0
        best_oof = None

        for candidate in features:
            if candidate in selected:
                continue
            trial = selected + [candidate]
            x_trial = prepare_features(df, trial)
            oof_trial = get_oof_probabilities(x_trial, y)
            auc_trial = roc_auc_score(y, oof_trial)

            if auc_trial > best_auc:
                best_auc = auc_trial
                best_feat = candidate
                best_oof = oof_trial

        if best_feat is None:
            break

        # DeLong test vs previous step
        p_value = None
        if prev_oof is not None and best_oof is not None:
            delong = delong_auc_test(y, best_oof, prev_oof)
            p_value = delong["p_value"]

        selected.append(best_feat)
        prev_oof = best_oof

        trajectory.append({
            "step": step + 1,
            "feature": best_feat,
            "auc": best_auc,
            "p_value": p_value,
            "selected": list(selected),
        })

    return trajectory


# ---------------------------------------------------------------------------
# Aggregation and recommendation
# ---------------------------------------------------------------------------

def aggregate_ablation(
    all_results: dict[str, list[dict[str, Any]]],
) -> None:
    """Print aggregated ablation results across cutoffs."""
    print("\n" + "=" * 80)
    print("AGGREGATED LOO ABLATION RESULTS")
    print("=" * 80)

    feature_verdicts: dict[str, list[str]] = defaultdict(list)
    feature_deltas: dict[str, list[float]] = defaultdict(list)

    for cutoff, results in all_results.items():
        print(f"\n--- Cutoff: {cutoff} ---")
        print(f"  Full AUC: {results[0]['full_auc']:.4f}")
        print(f"  {'Feature':<25s} {'Ablated AUC':>11s} {'Delta':>8s} "
              f"{'p-value':>10s} {'adj-p':>10s} {'Verdict'}")
        for r in sorted(results, key=lambda x: x["delta"], reverse=True):
            print(f"  {r['feature']:<25s} {r['ablated_auc']:>11.4f} "
                  f"{r['delta']:>+8.4f} {r['p_value']:>10.4f} "
                  f"{r['adjusted_p']:>10.4f} {r['verdict']}")
            feature_verdicts[r["feature"]].append(r["verdict"])
            feature_deltas[r["feature"]].append(r["delta"])

    print("\n--- Summary across cutoffs ---")
    print(f"  {'Feature':<25s} {'Mean Delta':>10s} {'KEEP count':>10s} Verdict")
    for feat in FEATURES:
        verdicts = feature_verdicts.get(feat, [])
        keep_count = sum(1 for v in verdicts if v == "KEEP")
        mean_delta = np.mean(feature_deltas.get(feat, [0.0]))
        overall = "KEEP" if keep_count >= 2 else "DISPENSABLE"
        print(f"  {feat:<25s} {mean_delta:>+10.4f} {keep_count:>10d}/{len(verdicts)} "
              f"{overall}")


def aggregate_forward_selection(
    all_trajectories: dict[str, list[dict[str, Any]]],
) -> None:
    """Print aggregated forward selection results."""
    print("\n" + "=" * 80)
    print("AGGREGATED FORWARD SELECTION RESULTS")
    print("=" * 80)

    for cutoff, trajectory in all_trajectories.items():
        print(f"\n--- Cutoff: {cutoff} ---")
        print(f"  {'Step':>4s} {'Feature':<25s} {'AUC':>8s} {'p-value':>10s}")
        for t in trajectory:
            p_str = f"{t['p_value']:.4f}" if t["p_value"] is not None else "---"
            print(f"  {t['step']:>4d} {t['feature']:<25s} {t['auc']:>8.4f} {p_str:>10s}")


def refit_recommended(
    con: duckdb.DuckDBPyConnection,
    recommended: list[str],
) -> None:
    """Refit model on all data with recommended features and print config."""
    print("\n" + "=" * 80)
    print(f"REFIT WITH RECOMMENDED FEATURES: {recommended}")
    print("=" * 80)

    # Use latest cutoff for refit
    df = load_cutoff_data(con, "2026-01-01")
    y = (df["account_status"] == "suspended").astype(int).values
    x = prepare_features(df, recommended)

    # CV AUC
    oof = get_oof_probabilities(x, y)
    cv_auc = roc_auc_score(y, oof)
    print(f"\n  CV AUC (all data): {cv_auc:.4f}")

    # Final fit
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42,
    )
    model.fit(x_scaled, y)

    # Convert to unscaled coefficients
    means = scaler.mean_
    stds = scaler.scale_
    coefs_scaled = model.coef_[0]
    intercept_scaled = model.intercept_[0]
    coefs_raw = coefs_scaled / stds
    intercept_raw = intercept_scaled - np.sum(coefs_scaled * means / stds)

    print("\n  Config update:")
    print(f"    intercept: float = {intercept_raw:.4f}")
    for feat, w in zip(recommended, coefs_raw, strict=True):
        key = feat
        if feat in LOG_TRANSFORM:
            key = feat
        print(f"    {key}_weight: float = {w:.4f}")

    # Threshold analysis
    logit = np.full(len(df), intercept_raw)
    for feat, w in zip(recommended, coefs_raw, strict=True):
        vals = df[feat].fillna(0).values.astype(float)
        if feat in LOG_TRANSFORM:
            vals = np.log1p(np.abs(vals)) * np.sign(vals)
        logit += w * vals
    probs = 1.0 / (1.0 + np.exp(-logit))

    print(f"\n  Probability range: [{probs.min():.4f}, {probs.max():.4f}]")
    print(f"  Mean (suspended): {probs[y == 1].mean():.4f}")
    print(f"  Mean (active):    {probs[y == 0].mean():.4f}")
    print()
    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        flagged = probs >= threshold
        n_flagged = int(flagged.sum())
        n_true_pos = int((flagged & (y == 1)).sum())
        precision = n_true_pos / n_flagged if n_flagged > 0 else 0
        recall = n_true_pos / y.sum() if y.sum() > 0 else 0
        print(f"    t={threshold:.2f}: flagged={n_flagged:5d} "
              f"({100 * n_flagged / len(y):.1f}%), "
              f"prec={precision:.3f}, recall={recall:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)

    # Population overview
    print("=" * 80)
    print("POPULATION OVERVIEW")
    print("=" * 80)
    total_authors = con.execute("SELECT COUNT(*) FROM authors").fetchone()[0]
    checked = con.execute(
        "SELECT COUNT(*) FROM authors WHERE account_status IS NOT NULL"
    ).fetchone()[0]
    print(f"Total authors: {total_authors}, Checked: {checked}")

    status_rows = con.execute(
        "SELECT account_status, COUNT(*) FROM authors GROUP BY account_status "
        "ORDER BY COUNT(*) DESC"
    ).fetchall()
    for status, count in status_rows:
        print(f"  {status}: {count}")

    susp_merged = con.execute("""
        SELECT COUNT(DISTINCT a.login)
        FROM authors a JOIN prs p ON a.login = p.author
        WHERE a.account_status = 'suspended' AND p.state = 'MERGED'
    """).fetchone()[0]
    print(f"\nSuspended authors with merged PRs: {susp_merged}")

    # Run per-cutoff analyses
    all_ablation: dict[str, list[dict[str, Any]]] = {}
    all_forward: dict[str, list[dict[str, Any]]] = {}

    for cutoff in CUTOFFS:
        df = load_cutoff_data(con, cutoff)
        n_susp = (df["account_status"] == "suspended").sum()
        cv_method = "LOO" if n_susp < 30 else "5-fold stratified"
        print(f"\n  CV method: {cv_method} (n_suspended={n_susp})")

        print(f"\n  Running LOO ablation for {cutoff}...")
        ablation = run_ablation(df, FEATURES)
        all_ablation[cutoff] = ablation

        print(f"  Running forward selection for {cutoff}...")
        forward = run_forward_selection(df, FEATURES)
        all_forward[cutoff] = forward

    # Aggregate results
    aggregate_ablation(all_ablation)
    aggregate_forward_selection(all_forward)

    # Compare full 10f vs recommended subset
    print("\n" + "=" * 80)
    print("FULL vs SUBSET COMPARISON")
    print("=" * 80)
    for cutoff in CUTOFFS:
        df = load_cutoff_data(con, cutoff)
        y = (df["account_status"] == "suspended").astype(int).values

        x_full = prepare_features(df, FEATURES)
        oof_full = get_oof_probabilities(x_full, y)
        auc_full = roc_auc_score(y, oof_full)

        # Try the PR 44 recommended 3-feature set
        subset_3 = ["merge_rate", "median_additions", "isolation_score"]
        x_3 = prepare_features(df, subset_3)
        oof_3 = get_oof_probabilities(x_3, y)
        auc_3 = roc_auc_score(y, oof_3)

        delong = delong_auc_test(y, oof_full, oof_3)

        print(f"\n  Cutoff {cutoff}: 10f AUC={auc_full:.4f}, "
              f"3f AUC={auc_3:.4f}, delta={auc_full - auc_3:+.4f}, "
              f"p={delong['p_value']:.4f}")

    # Refit with recommended features
    refit_recommended(con, ["merge_rate", "median_additions", "isolation_score"])

    con.close()


if __name__ == "__main__":
    main()
