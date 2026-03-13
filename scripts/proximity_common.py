"""Shared utilities for proximity-based suspension detection experiments.

Provides data loading, feature preparation, CV engine, metrics, and
statistical testing functions used by all proximity experiment scripts.
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent / "experiments" / "bot_detection"
DB_PATH = BASE / "data" / "bot_detection.duckdb"
PARQUET_PATH = BASE / "data" / "features" / "author_features.parquet"
RESULTS_DIR = BASE / "proximity_results"

# Feature sets
F10 = [
    "merge_rate", "total_prs", "career_span_days", "mean_title_length",
    "median_additions", "median_files_changed", "total_repos",
    "isolation_score", "hub_score", "bipartite_clustering",
]

F16 = F10 + [
    "rejection_rate", "hour_entropy", "empty_body_rate",
    "title_spam_score", "weekend_ratio", "prs_per_active_day",
]

F16_NO_MR = [f for f in F16 if f not in ("merge_rate", "rejection_rate")]

LOG_TRANSFORM = {"total_prs", "career_span_days", "median_additions", "median_files_changed"}

CUTOFFS = ["2022-07-01", "2023-01-01", "2024-01-01"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_time_features(
    merged_pr_only: bool = True,
) -> pd.DataFrame:
    """Load parquet features with updated DuckDB labels.

    Adds an 'is_original_discovery' column for Strategy A holdout.
    """
    pq = pd.read_parquet(PARQUET_PATH)
    original_suspended = set(
        pq.loc[pq["account_status"] == "suspended", "login"].tolist()
    )

    # Get updated labels from DuckDB
    con = duckdb.connect(str(DB_PATH), read_only=True)
    labels = con.execute(
        "SELECT login, account_status FROM authors "
        "WHERE account_status IN ('active', 'suspended')"
    ).fetchdf()

    if merged_pr_only:
        merged_authors = set(r[0] for r in con.execute(
            "SELECT DISTINCT author FROM prs "
            "WHERE state = 'MERGED' AND author IS NOT NULL"
        ).fetchall())
    con.close()

    # Join parquet features with updated labels
    df = pq.drop(columns=["account_status"], errors="ignore")
    df = df.merge(labels, on="login", how="inner")

    if merged_pr_only:
        df = df[df["login"].isin(merged_authors)].copy()

    # Mark discovery order
    df["is_original_discovery"] = (
        df["login"].isin(original_suspended)
        & (df["account_status"] == "suspended")
    )

    n_susp = (df["account_status"] == "suspended").sum()
    n_active = (df["account_status"] == "active").sum()
    n_orig = df["is_original_discovery"].sum()
    pop = "merged-PR" if merged_pr_only else "all"
    print(f"Loaded {pop}: {len(df)} authors "
          f"({n_susp} suspended [{n_orig} original], {n_active} active)")

    return df


def load_temporal_features(
    cutoff: str,
) -> pd.DataFrame:
    """Compute F16 features from DuckDB for authors with merged PRs before cutoff.

    Returns DataFrame with all 16 features + author + account_status +
    is_original_discovery.
    """
    con = duckdb.connect(str(DB_PATH), read_only=True)

    # Original 323 suspended from parquet
    pq = pd.read_parquet(PARQUET_PATH, columns=["login", "account_status"])
    original_suspended = set(
        pq.loc[pq["account_status"] == "suspended", "login"].tolist()
    )

    print(f"\n--- Loading temporal features for cutoff {cutoff} ---")

    # Core F10 features (8 SQL-native + 2 graph features)
    df = _get_features_before_cutoff(con, cutoff)
    print(f"  Authors with merged PRs before {cutoff}: {len(df)}")

    # Isolation score
    print("  Computing isolation_score...")
    iso = _compute_isolation_before(con, cutoff)
    df["isolation_score"] = df["author"].map(iso).fillna(1.0)

    # Graph features
    print("  Computing graph features...")
    graph_feats = _compute_graph_features_before(con, cutoff)
    df["hub_score"] = df["author"].map(graph_feats["hub_score"]).fillna(0.0)
    df["bipartite_clustering"] = (
        df["author"].map(graph_feats["bipartite_clustering"]).fillna(0.0)
    )

    # Extended F16 features
    print("  Computing extended features (F16)...")
    ext = _get_extended_features_before(con, cutoff)
    df = df.merge(ext, on="author", how="left")
    for col in ["rejection_rate", "hour_entropy", "empty_body_rate",
                 "title_spam_score", "weekend_ratio", "prs_per_active_day"]:
        df[col] = df[col].fillna(0.0)

    # Labels
    labels = con.execute(
        "SELECT login, account_status FROM authors "
        "WHERE account_status IN ('active', 'suspended')"
    ).fetchdf()
    con.close()

    df = df.merge(labels, left_on="author", right_on="login", how="inner")
    df = df[df["account_status"].isin(["active", "suspended"])].copy()

    df["is_original_discovery"] = (
        df["author"].isin(original_suspended)
        & (df["account_status"] == "suspended")
    )

    n_susp = (df["account_status"] == "suspended").sum()
    n_active = (df["account_status"] == "active").sum()
    print(f"  Labeled: {len(df)} ({n_susp} suspended, {n_active} active)")

    return df


def _get_features_before_cutoff(
    con: duckdb.DuckDBPyConnection, cutoff: str,
) -> pd.DataFrame:
    """Compute 8 SQL-native features per author from PRs before cutoff."""
    return con.execute("""
        WITH author_prs AS (
            SELECT author, state, created_at, title, additions,
                   files_changed, repo
            FROM prs
            WHERE created_at < ?::TIMESTAMP AND author IS NOT NULL
        ),
        merged_authors AS (
            SELECT DISTINCT author FROM author_prs WHERE state = 'MERGED'
        )
        SELECT
            ma.author,
            SUM(CASE WHEN ap.state = 'MERGED' THEN 1 ELSE 0 END)::DOUBLE
                / COUNT(*)::DOUBLE AS merge_rate,
            COUNT(*)::DOUBLE AS total_prs,
            COALESCE(
                EXTRACT(EPOCH FROM (MAX(ap.created_at) - MIN(ap.created_at)))
                / 86400.0, 0.0
            ) AS career_span_days,
            AVG(LENGTH(ap.title))::DOUBLE AS mean_title_length,
            MEDIAN(CASE WHEN ap.state = 'MERGED' THEN ap.additions END)::DOUBLE
                AS median_additions,
            MEDIAN(CASE WHEN ap.state = 'MERGED' THEN ap.files_changed END)::DOUBLE
                AS median_files_changed,
            COUNT(DISTINCT ap.repo)::DOUBLE AS total_repos
        FROM merged_authors ma
        JOIN author_prs ap ON ap.author = ma.author
        GROUP BY ma.author
    """, [cutoff]).fetchdf()


def _get_extended_features_before(
    con: duckdb.DuckDBPyConnection, cutoff: str,
) -> pd.DataFrame:
    """Compute F16 extension features from PRs before cutoff.

    Returns: rejection_rate, hour_entropy (stub=0), empty_body_rate,
    title_spam_score (stub=0), weekend_ratio, prs_per_active_day.
    """
    rows = con.execute("""
        SELECT
            author,
            -- rejection_rate = 1 - merge_rate
            1.0 - (SUM(CASE WHEN state = 'MERGED' THEN 1 ELSE 0 END)::DOUBLE
                   / COUNT(*)::DOUBLE) AS rejection_rate,
            -- empty_body_rate
            SUM(CASE WHEN body IS NULL OR TRIM(body) = '' THEN 1 ELSE 0 END)::DOUBLE
                / COUNT(*)::DOUBLE AS empty_body_rate,
            -- weekend_ratio (Saturday=6, Sunday=0 in DuckDB's dayofweek)
            SUM(CASE WHEN EXTRACT(DOW FROM created_at) IN (0, 6)
                     THEN 1 ELSE 0 END)::DOUBLE
                / COUNT(*)::DOUBLE AS weekend_ratio,
            -- prs_per_active_day
            CASE
                WHEN EXTRACT(EPOCH FROM (MAX(created_at) - MIN(created_at)))
                     / 86400.0 > 0
                THEN COUNT(*)::DOUBLE / (
                    EXTRACT(EPOCH FROM (MAX(created_at) - MIN(created_at)))
                    / 86400.0)
                ELSE COUNT(*)::DOUBLE
            END AS prs_per_active_day
        FROM prs
        WHERE created_at < ?::TIMESTAMP AND author IS NOT NULL
        GROUP BY author
    """, [cutoff]).fetchdf()

    # Hour entropy and title_spam_score need Python computation
    pr_data = con.execute("""
        SELECT author, EXTRACT(HOUR FROM created_at)::INT AS hour, title
        FROM prs
        WHERE created_at < ?::TIMESTAMP AND author IS NOT NULL
    """, [cutoff]).fetchdf()

    hour_entropy = _compute_hour_entropy(pr_data)
    title_spam = _compute_title_spam_score(pr_data)

    rows["hour_entropy"] = rows["author"].map(hour_entropy).fillna(0.0)
    rows["title_spam_score"] = rows["author"].map(title_spam).fillna(0.0)

    return rows


def _compute_hour_entropy(pr_data: pd.DataFrame) -> dict[str, float]:
    """Shannon entropy of PR submission hour distribution per author."""
    result: dict[str, float] = {}
    for author, group in pr_data.groupby("author"):
        hours = group["hour"].values
        if len(hours) < 2:
            result[str(author)] = 0.0
            continue
        counts = np.bincount(hours, minlength=24).astype(float)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        result[str(author)] = float(-np.sum(probs * np.log2(probs)))
    return result


def _compute_title_spam_score(pr_data: pd.DataFrame) -> dict[str, float]:
    """Heuristic spam score based on PR title patterns."""
    spam_patterns = [
        "update readme", "add files", "initial commit", "first commit",
        "create ", "delete ", "rename ", "added ", "updated ",
    ]
    result: dict[str, float] = {}
    for author, group in pr_data.groupby("author"):
        titles = group["title"].dropna().str.lower()
        if len(titles) == 0:
            result[str(author)] = 0.0
            continue
        spam_count = sum(
            1 for t in titles
            if any(p in t for p in spam_patterns)
        )
        result[str(author)] = spam_count / len(titles)
    return result


def _compute_isolation_before(
    con: duckdb.DuckDBPyConnection, cutoff: str,
) -> dict[str, float]:
    """Compute isolation_score per author from pre-cutoff data."""
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


def _compute_graph_features_before(
    con: duckdb.DuckDBPyConnection, cutoff: str,
) -> dict[str, dict[str, float]]:
    """Compute hub_score and bipartite_clustering from pre-cutoff graph."""
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

    centrality = nx.degree_centrality(g)
    hub_scores = {
        a.removeprefix("a:"): centrality.get(a, 0.0) for a in authors_set
    }

    try:
        clustering = nx.bipartite.clustering(g, authors_set)
        bip_clustering = {
            a.removeprefix("a:"): clustering.get(a, 0.0) for a in authors_set
        }
    except Exception:
        bip_clustering = {a.removeprefix("a:"): 0.0 for a in authors_set}

    return {"hub_score": hub_scores, "bipartite_clustering": bip_clustering}


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_features(
    df: pd.DataFrame,
    feature_list: list[str],
) -> np.ndarray:
    """Extract columns, log-transform skewed ones, fill NaN with 0."""
    arrays = []
    for col in feature_list:
        vals = df[col].fillna(0).values.astype(float)
        if col in LOG_TRANSFORM:
            vals = np.log1p(np.abs(vals)) * np.sign(vals)
        arrays.append(vals)
    return np.column_stack(arrays)


# ---------------------------------------------------------------------------
# CV engine — suspended-only splitting for k-NN
# ---------------------------------------------------------------------------

def run_suspended_cv(
    df: pd.DataFrame,
    score_fn: Any,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """CV that folds only suspended accounts; active in every fold's eval.

    score_fn(seed_df, eval_df) -> np.ndarray of scores for eval_df rows.
    Returns (y_true, y_scores) for the full population.
    """
    y = (df["account_status"] == "suspended").astype(int).values
    susp_idx = np.where(y == 1)[0]
    active_idx = np.where(y == 0)[0]
    n_susp = len(susp_idx)

    use_loo = n_susp < 30

    susp_oof_scores = np.full(n_susp, np.nan)
    active_score_accum = np.zeros(len(active_idx))
    active_fold_count = np.zeros(len(active_idx))

    if use_loo:
        for i in range(n_susp):
            seed_susp_pos = np.array([j for j in range(n_susp) if j != i])
            if len(seed_susp_pos) == 0:
                continue

            seed_susp_df = df.iloc[susp_idx[seed_susp_pos]]

            eval_idx = np.concatenate([susp_idx[[i]], active_idx])
            eval_df = df.iloc[eval_idx]

            scores = score_fn(seed_susp_df, eval_df)

            susp_oof_scores[i] = scores[0]
            active_score_accum += scores[1:]
            active_fold_count += 1
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        for fold_train_pos, fold_test_pos in kf.split(np.arange(n_susp)):
            seed_susp_idx = susp_idx[fold_train_pos]
            held_out_susp_idx = susp_idx[fold_test_pos]

            seed_susp_df = df.iloc[seed_susp_idx]

            eval_idx = np.concatenate([held_out_susp_idx, active_idx])
            eval_df = df.iloc[eval_idx]

            scores = score_fn(seed_susp_df, eval_df)

            n_held = len(held_out_susp_idx)
            susp_oof_scores[fold_test_pos] = scores[:n_held]
            active_score_accum += scores[n_held:]
            active_fold_count += 1

    safe_count = np.where(active_fold_count > 0, active_fold_count, 1.0)
    active_avg_scores = active_score_accum / safe_count

    y_scores = np.full(len(y), np.nan)
    y_scores[susp_idx] = susp_oof_scores
    y_scores[active_idx] = active_avg_scores

    return y, y_scores


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> dict[str, Any]:
    """Compute AUC-ROC, AUC-PR, Precision@25, Precision@50."""
    metrics: dict[str, Any] = {}
    finite = np.isfinite(y_scores)
    if y_true[finite].sum() > 0 and (1 - y_true[finite]).sum() > 0:
        metrics["auc_roc"] = float(roc_auc_score(y_true[finite], y_scores[finite]))
        metrics["auc_pr"] = float(average_precision_score(
            y_true[finite], y_scores[finite],
        ))
        for k in [25, 50]:
            if k <= finite.sum():
                top_k_idx = np.argsort(y_scores[finite])[-k:]
                metrics[f"precision_at_{k}"] = float(
                    y_true[finite][top_k_idx].sum() / k
                )
    else:
        metrics["auc_roc"] = float("nan")
        metrics["auc_pr"] = float("nan")
    return metrics


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def lr_baseline(
    df: pd.DataFrame,
    feature_list: list[str],
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Logistic regression baseline with LOO/5-fold CV.

    Returns (y_true, y_scores).
    """
    y = (df["account_status"] == "suspended").astype(int).values
    x = prepare_features(df, feature_list)
    n_pos = y.sum()

    oof = np.full(len(y), np.nan)

    if n_pos < 30:
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

    return y, oof


def merge_rate_baseline(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """1 - merge_rate as a simple baseline score."""
    y = (df["account_status"] == "suspended").astype(int).values
    scores = 1.0 - df["merge_rate"].fillna(0).values.astype(float)
    return y, scores


# ---------------------------------------------------------------------------
# Statistical tests
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
