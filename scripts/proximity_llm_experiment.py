"""LLM-based suspension detection experiment with temporal holdout.

Tests whether LLM scoring of PR titles/bodies adds signal beyond behavioral
and graph-proximity features. Uses temporal cutoffs (Strategy C) to prevent
lookahead bias: the LLM only sees PR titles and bodies that existed before
the cutoff date, and all behavioral features and graph data are similarly
restricted to pre-cutoff information.

Three prompt variants:
  V1: PR titles only (cheapest)
  V2: Titles + body excerpts (first 500 chars at submission time)
  V3: Full profile with pre-cutoff metadata + titles + bodies

Integration modes:
  1. Combined model: LLM score as extra LR feature alongside behavioral features
  2. Second-phase re-ranking: LLM re-ranks top-N from first-phase model

Usage:
    uv run python scripts/proximity_llm_experiment.py --warmup-only
    uv run python scripts/proximity_llm_experiment.py
    uv run python scripts/proximity_llm_experiment.py --cutoff 2024-01-01
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from proximity_combined import compute_graph_scores_cv, lr_with_proximity_cv
from proximity_common import (
    CUTOFFS,
    DB_PATH,
    F10,
    F16,
    RESULTS_DIR,
    compute_metrics,
    delong_auc_test,
    holm_bonferroni,
    load_temporal_features,
    prepare_features,
)
from proximity_graph_experiment import build_author_repo_data
from proximity_llm_client import score_authors_batch
from sklearn.metrics import roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL = "gemini/gemini-3.1-pro-preview"
CACHE_DIR = Path("experiments/bot_detection/data/llm_cache")
OUTPUT_PATH = RESULTS_DIR / "llm_results.json"


# ---------------------------------------------------------------------------
# Data loading — PR text from DuckDB
# ---------------------------------------------------------------------------

def load_author_pr_data(
    con: duckdb.DuckDBPyConnection,
    authors: list[str],
    cutoff: str,
    max_prs: int = 20,
) -> dict[str, dict]:
    """Load up to max_prs most recent merged PRs per author before cutoff.

    Only includes PR title and body as they existed at creation time
    (the created_at timestamp). No post-submission data is included.

    Returns {author: {"titles": [...], "bodies": [...], "metadata": {...}}}
    """
    placeholders = ", ".join(["?"] * len(authors))
    rows = con.execute(f"""
        WITH ranked AS (
            SELECT
                author, title, body,
                ROW_NUMBER() OVER (
                    PARTITION BY author
                    ORDER BY created_at DESC
                ) AS rn
            FROM prs
            WHERE author IN ({placeholders})
              AND state = 'MERGED'
              AND created_at < ?::TIMESTAMP
              AND author IS NOT NULL
        )
        SELECT author, title, body
        FROM ranked
        WHERE rn <= {max_prs}
        ORDER BY author
    """, [*authors, cutoff]).fetchall()

    # Per-author metadata computed only from pre-cutoff PRs
    meta_rows = con.execute(f"""
        SELECT
            author,
            COUNT(*) AS total_prs,
            SUM(CASE WHEN state = 'MERGED' THEN 1 ELSE 0 END)::DOUBLE
                / NULLIF(COUNT(*), 0)::DOUBLE AS merge_rate,
            COUNT(DISTINCT repo) AS total_repos,
            COALESCE(
                EXTRACT(EPOCH FROM (MAX(created_at) - MIN(created_at)))
                / 86400.0, 0.0
            ) AS career_span_days
        FROM prs
        WHERE author IN ({placeholders})
          AND created_at < ?::TIMESTAMP
          AND author IS NOT NULL
        GROUP BY author
    """, [*authors, cutoff]).fetchall()

    meta_map: dict[str, dict] = {}
    for author, total_prs, merge_rate, total_repos, career_span in meta_rows:
        meta_map[author] = {
            "total_prs": int(total_prs),
            "merge_rate": round(float(merge_rate or 0), 3),
            "total_repos": int(total_repos),
            "career_span_days": round(float(career_span or 0), 1),
        }

    author_data: dict[str, dict] = {}
    for author in authors:
        author_data[author] = {"titles": [], "bodies": [], "metadata": meta_map.get(author, {})}

    for author, title, body in rows:
        if author in author_data:
            author_data[author]["titles"].append(title or "")
            author_data[author]["bodies"].append(body or "")

    return author_data


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_v1_prompt(data: dict) -> str:
    """V1: Title-only prompt (cheapest/fastest)."""
    titles = data.get("titles", [])
    if not titles:
        titles = ["(no merged PRs found)"]
    title_list = "\n".join(f"- {t}" for t in titles[:20])
    return f"""You are analyzing GitHub PR titles from a single author. These PRs were all \
merged into their target repositories, meaning they passed code review.

Assess whether this author's contribution pattern suggests they may be a \
suspended or malicious GitHub account. Obvious spam is unlikely since these \
PRs were accepted. Look for subtle patterns: repetitive trivial changes, \
unusual naming conventions, contributions that look like gaming metrics.

PR titles (most recent first):
{title_list}

Rate suspicion from 0.0 (clearly legitimate) to 1.0 (highly suspicious).
Respond with JSON: {{"score": 0.X, "reasoning": "brief explanation"}}"""


def build_v2_prompt(data: dict) -> str:
    """V2: Titles + body excerpts (first 500 chars of up to 10 PRs)."""
    titles = data.get("titles", [])
    bodies = data.get("bodies", [])
    if not titles:
        titles = ["(no merged PRs found)"]
        bodies = [""]

    pr_list_parts = []
    for i, (title, body) in enumerate(zip(titles[:10], bodies[:10], strict=True)):
        excerpt = (body[:500] + "...") if len(body) > 500 else body
        excerpt = excerpt.strip() or "(empty body)"
        pr_list_parts.append(f"PR {i + 1}: {title}\n  Body: {excerpt}")

    pr_list = "\n".join(pr_list_parts)
    remaining = len(titles) - 10
    if remaining > 0:
        pr_list += f"\n... and {remaining} more PRs (titles only):"
        for t in titles[10:20]:
            pr_list += f"\n- {t}"

    return f"""You are analyzing GitHub PRs from a single author. These PRs were all \
merged into their target repositories, meaning they passed code review.

Assess whether this author's contribution pattern suggests they may be a \
suspended or malicious GitHub account. Obvious spam is unlikely since these \
PRs were accepted. Look for subtle patterns: repetitive trivial changes, \
unusual naming conventions, low-effort body text, contributions that look \
like gaming metrics.

PRs (most recent first):
{pr_list}

Rate suspicion from 0.0 (clearly legitimate) to 1.0 (highly suspicious).
Respond with JSON: {{"score": 0.X, "reasoning": "brief explanation"}}"""


def build_v3_prompt(data: dict) -> str:
    """V3: Full profile with metadata + titles + body excerpts."""
    meta = data.get("metadata", {})
    titles = data.get("titles", [])
    bodies = data.get("bodies", [])

    meta_block = f"""Author profile:
- Total PRs: {meta.get('total_prs', 'unknown')}
- Merge rate: {meta.get('merge_rate', 'unknown')}
- Repos contributed to: {meta.get('total_repos', 'unknown')}
- Career span: {meta.get('career_span_days', 'unknown')} days"""

    if not titles:
        titles = ["(no merged PRs found)"]
        bodies = [""]

    pr_list_parts = []
    for i, (title, body) in enumerate(zip(titles[:10], bodies[:10], strict=True)):
        excerpt = (body[:500] + "...") if len(body) > 500 else body
        excerpt = excerpt.strip() or "(empty body)"
        pr_list_parts.append(f"PR {i + 1}: {title}\n  Body: {excerpt}")

    pr_list = "\n".join(pr_list_parts)
    remaining = len(titles) - 10
    if remaining > 0:
        pr_list += f"\n... and {remaining} more PRs (titles only):"
        for t in titles[10:20]:
            pr_list += f"\n- {t}"

    return f"""You are analyzing a GitHub author's complete contribution profile. These PRs \
were all merged into their target repositories, meaning they passed code review.

Assess whether this author's contribution pattern suggests they may be a \
suspended or malicious GitHub account. Obvious spam is unlikely since these \
PRs were accepted. Look for subtle patterns: repetitive trivial changes, \
unusual naming conventions, low-effort contributions, metrics inconsistent \
with genuine development work.

{meta_block}

PRs (most recent first):
{pr_list}

Rate suspicion from 0.0 (clearly legitimate) to 1.0 (highly suspicious).
Respond with JSON: {{"score": 0.X, "reasoning": "brief explanation"}}"""


# ---------------------------------------------------------------------------
# Standalone evaluation
# ---------------------------------------------------------------------------

def evaluate_standalone(
    y: np.ndarray,
    authors: list[str],
    scores_dict: dict[str, float],
    label: str,
) -> dict[str, Any]:
    """Evaluate standalone LLM scores (authors without scores are dropped)."""
    mask = np.array([a in scores_dict for a in authors])
    y_eval = y[mask]
    llm_arr = np.array([scores_dict[a] for a in authors if a in scores_dict])
    n_dropped = int((~mask).sum())

    metrics = compute_metrics(y_eval, llm_arr)

    logger.info(
        "  %s standalone: AUC=%.4f, AUC-PR=%.4f, P@25=%.2f, P@50=%.2f "
        "(%d scored, %d dropped)",
        label,
        metrics.get("auc_roc", float("nan")),
        metrics.get("auc_pr", float("nan")),
        metrics.get("precision_at_25", float("nan")),
        metrics.get("precision_at_50", float("nan")),
        len(llm_arr),
        n_dropped,
    )

    return {
        **metrics,
        "n_scored": len(llm_arr),
        "n_dropped": n_dropped,
    }


# ---------------------------------------------------------------------------
# Combined model evaluation
# ---------------------------------------------------------------------------

def _lr_multi_proximity_cv(
    df: pd.DataFrame,
    feature_list: list[str],
    extra_columns: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """LR CV with behavioral features + multiple extra columns.

    Like lr_with_proximity_cv but accepts a 2D array of extra features,
    letting LR learn separate weights for each.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut, StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    y = (df["account_status"] == "suspended").astype(int).values
    x_base = prepare_features(df, feature_list)

    # Fill NaN with column medians
    if extra_columns.ndim == 1:
        extra_columns = extra_columns.reshape(-1, 1)
    medians = np.nanmedian(extra_columns, axis=0)
    extra_clean = np.where(np.isfinite(extra_columns), extra_columns, medians)
    x_combined = np.hstack([x_base, extra_clean])

    n_pos = y.sum()
    oof = np.full(len(y), np.nan)

    if n_pos < 30:
        splitter = LeaveOneOut()
    else:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for train_idx, test_idx in splitter.split(x_combined, y):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_combined[train_idx])
        x_test = scaler.transform(x_combined[test_idx])
        model = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=seed,
        )
        model.fit(x_train, y[train_idx])
        oof[test_idx] = model.predict_proba(x_test)[:, 1]

    return y, oof


def evaluate_combined(
    df: pd.DataFrame,
    y: np.ndarray,
    authors: list[str],
    llm_scores: dict[str, float],
    jaccard_oof: np.ndarray,
    feature_sets: dict[str, list[str]],
    variant_label: str,
) -> dict[str, Any]:
    """Test LLM as extra LR feature, alone and combined with Jaccard.

    Authors without LLM scores are dropped from all evaluations to avoid
    corrupting results with default values.
    """
    mask = np.array([a in llm_scores for a in authors])
    n_dropped = int((~mask).sum())
    if n_dropped > 0:
        logger.info(
            "  Dropping %d/%d authors without LLM scores for %s",
            n_dropped, len(authors), variant_label,
        )
    df_eval = df.iloc[mask].reset_index(drop=True)
    y_eval = y[mask]
    jaccard_eval = jaccard_oof[mask]
    llm_arr = np.array([llm_scores[a] for a in authors if a in llm_scores])
    results: dict[str, Any] = {"n_dropped": n_dropped}

    for fs_name, fs_list in feature_sets.items():
        logger.info("  Combined models: %s + %s", fs_name, variant_label)

        # Baseline: LR(behavioral) only — on filtered population
        y_base, oof_base = lr_with_proximity_cv(
            df_eval, fs_list, np.zeros(len(df_eval)),
        )
        base_auc = roc_auc_score(y_base, oof_base)

        # LR + Jaccard (prior best)
        y_jac, oof_jac = lr_with_proximity_cv(df_eval, fs_list, jaccard_eval)
        jac_auc = roc_auc_score(y_jac, oof_jac)

        # LR + LLM
        y_llm, oof_llm = lr_with_proximity_cv(df_eval, fs_list, llm_arr)
        llm_auc = roc_auc_score(y_llm, oof_llm)

        # LR + LLM + Jaccard (both as separate features — LR learns weights)
        both_extra = np.column_stack([llm_arr, jaccard_eval])
        y_both, oof_both = _lr_multi_proximity_cv(df_eval, fs_list, both_extra)
        both_auc = roc_auc_score(y_both, oof_both)

        # DeLong tests (all use same y since same filtered population)
        dl_llm_vs_base = delong_auc_test(y_eval, oof_llm, oof_base)
        dl_llm_vs_jac = delong_auc_test(y_eval, oof_llm, oof_jac)
        dl_both_vs_jac = delong_auc_test(y_eval, oof_both, oof_jac)
        dl_both_vs_base = delong_auc_test(y_eval, oof_both, oof_base)

        logger.info(
            "    LR(%s) baseline:     AUC=%.4f", fs_name, base_auc,
        )
        logger.info(
            "    LR(%s) + Jaccard:    AUC=%.4f (delta=%+.4f)",
            fs_name, jac_auc, jac_auc - base_auc,
        )
        logger.info(
            "    LR(%s) + LLM(%s):  AUC=%.4f (delta=%+.4f, p=%.4f)",
            fs_name, variant_label, llm_auc, llm_auc - base_auc,
            dl_llm_vs_base["p_value"],
        )
        logger.info(
            "    LR(%s) + LLM + Jac: AUC=%.4f (delta=%+.4f vs Jac, p=%.4f)",
            fs_name, both_auc, both_auc - jac_auc,
            dl_both_vs_jac["p_value"],
        )

        results[fs_name] = {
            "baseline_auc": float(base_auc),
            "jaccard_auc": float(jac_auc),
            "llm_auc": float(llm_auc),
            "llm_plus_jaccard_auc": float(both_auc),
            "delong_llm_vs_baseline": {
                "z": float(dl_llm_vs_base["z_statistic"]),
                "p": float(dl_llm_vs_base["p_value"]),
            },
            "delong_llm_vs_jaccard": {
                "z": float(dl_llm_vs_jac["z_statistic"]),
                "p": float(dl_llm_vs_jac["p_value"]),
            },
            "delong_both_vs_jaccard": {
                "z": float(dl_both_vs_jac["z_statistic"]),
                "p": float(dl_both_vs_jac["p_value"]),
            },
            "delong_both_vs_baseline": {
                "z": float(dl_both_vs_base["z_statistic"]),
                "p": float(dl_both_vs_base["p_value"]),
            },
        }

    return results


# ---------------------------------------------------------------------------
# Second-phase re-ranking
# ---------------------------------------------------------------------------

def evaluate_second_phase(
    y: np.ndarray,
    authors: list[str],
    first_phase_scores: np.ndarray,
    llm_scores: dict[str, float],
    variant_label: str,
    top_ns: list[int] | None = None,
    alphas: list[float] | None = None,
) -> dict[str, Any]:
    """Re-rank top-N from first-phase model using LLM scores.

    Authors without LLM scores are dropped before evaluation.
    """
    if top_ns is None:
        top_ns = [100, 200, 500]
    if alphas is None:
        alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]

    # Drop authors without LLM scores
    mask = np.array([a in llm_scores for a in authors])
    y = y[mask]
    first_phase_scores = first_phase_scores[mask]
    scored_authors = [a for a in authors if a in llm_scores]
    llm_arr = np.array([llm_scores[a] for a in scored_authors])
    results: dict[str, Any] = {"n_dropped": int((~mask).sum())}

    for top_n in top_ns:
        if top_n > len(y):
            continue

        top_idx = np.argsort(first_phase_scores)[-top_n:]
        y_top = y[top_idx]
        first_top = first_phase_scores[top_idx]
        llm_top = llm_arr[top_idx]

        # Baseline: first-phase ordering precision
        first_order = np.argsort(first_top)[::-1]
        llm_order = np.argsort(llm_top)[::-1]

        phase_results: dict[str, Any] = {
            "n_suspended_in_top": int(y_top.sum()),
        }

        for k in [25, 50]:
            if k <= top_n:
                phase_results[f"first_phase_p_at_{k}"] = float(
                    y_top[first_order[:k]].sum() / k
                )
                phase_results[f"llm_rerank_p_at_{k}"] = float(
                    y_top[llm_order[:k]].sum() / k
                )

        # Alpha sweep: combined = alpha * first_phase + (1-alpha) * llm
        # Z-score normalize both to make alpha values interpretable
        fp_std = np.std(first_top)
        llm_std = np.std(llm_top)
        first_z = ((first_top - np.mean(first_top)) / fp_std) if fp_std > 0 else first_top
        llm_z = ((llm_top - np.mean(llm_top)) / llm_std) if llm_std > 0 else llm_top

        alpha_results: dict[str, Any] = {}
        for alpha in alphas:
            combined = alpha * first_z + (1 - alpha) * llm_z
            combined_order = np.argsort(combined)[::-1]
            a_result: dict[str, Any] = {}
            for k in [25, 50]:
                if k <= top_n:
                    a_result[f"p_at_{k}"] = float(
                        y_top[combined_order[:k]].sum() / k
                    )
            alpha_results[f"alpha_{alpha:.1f}"] = a_result

        phase_results["alpha_sweep"] = alpha_results
        results[f"top_{top_n}"] = phase_results

        logger.info(
            "  Second-phase top_%d (%s): %d suspended, "
            "first-phase P@25=%.2f, LLM-rerank P@25=%.2f",
            top_n,
            variant_label,
            y_top.sum(),
            phase_results.get("first_phase_p_at_25", float("nan")),
            phase_results.get("llm_rerank_p_at_25", float("nan")),
        )

    return results


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def run_warmup(
    df: pd.DataFrame,
    con: duckdb.DuckDBPyConnection,
    cutoff: str,
) -> dict[str, Any]:
    """Run end-to-end pipeline on 50 authors to validate everything works.

    Uses the given cutoff to ensure no lookahead bias even in warmup.
    """
    logger.info("=== WARMUP: 50 authors (25 suspended, 25 active), cutoff=%s ===", cutoff)

    y = (df["account_status"] == "suspended").astype(int).values
    author_col = "author" if "author" in df.columns else "login"
    susp_idx = np.where(y == 1)[0]
    active_idx = np.where(y == 0)[0]

    rng = np.random.RandomState(42)
    sample_susp = rng.choice(susp_idx, size=min(25, len(susp_idx)), replace=False)
    sample_active = rng.choice(active_idx, size=min(25, len(active_idx)), replace=False)
    sample_idx = np.concatenate([sample_susp, sample_active])

    sample_df = df.iloc[sample_idx].reset_index(drop=True)
    sample_authors = sample_df[author_col].tolist()
    sample_y = (sample_df["account_status"] == "suspended").astype(int).values

    logger.info(
        "  Sample: %d authors (%d suspended, %d active)",
        len(sample_authors), sample_y.sum(), (1 - sample_y).sum(),
    )

    # Load PR data for sample — only PRs before cutoff
    author_pr_data = load_author_pr_data(con, sample_authors, cutoff)

    # Score with V1
    logger.info("  Scoring with V1 (title-only)...")
    scores_v1 = asyncio.run(
        score_authors_batch(MODEL, author_pr_data, build_v1_prompt, CACHE_DIR)
    )

    # Standalone metrics — drop authors without scores
    scored_mask = np.array([a in scores_v1 for a in sample_authors])
    llm_arr = np.array([scores_v1[a] for a in sample_authors if a in scores_v1])
    metrics = compute_metrics(sample_y[scored_mask], llm_arr)
    v1_auc = metrics.get("auc_roc", float("nan"))

    logger.info(
        "  V1 warmup AUC=%.4f, P@25=%.2f",
        v1_auc,
        metrics.get("precision_at_25", float("nan")),
    )

    # Score distribution
    status_map = dict(zip(
        sample_df[author_col], sample_df["account_status"], strict=True,
    ))
    susp_scores = [scores_v1[a] for a in sample_authors
                   if a in scores_v1 and status_map[a] == "suspended"]
    active_scores = [scores_v1[a] for a in sample_authors
                     if a in scores_v1 and status_map[a] == "active"]

    logger.info(
        "  Score distribution — suspended: mean=%.3f, active: mean=%.3f",
        np.mean(susp_scores) if susp_scores else 0,
        np.mean(active_scores) if active_scores else 0,
    )

    # Validate parse success rate
    n_parsed = sum(1 for a in sample_authors if a in scores_v1)
    logger.info("  Parse success: %d/%d (%.1f%%)",
                n_parsed, len(sample_authors), 100 * n_parsed / len(sample_authors))

    result = {
        "cutoff": cutoff,
        "n_authors": len(sample_authors),
        "n_suspended": int(sample_y.sum()),
        "n_active": int((1 - sample_y).sum()),
        "v1_auc": float(v1_auc),
        "v1_metrics": metrics,
        "n_parsed": n_parsed,
        "parse_rate": n_parsed / len(sample_authors),
        "susp_mean_score": float(np.mean(susp_scores)) if susp_scores else None,
        "active_mean_score": float(np.mean(active_scores)) if active_scores else None,
        "pipeline_status": "ok",
    }

    logger.info("=== WARMUP COMPLETE ===")
    return result


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------

def run_cutoff_experiment(
    df: pd.DataFrame,
    con: duckdb.DuckDBPyConnection,
    author_repos: dict[str, set],
    graph: Any,
    cutoff: str,
    all_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run LLM experiment for a single temporal cutoff.

    All PR data shown to the LLM is restricted to before the cutoff date,
    preventing lookahead bias. Behavioral features and graph data are also
    pre-cutoff (handled by caller via load_temporal_features / build_author_repo_data).
    """
    author_col = "author" if "author" in df.columns else "login"
    authors = df[author_col].tolist()
    y = (df["account_status"] == "suspended").astype(int).values

    results: dict[str, Any] = {
        "cutoff": cutoff,
        "n_authors": len(authors),
        "n_suspended": int(y.sum()),
        "n_active": int((1 - y).sum()),
    }

    # Load PR text data — only PRs before cutoff
    logger.info("Loading PR data for %d authors (cutoff=%s)...", len(authors), cutoff)
    author_pr_data = load_author_pr_data(con, authors, cutoff)
    n_with_data = sum(1 for d in author_pr_data.values() if d["titles"])
    logger.info("  %d/%d authors have PR title data before %s", n_with_data, len(authors), cutoff)

    def _checkpoint() -> None:
        """Save incremental results to disk."""
        if all_results is not None:
            all_results[cutoff] = results
            _save_results(all_results)

    # --- Scoring phase ---
    logger.info("=== SCORING PHASE (cutoff=%s) ===", cutoff)

    all_scores: dict[str, dict[str, float]] = {}
    for label, builder in [("v1", build_v1_prompt), ("v2", build_v2_prompt),
                           ("v3", build_v3_prompt)]:
        logger.info("Scoring %s for %d authors...", label, len(authors))
        scores = asyncio.run(
            score_authors_batch(MODEL, author_pr_data, builder, CACHE_DIR)
        )
        logger.info("%s complete: %d scored", label, len(scores))
        all_scores[label] = scores
        results["scoring_progress"] = {
            "completed": label,
            "n_scored": len(scores),
        }
        _checkpoint()

    # --- Standalone evaluation ---
    logger.info("=== STANDALONE EVALUATION ===")
    standalone = {}
    for label, scores in all_scores.items():
        standalone[label] = evaluate_standalone(y, authors, scores, label)
    results["standalone"] = standalone
    _checkpoint()

    # --- Combined model ---
    logger.info("=== COMBINED MODEL EVALUATION ===")

    # Compute Jaccard OOF scores from pre-cutoff graph
    logger.info("Computing Jaccard OOF scores...")
    jaccard_oof = compute_graph_scores_cv(
        df, author_repos, graph, "jaccard_max",
    )

    available_f16 = [f for f in F16 if f in df.columns]
    feature_sets = {"F10": [f for f in F10 if f in df.columns], "F16": available_f16}

    combined = {}
    for label, scores in all_scores.items():
        combined[label] = evaluate_combined(
            df, y, authors, scores, jaccard_oof, feature_sets, label,
        )
    results["combined"] = combined
    _checkpoint()

    # Holm-Bonferroni correction across all DeLong tests
    all_p_values: dict[str, float] = {}
    for variant, variant_data in combined.items():
        for fs_name, fs_data in variant_data.items():
            if not isinstance(fs_data, dict):
                continue
            for test_key in ["delong_llm_vs_baseline", "delong_llm_vs_jaccard",
                             "delong_both_vs_jaccard", "delong_both_vs_baseline"]:
                if test_key in fs_data:
                    label_key = f"{variant}/{fs_name}/{test_key}"
                    all_p_values[label_key] = fs_data[test_key]["p"]

    if all_p_values:
        corrected = holm_bonferroni(all_p_values)
        results["holm_bonferroni_correction"] = {
            k: {"raw_p": v["p_value"], "adjusted_p": v["adjusted_p"],
                "reject_h0": v["reject"]}
            for k, v in corrected.items()
        }
        n_reject = sum(1 for v in corrected.values() if v["reject"])
        logger.info(
            "Holm-Bonferroni: %d/%d tests significant after correction",
            n_reject, len(corrected),
        )

    # --- Second-phase re-ranking ---
    logger.info("=== SECOND-PHASE RE-RANKING ===")

    # First-phase: LR(F10) + Jaccard from pre-cutoff features
    f10_available = [f for f in F10 if f in df.columns]
    logger.info("Computing first-phase OOF scores (LR(F10) + Jaccard)...")
    _, first_phase_oof = lr_with_proximity_cv(df, f10_available, jaccard_oof)

    second_phase = {}
    for label, scores in all_scores.items():
        second_phase[label] = evaluate_second_phase(
            y, authors, first_phase_oof, scores, label,
        )
    results["second_phase"] = second_phase
    _checkpoint()

    return results


# ---------------------------------------------------------------------------
# JSON serializer
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


def _save_results(results: dict[str, Any]) -> None:
    """Write results to disk immediately. Called after every major phase."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    logger.info("Results checkpoint saved to %s", OUTPUT_PATH)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM suspension detection experiment")
    parser.add_argument(
        "--warmup-only", action="store_true",
        help="Run warmup phase only (50 authors at first cutoff)",
    )
    parser.add_argument(
        "--cutoff", type=str, default=None,
        help="Run a single cutoff instead of all (e.g. '2024-01-01')",
    )
    args = parser.parse_args()

    cutoffs = [args.cutoff] if args.cutoff else CUTOFFS
    con = duckdb.connect(str(DB_PATH), read_only=True)

    # Warmup on first cutoff
    warmup_cutoff = cutoffs[0]
    logger.info("Loading temporal features for warmup cutoff %s...", warmup_cutoff)
    warmup_df = load_temporal_features(warmup_cutoff)
    if "login" in warmup_df.columns and "author" not in warmup_df.columns:
        warmup_df = warmup_df.rename(columns={"login": "author"})

    n_susp = (warmup_df["account_status"] == "suspended").sum()
    if n_susp < 5:
        logger.error(
            "Warmup cutoff %s has only %d suspended, aborting.",
            warmup_cutoff, n_susp,
        )
        con.close()
        sys.exit(1)

    warmup_result = run_warmup(warmup_df, con, warmup_cutoff)

    if warmup_result["pipeline_status"] != "ok":
        logger.error("Warmup failed! Aborting.")
        con.close()
        sys.exit(1)

    if args.warmup_only:
        logger.info("Warmup-only mode, saving warmup results and exiting.")
        output = {"warmup": warmup_result, "config": {"model": MODEL, "cutoffs": cutoffs}}
        _save_results(output)
        con.close()
        return

    # Full experiment — one run per cutoff
    # Load prior results to resume from if they exist
    all_results: dict[str, Any] = {}
    if OUTPUT_PATH.exists():
        try:
            with open(OUTPUT_PATH) as f:
                all_results = json.load(f)
            logger.info("Loaded prior results from %s", OUTPUT_PATH)
        except (json.JSONDecodeError, OSError):
            pass
    all_results["config"] = {"model": MODEL, "cutoffs": cutoffs}
    all_results["warmup"] = warmup_result
    _save_results(all_results)

    for cutoff in cutoffs:
        # Skip cutoffs that already have complete results (second_phase present)
        prior = all_results.get(cutoff, {})
        if isinstance(prior, dict) and "second_phase" in prior:
            logger.info("Cutoff %s already complete, skipping", cutoff)
            continue

        logger.info("\n" + "=" * 60)
        logger.info("CUTOFF: %s", cutoff)
        logger.info("=" * 60)

        df = load_temporal_features(cutoff)
        if "login" in df.columns and "author" not in df.columns:
            df = df.rename(columns={"login": "author"})

        n_susp = (df["account_status"] == "suspended").sum()
        if n_susp < 5:
            logger.warning("Skipping cutoff %s: only %d suspended authors", cutoff, n_susp)
            continue

        # Build graph from pre-cutoff data
        author_repos, graph = build_author_repo_data(con, cutoff)

        cutoff_results = run_cutoff_experiment(
            df, con, author_repos, graph, cutoff, all_results,
        )
        all_results[cutoff] = cutoff_results
        _save_results(all_results)

    con.close()
    logger.info("All cutoffs complete. Final results at %s", OUTPUT_PATH)

    # Print summary
    print("\n" + "=" * 60)
    print("LLM EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"\nModel: {MODEL}")
    print(f"Cutoffs: {cutoffs}")

    for cutoff in cutoffs:
        cr = all_results.get(cutoff)
        if not cr:
            continue
        print(f"\n--- Cutoff: {cutoff} ({cr['n_authors']} authors, "
              f"{cr['n_suspended']} suspended) ---")

        print("  Standalone AUC-ROC:")
        for variant in ["v1", "v2", "v3"]:
            sa = cr.get("standalone", {}).get(variant, {})
            print(f"    {variant}: {sa.get('auc_roc', float('nan')):.4f}")

        print("  Best combined AUC-ROC:")
        for variant in ["v1", "v2", "v3"]:
            comb = cr.get("combined", {}).get(variant, {})
            for fs_name, fs_data in comb.items():
                if isinstance(fs_data, dict) and "llm_plus_jaccard_auc" in fs_data:
                    print(f"    {variant}/{fs_name}: "
                          f"LLM+Jac={fs_data['llm_plus_jaccard_auc']:.4f}"
                          f"  (Jac only={fs_data['jaccard_auc']:.4f})")


if __name__ == "__main__":
    main()
