"""Evaluate whether incorporating rejection (closed PR) data improves merge prediction.

Tests four approaches:
  1. Full model (no rejection awareness) — existing normalized_score
  2. Per-repo scaling — scale edges by author's merge rate at each repo
  3. Author-level scaling — scale all edges by author's overall merge rate
  4. LR(GE + merge_rate) — logistic regression combining GE score with merge rate

A hybrid mode (per-repo with author-level fallback) was initially considered
but dropped: the trust graph is built exclusively from merged PRs, so every
repo node has at least one merged PR, guaranteeing a per-repo rate always
exists. The fallback is structurally unreachable.
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from good_egg.config import GoodEggConfig
from good_egg.graph_builder import TrustGraphBuilder
from good_egg.models import UserContributionData
from good_egg.scorer import TrustScorer

# Add parent directory for local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checkpoint import read_json  # noqa: E402
from stats import (  # noqa: E402
    auc_roc_with_ci,
    delong_auc_test,
    holm_bonferroni,
    likelihood_ratio_test,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", font_scale=1.1)
DPI = 150
SEED = 42

BASE_DIR = Path(__file__).resolve().parents[1]
BACKFILL_CAP = 500


# ---------------------------------------------------------------------------
# RejectionAwareGraphBuilder
# ---------------------------------------------------------------------------

class RejectionAwareGraphBuilder(TrustGraphBuilder):
    """Graph builder that scales edge weights by merge rate."""

    def __init__(
        self,
        config: GoodEggConfig,
        mode: str,
        per_repo_merge_rates: dict[str, float] | None = None,
        author_merge_rate: float | None = None,
    ) -> None:
        super().__init__(config)
        self.mode = mode  # "per_repo" or "author_level"
        self.per_repo_merge_rates = per_repo_merge_rates or {}
        self.author_merge_rate = author_merge_rate

    def build_graph(
        self,
        user_data: UserContributionData,
        context_repo: str,
    ) -> Any:
        graph = super().build_graph(user_data, context_repo)
        user_node = f"user:{user_data.profile.login}"

        for repo_node in list(graph.successors(user_node)):
            repo_name = repo_node.replace("repo:", "")

            if self.mode == "per_repo":
                rate = self.per_repo_merge_rates.get(repo_name)
                if rate is None:
                    continue
            elif self.mode == "author_level":
                rate = self.author_merge_rate
                if rate is None:
                    continue
            else:
                continue

            # Scale forward edge
            if graph.has_edge(user_node, repo_node):
                graph[user_node][repo_node]["weight"] *= rate
            # Scale reverse edge
            if graph.has_edge(repo_node, user_node):
                graph[repo_node][user_node]["weight"] *= rate

        return graph


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_author_data(
    authors_dir: Path, login: str,
) -> UserContributionData | None:
    """Load author's UserContributionData from persisted JSON."""
    path = authors_dir / f"{login}.json"
    if not path.exists():
        return None
    raw = read_json(path)
    contrib = raw.get("contribution_data")
    if not contrib:
        return None
    return UserContributionData(**contrib)


def _load_closed_prs(
    authors_dir: Path, login: str,
) -> list[dict[str, str]]:
    """Load closed PRs from raw author JSON."""
    path = authors_dir / f"{login}.json"
    if not path.exists():
        return []
    raw = read_json(path)
    return raw.get("closed_prs", [])


def _apply_anti_lookahead(
    user_data: UserContributionData,
    cutoff: datetime,
) -> UserContributionData:
    """Filter merged PRs to only those merged before cutoff."""
    filtered_prs = [
        pr for pr in user_data.merged_prs
        if pr.merged_at < cutoff
    ]
    remaining_repos = {
        pr.repo_name_with_owner for pr in filtered_prs
    }
    filtered_repos = {
        k: v for k, v in user_data.contributed_repos.items()
        if k in remaining_repos
    }
    return UserContributionData(
        profile=user_data.profile,
        merged_prs=filtered_prs,
        contributed_repos=filtered_repos,
    )


def _filter_closed_prs(
    closed_prs: list[dict[str, str]],
    cutoff: datetime,
) -> list[dict[str, str]]:
    """Filter closed PRs to only those closed before cutoff."""
    result = []
    for cp in closed_prs:
        closed_at_str = cp.get("closed_at", "")
        if not closed_at_str:
            continue
        closed_at = datetime.fromisoformat(closed_at_str)
        if closed_at < cutoff:
            result.append(cp)
    return result


def _compute_merge_rates(
    filtered_data: UserContributionData,
    filtered_closed: list[dict[str, str]],
) -> tuple[dict[str, float], float | None]:
    """Compute per-repo and author-level merge rates.

    Returns (per_repo_rates, author_rate).
    """
    # Count merged PRs per repo
    merged_counts: dict[str, int] = {}
    for pr in filtered_data.merged_prs:
        repo = pr.repo_name_with_owner
        merged_counts[repo] = merged_counts.get(repo, 0) + 1

    # Count closed PRs per repo
    closed_counts: dict[str, int] = {}
    for cp in filtered_closed:
        repo = cp.get("repo", "")
        if repo:
            closed_counts[repo] = closed_counts.get(repo, 0) + 1

    # Per-repo merge rates
    all_repos = set(merged_counts.keys()) | set(closed_counts.keys())
    per_repo_rates: dict[str, float] = {}
    total_merged = 0
    total_closed = 0

    for repo in all_repos:
        m = merged_counts.get(repo, 0)
        c = closed_counts.get(repo, 0)
        total = m + c
        if total > 0:
            per_repo_rates[repo] = m / total
        total_merged += m
        total_closed += c

    # Author-level merge rate
    author_total = total_merged + total_closed
    author_rate: float | None = None
    if author_total > 0:
        author_rate = total_merged / author_total

    return per_repo_rates, author_rate


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_with_builder(
    user_data: UserContributionData,
    context_repo: str,
    config: GoodEggConfig,
    builder: TrustGraphBuilder,
) -> float:
    """Score using a custom graph builder."""
    scorer = TrustScorer(config)
    scorer._graph_builder = builder
    result = scorer.score(user_data, context_repo)
    return result.normalized_score


def score_all_prs(
    df: pd.DataFrame,
    authors_dir: Path,
) -> pd.DataFrame:
    """Score all PRs with rejection-aware approaches.

    Returns DataFrame with columns: per_repo_score, author_level_score,
    author_merge_rate_temporal, account_age_days.
    """
    config = GoodEggConfig()

    # Caches
    author_data_cache: dict[str, UserContributionData | None] = {}
    closed_prs_cache: dict[str, list[dict[str, str]]] = {}
    author_created_cache: dict[str, datetime | None] = {}

    per_repo_scores = np.full(len(df), np.nan)
    author_level_scores = np.full(len(df), np.nan)
    merge_rates_temporal = np.full(len(df), np.nan)
    account_age_arr = np.full(len(df), np.nan)

    # Sanity check tracking
    sanity_samples: list[dict[str, Any]] = []
    sanity_checks_total = 0
    sanity_violations = 0

    total = len(df)
    for idx, (_, row) in enumerate(df.iterrows()):
        if idx % 500 == 0:
            logger.info("Scoring PR %d / %d", idx, total)

        login = row["author_login"]
        repo = row["repo"]
        created_at_str = row["created_at"]
        if isinstance(created_at_str, str):
            cutoff = datetime.fromisoformat(created_at_str)
        else:
            cutoff = created_at_str.to_pydatetime()

        # Load author data
        if login not in author_data_cache:
            author_data_cache[login] = _load_author_data(
                authors_dir, login,
            )
        user_data = author_data_cache[login]
        if user_data is None:
            continue

        # Compute account age at PR creation time
        if login not in author_created_cache:
            created_str = getattr(user_data.profile, "created_at", None)
            if created_str:
                try:
                    author_created_cache[login] = datetime.fromisoformat(
                        str(created_str),
                    )
                except (ValueError, TypeError):
                    author_created_cache[login] = None
            else:
                author_created_cache[login] = None
        author_created = author_created_cache[login]
        if author_created is not None:
            age_days = (cutoff - author_created).days
            if age_days >= 0:
                account_age_arr[idx] = age_days

        # Load closed PRs
        if login not in closed_prs_cache:
            closed_prs_cache[login] = _load_closed_prs(
                authors_dir, login,
            )
        closed_prs = closed_prs_cache[login]

        # Apply anti-lookahead to merged PRs
        filtered_data = _apply_anti_lookahead(user_data, cutoff)

        # Filter closed PRs temporally
        filtered_closed = _filter_closed_prs(closed_prs, cutoff)

        # If no closed PR data, scores = full model (no scaling)
        if not closed_prs:
            per_repo_scores[idx] = row["normalized_score"]
            author_level_scores[idx] = row["normalized_score"]
            merge_rates_temporal[idx] = np.nan
            continue

        # Compute merge rates
        per_repo_rates, author_rate = _compute_merge_rates(
            filtered_data, filtered_closed,
        )
        merge_rates_temporal[idx] = (
            author_rate if author_rate is not None else np.nan
        )

        # Sanity check: verify ALL entries, log first 50 as samples
        if filtered_closed:
            max_closed_at = max(
                datetime.fromisoformat(cp["closed_at"])
                for cp in filtered_closed
            )
            sanity_checks_total += 1
            ok = max_closed_at < cutoff
            if not ok:
                sanity_violations += 1
                logger.error(
                    "VIOLATION: %s closed_at=%s >= cutoff=%s",
                    login, max_closed_at, cutoff,
                )
            if len(sanity_samples) < 50:
                sanity_samples.append({
                    "login": login,
                    "pr_created_at": str(cutoff),
                    "max_closed_at": str(max_closed_at),
                    "ok": ok,
                })

        # Score with each mode
        if not filtered_data.merged_prs:
            # No merged PRs before cutoff => score is 0
            per_repo_scores[idx] = 0.0
            author_level_scores[idx] = 0.0
            continue

        for mode, arr in [
            ("per_repo", per_repo_scores),
            ("author_level", author_level_scores),
        ]:
            builder = RejectionAwareGraphBuilder(
                config=config,
                mode=mode,
                per_repo_merge_rates=per_repo_rates,
                author_merge_rate=author_rate,
            )
            try:
                score = _score_with_builder(
                    filtered_data, repo, config, builder,
                )
                arr[idx] = score
            except Exception:
                logger.debug(
                    "Scoring failed for %s PR #%s mode=%s",
                    repo, row["pr_number"], mode,
                )
                arr[idx] = row["normalized_score"]

    # Run sanity check (comprehensive: all entries verified, 50 logged)
    _run_sanity_check(sanity_samples, sanity_checks_total, sanity_violations)

    return pd.DataFrame({
        "per_repo_score": per_repo_scores,
        "author_level_score": author_level_scores,
        "author_merge_rate_temporal": merge_rates_temporal,
        "account_age_days": account_age_arr,
    })


def _run_sanity_check(
    samples: list[dict[str, Any]],
    total_checks: int,
    total_violations: int,
) -> None:
    """Verify anti-lookahead on closed PRs (comprehensive)."""
    logger.info(
        "Anti-lookahead sanity check: %d/%d passed (0 violations expected)",
        total_checks - total_violations, total_checks,
    )
    for s in samples[:5]:
        logger.info(
            "  login=%s pr_created=%s max_closed=%s ok=%s",
            s["login"], s["pr_created_at"],
            s["max_closed_at"], s["ok"],
        )
    if total_violations > 0:
        logger.error(
            "SANITY CHECK FAILED: %d violations found!",
            total_violations,
        )
        sample_violations = [s for s in samples if not s["ok"]]
        for v in sample_violations[:5]:
            logger.error(
                "  VIOLATION: login=%s pr_created=%s max_closed=%s",
                v["login"], v["pr_created_at"], v["max_closed_at"],
            )
        msg = (
            f"Anti-lookahead violation: {total_violations} "
            f"closed PRs with closed_at >= cutoff"
        )
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Log loss helper (matching stage6)
# ---------------------------------------------------------------------------

def log_loss_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean log loss."""
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -float(np.mean(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    ))


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def run_analysis(
    df: pd.DataFrame,
    scores_df: pd.DataFrame,
) -> dict[str, Any]:
    """Run full statistical analysis."""
    y = (df["outcome"] == "merged").astype(int).values
    full_scores = df["normalized_score"].values

    per_repo = scores_df["per_repo_score"].values
    author_level = scores_df["author_level_score"].values
    merge_rates = scores_df["author_merge_rate_temporal"].values

    # --- AUC for all approaches ---
    approaches: dict[str, np.ndarray] = {
        "Full model": full_scores,
        "Per-repo scaling": per_repo,
        "Author-level scaling": author_level,
    }

    # LR(GE + merge_rate) — cross-validated to avoid in-sample bias
    valid_mr = ~np.isnan(merge_rates)
    x_lr = np.column_stack([full_scores[valid_mr], merge_rates[valid_mr]])
    y_lr = y[valid_mr]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    lr = LogisticRegression(penalty=None, max_iter=1000, random_state=SEED)
    lr_proba_valid = cross_val_predict(
        lr, x_lr, y_lr, cv=cv, method="predict_proba",
    )[:, 1]
    # Expand to full array (NaN for missing)
    lr_scores = np.full(len(y), np.nan)
    lr_scores[valid_mr] = lr_proba_valid
    approaches["LR(GE + merge_rate)"] = lr_scores

    # LR(GE + merge_rate + age) — intermediate combined model
    account_age = scores_df["account_age_days"].values
    log_age = np.log(account_age + 1)
    valid_mra = valid_mr & ~np.isnan(log_age)
    x_mra = np.column_stack([
        full_scores[valid_mra],
        merge_rates[valid_mra],
        log_age[valid_mra],
    ])
    y_mra = y[valid_mra]
    cv_mra = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    lr_mra = LogisticRegression(
        penalty=None, max_iter=1000, random_state=SEED,
    )
    lr_mra_proba = cross_val_predict(
        lr_mra, x_mra, y_mra, cv=cv_mra, method="predict_proba",
    )[:, 1]
    lr_mra_scores = np.full(len(y), np.nan)
    lr_mra_scores[valid_mra] = lr_mra_proba
    approaches["LR(GE + merge_rate + age)"] = lr_mra_scores

    auc_results: dict[str, dict[str, Any]] = {}
    for name, scores in approaches.items():
        valid = ~np.isnan(scores)
        if valid.sum() < 50:
            continue
        auc_ci = auc_roc_with_ci(y[valid], scores[valid])
        auc_results[name] = {
            **auc_ci,
            "n": int(valid.sum()),
        }

    # Full model AUC on merge-rate subset for apples-to-apples LR comparison
    auc_ci_subset = auc_roc_with_ci(y[valid_mr], full_scores[valid_mr])
    auc_results["Full model (merge rate subset)"] = {
        **auc_ci_subset,
        "n": int(valid_mr.sum()),
    }

    # --- Pairwise DeLong tests ---
    delong_pairs = [
        ("Full model", "Per-repo scaling"),
        ("Full model", "Author-level scaling"),
        ("Full model", "LR(GE + merge_rate)"),
        ("Full model", "LR(GE + merge_rate + age)"),
        ("LR(GE + merge_rate)", "LR(GE + merge_rate + age)"),
    ]
    delong_results: dict[str, dict[str, Any]] = {}
    for name_a, name_b in delong_pairs:
        scores_a = approaches[name_a]
        scores_b = approaches[name_b]
        valid = ~np.isnan(scores_a) & ~np.isnan(scores_b)
        if valid.sum() < 50:
            continue
        result = delong_auc_test(
            y[valid], scores_a[valid], scores_b[valid],
        )
        delong_results[f"{name_a} vs {name_b}"] = result

    # Holm-Bonferroni correction
    delong_pvals = {
        k: v["p_value"] for k, v in delong_results.items()
    }
    delong_corrections = holm_bonferroni(delong_pvals)

    # LRT using out-of-fold log-likelihoods (consistent with CV AUC above)
    x_base = full_scores[valid_mr].reshape(-1, 1)
    lr_base_cv = LogisticRegression(
        penalty=None, max_iter=1000, random_state=SEED,
    )
    lr_base_proba = cross_val_predict(
        lr_base_cv, x_base, y_lr, cv=cv, method="predict_proba",
    )[:, 1]

    ll_base_cv = -log_loss_manual(y_lr, lr_base_proba) * len(y_lr)
    ll_full_cv = -log_loss_manual(y_lr, lr_proba_valid) * len(y_lr)

    lrt = likelihood_ratio_test(ll_base_cv, ll_full_cv, df_diff=1)

    # LRT for age: LR(GE + merge_rate + age) vs LR(GE + merge_rate)
    # on the subset where all three features are available
    lrt_age: dict[str, Any] | None = None
    if valid_mra.sum() >= 50:
        x_mr_sub = np.column_stack([
            full_scores[valid_mra], merge_rates[valid_mra],
        ])
        cv_age = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        lr_mr_sub = LogisticRegression(
            penalty=None, max_iter=1000, random_state=SEED,
        )
        lr_mr_sub_proba = cross_val_predict(
            lr_mr_sub, x_mr_sub, y_mra, cv=cv_age, method="predict_proba",
        )[:, 1]

        ll_mr_sub_cv = -log_loss_manual(y_mra, lr_mr_sub_proba) * len(y_mra)
        ll_mra_cv = -log_loss_manual(y_mra, lr_mra_proba) * len(y_mra)
        lrt_age = likelihood_ratio_test(ll_mr_sub_cv, ll_mra_cv, df_diff=1)

    # --- Subgroup: high-rejection authors (merge_rate < 0.5) ---
    high_rej_mask = merge_rates < 0.5
    high_rej_valid = high_rej_mask & ~np.isnan(merge_rates)
    subgroup_results: dict[str, Any] = {
        "n": int(high_rej_valid.sum()),
    }
    if high_rej_valid.sum() >= 30:
        for name, scores in approaches.items():
            valid = high_rej_valid & ~np.isnan(scores)
            if valid.sum() >= 30:
                auc_ci = auc_roc_with_ci(y[valid], scores[valid])
                subgroup_results[name] = {
                    **auc_ci,
                    "n": int(valid.sum()),
                }

    # --- Sensitivity: authors with vs without closed_prs ---
    has_closed = ~np.isnan(merge_rates)
    no_closed = np.isnan(merge_rates)
    sensitivity_data: dict[str, Any] = {
        "with_closed_n": int(has_closed.sum()),
        "without_closed_n": int(no_closed.sum()),
    }
    if has_closed.sum() >= 50:
        auc_with = auc_roc_with_ci(y[has_closed], full_scores[has_closed])
        sensitivity_data["with_closed_auc"] = auc_with
    if no_closed.sum() >= 50:
        auc_without = auc_roc_with_ci(y[no_closed], full_scores[no_closed])
        sensitivity_data["without_closed_auc"] = auc_without

    # --- Sensitivity: backfill truncation (>500 closed PRs) ---
    authors_dir = BASE_DIR / "data" / "raw" / "authors"
    truncated_authors = set()
    for login in df["author_login"].unique():
        path = authors_dir / f"{login}.json"
        if path.exists():
            raw = read_json(path)
            closed_count = raw.get("closed_count", 0)
            if closed_count >= BACKFILL_CAP:
                truncated_authors.add(login)

    truncated_mask = df["author_login"].isin(truncated_authors).values
    truncation_results: dict[str, Any] = {
        "n_authors_truncated": len(truncated_authors),
        "n_prs_truncated": int(truncated_mask.sum()),
    }
    if truncated_mask.sum() >= 30:
        for name, scores in approaches.items():
            valid = truncated_mask & ~np.isnan(scores)
            if valid.sum() >= 30:
                auc_ci = auc_roc_with_ci(y[valid], scores[valid])
                truncation_results[name] = {
                    **auc_ci,
                    "n": int(valid.sum()),
                }

    return {
        "auc_results": auc_results,
        "delong_results": delong_results,
        "delong_corrections": delong_corrections,
        "lrt_ge_merge_rate": lrt,
        "lrt_age_increment": lrt_age,
        "subgroup_high_rejection": subgroup_results,
        "sensitivity_closed_data": sensitivity_data,
        "sensitivity_truncation": truncation_results,
    }


# ---------------------------------------------------------------------------
# Data coverage stats
# ---------------------------------------------------------------------------

def compute_coverage_stats(
    df: pd.DataFrame,
    scores_df: pd.DataFrame,
    authors_dir: Path,
) -> dict[str, Any]:
    """Compute data coverage statistics."""
    total_authors = df["author_login"].nunique()

    authors_with_closed = 0
    authors_truncated = 0
    total_closed_prs = 0
    for login in df["author_login"].unique():
        path = authors_dir / f"{login}.json"
        if path.exists():
            raw = read_json(path)
            cp = raw.get("closed_prs", [])
            if cp:
                authors_with_closed += 1
                total_closed_prs += len(cp)
            closed_count = raw.get("closed_count", 0)
            if closed_count >= BACKFILL_CAP:
                authors_truncated += 1

    has_temporal_mr = (~np.isnan(
        scores_df["author_merge_rate_temporal"].values
    )).sum()

    return {
        "total_authors": total_authors,
        "authors_with_closed_prs": authors_with_closed,
        "authors_without_closed_prs": total_authors - authors_with_closed,
        "authors_truncated_at_cap": authors_truncated,
        "backfill_cap": BACKFILL_CAP,
        "total_closed_prs_loaded": total_closed_prs,
        "prs_with_temporal_merge_rate": int(has_temporal_mr),
        "prs_without_temporal_merge_rate": (
            len(df) - int(has_temporal_mr)
        ),
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _sig_marker(p: float) -> str:
    """Return significance marker."""
    if p < 0.001:
        return " ***"
    if p < 0.01:
        return " **"
    if p < 0.05:
        return " *"
    return ""


def generate_report(
    analysis: dict[str, Any],
    coverage: dict[str, Any],
    output_path: Path,
) -> None:
    """Generate the markdown report."""
    auc_results = analysis["auc_results"]
    delong_results = analysis["delong_results"]
    delong_corrections = analysis["delong_corrections"]
    lrt = analysis["lrt_ge_merge_rate"]
    subgroup = analysis["subgroup_high_rejection"]
    sensitivity = analysis["sensitivity_closed_data"]
    truncation = analysis["sensitivity_truncation"]

    lines = [
        "# Rejection Awareness Evaluation Report",
        "",
        "## Overview",
        "",
        "Survivorship bias is a potential concern in the Good Egg trust",
        "graph: the graph is built exclusively from *merged* PRs, ignoring",
        "rejected (closed-without-merge) contributions. This means an",
        "author who has 10 merged PRs out of 100 attempts looks identical",
        "to one who merged 10 out of 10. This experiment tests whether",
        "incorporating rejection data into the trust graph improves merge",
        "prediction.",
        "",
        "## Data Coverage",
        "",
        f"- **Total unique authors**: {coverage['total_authors']:,}",
        f"- **Authors with closed PR data**: "
        f"{coverage['authors_with_closed_prs']:,} "
        f"({coverage['authors_with_closed_prs'] / coverage['total_authors']:.1%})",
        f"- **Authors without closed PR data**: "
        f"{coverage['authors_without_closed_prs']:,}",
        f"- **Authors at backfill cap ({BACKFILL_CAP} closed PRs)**: "
        f"{coverage['authors_truncated_at_cap']:,}",
        f"- **Total closed PRs loaded**: "
        f"{coverage['total_closed_prs_loaded']:,}",
        f"- **PRs with temporally-scoped merge rate**: "
        f"{coverage['prs_with_temporal_merge_rate']:,}",
        f"- **PRs without (authors lacking closed data)**: "
        f"{coverage['prs_without_temporal_merge_rate']:,}",
        "",
    ]

    # --- AUC Comparison ---
    lines.extend([
        "## AUC Comparison",
        "",
        "| Approach | AUC | 95% CI | SE | n |",
        "|----------|-----|--------|----|----|",
    ])
    for name, res in auc_results.items():
        lines.append(
            f"| {name} | {res['auc']:.4f}"
            f" | [{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]"
            f" | {res['se']:.4f}"
            f" | {res['n']:,} |",
        )
    lines.append("")

    # --- Pairwise DeLong Tests ---
    lines.extend([
        "## Pairwise DeLong Tests",
        "",
        "| Comparison | AUC A | AUC B | z | Raw p"
        " | Adj. p |",
        "|------------|-------|-------|---|------"
        "|--------|",
    ])
    for pair_name, res in delong_results.items():
        adj = delong_corrections.get(pair_name, {}).get("adjusted_p")
        adj_str = f"{adj:.4e}" if adj is not None else "---"
        marker_p = adj if adj is not None else res["p_value"]
        sig = _sig_marker(marker_p)
        lines.append(
            f"| {pair_name}{sig} | {res['auc_a']:.4f}"
            f" | {res['auc_b']:.4f}"
            f" | {res['z_statistic']:.3f}"
            f" | {res['p_value']:.4e}"
            f" | {adj_str} |",
        )
    lines.append("")

    # --- Graph Integration vs Feature Engineering ---
    lines.extend([
        "## Graph Integration vs Feature Engineering",
        "",
        "Comparing graph-integrated edge scaling against logistic",
        "regression that uses merge rate as a separate feature.",
        "",
        f"- **LRT statistic (GE+merge_rate vs GE)**: "
        f"{lrt['lr_statistic']:.3f} (cross-validated)",
        f"- **LRT p-value**: {lrt['p_value']:.4e}",
        f"- **LRT df**: {lrt['df']}",
        "",
    ])

    per_repo_auc = auc_results.get("Per-repo scaling", {})
    lr_auc = auc_results.get("LR(GE + merge_rate)", {})
    if per_repo_auc and lr_auc:
        lines.extend([
            f"Per-repo scaling AUC = {per_repo_auc['auc']:.4f}, "
            f"LR(GE + merge_rate) AUC = {lr_auc['auc']:.4f}.",
            "",
        ])

    # Intermediate combined model
    lr_mra_auc = auc_results.get("LR(GE + merge_rate + age)", {})
    lrt_age = analysis.get("lrt_age_increment")
    if lr_mra_auc:
        lines.extend([
            "### Intermediate Combined Model (GE + merge_rate + age)",
            "",
            f"- **AUC**: {lr_mra_auc['auc']:.4f}"
            f" (95% CI: [{lr_mra_auc['ci_lower']:.4f},"
            f" {lr_mra_auc['ci_upper']:.4f}],"
            f" n={lr_mra_auc['n']:,})",
        ])
        if lrt_age:
            lines.extend([
                f"- **LRT (age increment over GE+merge_rate)**:"
                f" {lrt_age['lr_statistic']:.3f}"
                f" (p={lrt_age['p_value']:.4e}, cross-validated)",
            ])
        lines.extend([""])

    # --- Subgroup: High-Rejection Authors ---
    lines.extend([
        "## Subgroup: High-Rejection Authors",
        "",
        f"Authors with temporally-scoped merge rate < 0.5"
        f" (n={subgroup['n']:,}).",
        "",
    ])
    sub_approaches = {
        k: v for k, v in subgroup.items()
        if isinstance(v, dict) and "auc" in v
    }
    if sub_approaches:
        lines.extend([
            "| Approach | AUC | 95% CI | n |",
            "|----------|-----|--------|---|",
        ])
        for name, res in sub_approaches.items():
            lines.append(
                f"| {name} | {res['auc']:.4f}"
                f" | [{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]"
                f" | {res['n']:,} |",
            )
        lines.append("")
    else:
        lines.extend([
            "Insufficient data for subgroup analysis.",
            "",
        ])

    # --- Sensitivity: Missing Data ---
    lines.extend([
        "## Sensitivity: Missing Data",
        "",
        f"- **PRs from authors with closed data**: "
        f"{sensitivity['with_closed_n']:,}",
        f"- **PRs from authors without closed data**: "
        f"{sensitivity['without_closed_n']:,}",
        "",
    ])
    if "with_closed_auc" in sensitivity:
        a = sensitivity["with_closed_auc"]
        lines.append(
            f"- Full model AUC (with closed data): {a['auc']:.4f}"
            f" [{a['ci_lower']:.4f}, {a['ci_upper']:.4f}]",
        )
    if "without_closed_auc" in sensitivity:
        a = sensitivity["without_closed_auc"]
        lines.append(
            f"- Full model AUC (without closed data): {a['auc']:.4f}"
            f" [{a['ci_lower']:.4f}, {a['ci_upper']:.4f}]",
        )
    lines.append("")

    # --- Sensitivity: Backfill Truncation ---
    lines.extend([
        "## Sensitivity: Backfill Truncation",
        "",
        f"The closed PR backfill was capped at {BACKFILL_CAP} closed PRs"
        f" per author.",
        f"**{truncation['n_authors_truncated']}** authors hit this cap,",
        f"affecting **{truncation['n_prs_truncated']}** test PRs.",
        "",
        "For these authors, the merge rate may be biased upward"
        " (missing older",
        "rejections). If their true rejection rate is higher than"
        " observed,",
        "the scaling approaches would under-penalize their scores.",
        "",
    ])
    trunc_approaches = {
        k: v for k, v in truncation.items()
        if isinstance(v, dict) and "auc" in v
    }
    if trunc_approaches:
        lines.extend([
            "| Approach | AUC | 95% CI | n |",
            "|----------|-----|--------|---|",
        ])
        for name, res in trunc_approaches.items():
            lines.append(
                f"| {name} | {res['auc']:.4f}"
                f" | [{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]"
                f" | {res['n']:,} |",
            )
        lines.append("")

    # --- Conclusions ---
    lines.extend([
        "## Conclusions",
        "",
    ])

    # Check if any rejection-aware approach significantly differs
    any_significant = False
    for _pair_name, corr in delong_corrections.items():
        if corr.get("reject"):
            any_significant = True
            break

    if any_significant:
        lines.extend([
            "At least one rejection-aware approach shows a statistically",
            "significant difference in AUC compared to the full model.",
            "Incorporating rejection data may improve merge prediction.",
        ])
    else:
        lines.extend([
            "No rejection-aware approach shows a statistically significant",
            "improvement over the full model after Holm-Bonferroni",
            "correction. The survivorship bias in the trust graph does not",
            "appear to materially affect merge prediction in this dataset.",
        ])

    lines.extend([
        "",
        "---",
        "*Generated by evaluate_rejection_awareness.py*",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    logger.info("Saved report to %s", output_path)


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def generate_figure(
    df: pd.DataFrame,
    scores_df: pd.DataFrame,
    analysis: dict[str, Any],
    output_path: Path,
) -> None:
    """Generate 6-panel (3x2) figure."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    y = (df["outcome"] == "merged").astype(int).values
    full_scores = df["normalized_score"].values
    per_repo = scores_df["per_repo_score"].values
    merge_rates = scores_df["author_merge_rate_temporal"].values

    auc_results = analysis["auc_results"]

    # (A) Bar: AUC with CI for all 5 approaches
    ax = axes[0, 0]
    names = list(auc_results.keys())
    aucs = [auc_results[n]["auc"] for n in names]
    ci_lo = [auc_results[n]["ci_lower"] for n in names]
    ci_hi = [auc_results[n]["ci_upper"] for n in names]
    yerr_lo = [a - lo for a, lo in zip(aucs, ci_lo, strict=True)]
    yerr_hi = [hi - a for a, hi in zip(aucs, ci_hi, strict=True)]

    x_pos = np.arange(len(names))
    ax.bar(
        x_pos, aucs, color="#3498db", alpha=0.7,
        yerr=[yerr_lo, yerr_hi], capsize=4,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [n.replace(" ", "\n") for n in names],
        fontsize=7, rotation=0,
    )
    ax.set_ylabel("AUC-ROC")
    ax.set_title("(A) AUC Comparison with 95% CI")
    # Add value labels
    for i, v in enumerate(aucs):
        ax.text(i, v + yerr_hi[i] + 0.002, f"{v:.4f}", ha="center", fontsize=7)

    # (B) KDE: score distributions for full model vs per-repo
    ax = axes[0, 1]
    valid_pr = ~np.isnan(per_repo)
    merged_mask = y == 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sns.kdeplot(
            full_scores[merged_mask], ax=ax,
            color="#2ecc71", label="Full (merged)", linewidth=2,
        )
        sns.kdeplot(
            full_scores[~merged_mask], ax=ax,
            color="#e74c3c", label="Full (not merged)",
            linewidth=2, linestyle="--",
        )
        if valid_pr.sum() > 0:
            sns.kdeplot(
                per_repo[valid_pr & merged_mask], ax=ax,
                color="#3498db", label="Per-repo (merged)", linewidth=2,
            )
            sns.kdeplot(
                per_repo[valid_pr & ~merged_mask], ax=ax,
                color="#f39c12", label="Per-repo (not merged)",
                linewidth=2, linestyle="--",
            )
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_title("(B) Score Distributions: Full vs Per-repo")
    ax.legend(fontsize=7)

    # (C) Scatter: full vs per-repo scores, colored by outcome
    ax = axes[1, 0]
    valid = valid_pr
    colors = np.where(y[valid] == 1, "#2ecc71", "#e74c3c")
    ax.scatter(
        full_scores[valid], per_repo[valid],
        c=colors, alpha=0.3, s=10,
    )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("Full Model Score")
    ax.set_ylabel("Per-repo Score")
    ax.set_title("(C) Full vs Per-repo Scores")
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor="#2ecc71", markersize=6,
            label="Merged",
        ),
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor="#e74c3c", markersize=6,
            label="Not Merged",
        ),
    ]
    ax.legend(handles=legend_elements, fontsize=8)

    # (D) Histogram: per-author merge rate distribution
    ax = axes[1, 1]
    valid_mr = ~np.isnan(merge_rates)
    if valid_mr.sum() > 0:
        ax.hist(
            merge_rates[valid_mr], bins=30,
            color="#3498db", alpha=0.7, edgecolor="white",
        )
    ax.set_xlabel("Author Merge Rate (temporally scoped)")
    ax.set_ylabel("Count")
    ax.set_title("(D) Per-Author Merge Rate Distribution")
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.5, label="0.5 threshold")
    ax.legend(fontsize=8)

    # (E) Bar: mean score delta by merge rate quintile
    ax = axes[1 + 1, 0]
    if valid_mr.sum() > 100:
        mr_valid = merge_rates[valid_mr]
        delta_valid = per_repo[valid_mr] - full_scores[valid_mr]
        delta_nonan = ~np.isnan(delta_valid)
        if delta_nonan.sum() > 50:
            mr_q = mr_valid[delta_nonan]
            dq = delta_valid[delta_nonan]
            quintiles = pd.qcut(
                mr_q, 5, labels=False, duplicates="drop",
            )
            q_labels = []
            q_means = []
            q_sems = []
            for q in sorted(set(quintiles)):
                mask = quintiles == q
                lo = mr_q[mask].min()
                hi = mr_q[mask].max()
                q_labels.append(f"Q{q + 1}\n[{lo:.2f}-{hi:.2f}]")
                q_means.append(dq[mask].mean())
                q_sems.append(
                    dq[mask].std() / np.sqrt(mask.sum()),
                )
            x_q = np.arange(len(q_labels))
            ax.bar(
                x_q, q_means, yerr=q_sems,
                color="#e67e22", alpha=0.7, capsize=4,
            )
            ax.set_xticks(x_q)
            ax.set_xticklabels(q_labels, fontsize=7)
            ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Merge Rate Quintile")
    ax.set_ylabel("Mean Score Delta (Per-repo - Full)")
    ax.set_title("(E) Score Delta by Merge Rate Quintile")

    # (F) Scatter: merge rate vs score delta with LOWESS trend
    ax = axes[2, 1]
    if valid_mr.sum() > 50:
        delta = per_repo[valid_mr] - full_scores[valid_mr]
        delta_valid_mask = ~np.isnan(delta)
        mr_plot = merge_rates[valid_mr][delta_valid_mask]
        delta_plot = delta[delta_valid_mask]
        ax.scatter(
            mr_plot, delta_plot,
            alpha=0.2, s=8, color="#3498db",
        )
        # LOWESS trend
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            sorted_idx = np.argsort(mr_plot)
            mr_sorted = mr_plot[sorted_idx]
            delta_sorted = delta_plot[sorted_idx]
            lowess_result = lowess(
                delta_sorted, mr_sorted, frac=0.3,
            )
            ax.plot(
                lowess_result[:, 0], lowess_result[:, 1],
                color="#e74c3c", linewidth=2, label="LOWESS",
            )
            ax.legend(fontsize=8)
        except ImportError:
            logger.warning("statsmodels not available, skipping LOWESS")
        ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Author Merge Rate")
    ax.set_ylabel("Score Delta (Per-repo - Full)")
    ax.set_title("(F) Merge Rate vs Score Delta")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure to %s", output_path)


# ---------------------------------------------------------------------------
# Serialization helper
# ---------------------------------------------------------------------------

def _serialize(obj: Any) -> Any:
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full rejection awareness evaluation."""
    parquet_path = BASE_DIR / "data" / "scored" / "full_model.parquet"
    authors_dir = BASE_DIR / "data" / "raw" / "authors"

    if not parquet_path.exists():
        logger.error("Parquet file not found: %s", parquet_path)
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    logger.info("Loaded %d scored PRs", len(df))

    # Verify full model AUC
    y = (df["outcome"] == "merged").astype(int).values
    full_auc = auc_roc_with_ci(y, df["normalized_score"].values)
    logger.info("Full model AUC: %.4f", full_auc["auc"])

    # Score all PRs with rejection-aware approaches
    logger.info("Scoring PRs with rejection-aware approaches...")
    scores_df = score_all_prs(df, authors_dir)

    # Run analysis
    logger.info("Running statistical analysis...")
    analysis = run_analysis(df, scores_df)

    # Compute coverage stats
    coverage = compute_coverage_stats(df, scores_df, authors_dir)

    # Generate outputs
    output_dir = BASE_DIR / "results" / "rejection_awareness"
    output_dir.mkdir(parents=True, exist_ok=True)

    # results.json
    results_json = {
        "analysis": _serialize(analysis),
        "coverage": _serialize(coverage),
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2, default=str)
    logger.info("Saved results to %s", results_path)

    # report.md
    generate_report(analysis, coverage, output_dir / "report.md")

    # rejection_awareness.png
    generate_figure(
        df, scores_df, analysis,
        output_dir / "rejection_awareness.png",
    )

    logger.info("Done. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
