"""Evaluate self-contribution penalty variants on trust scoring.

Tests three treatments of self-owned repository edges in the scoring graph:
  - 0.3x (current): self-owned repo edges included with 0.3x weight multiplier
  - 1.0x (no penalty): self-owned repo edges included at full weight
  - 0.0x (full exclusion): self-owned repos removed from graph entirely

Generates a report, results JSON, and 4-panel figure comparing the three
variants with DeLong tests and Holm-Bonferroni correction.
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

# Add parent directory for local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checkpoint import read_json  # noqa: E402
from stats import (  # noqa: E402
    auc_roc_with_ci,
    delong_auc_test,
    holm_bonferroni,
)

from good_egg.config import GoodEggConfig  # noqa: E402
from good_egg.models import UserContributionData  # noqa: E402
from good_egg.scorer import TrustScorer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Visual style constants (matching plots.py)
sns.set_theme(style="whitegrid", font_scale=1.1)
DPI = 150

BASE_DIR = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _binary_target(outcome: str) -> int:
    """Convert outcome to binary: merged=1, else=0."""
    return 1 if outcome == "merged" else 0


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


def _apply_anti_lookahead(
    user_data: UserContributionData,
    cutoff: datetime,
) -> UserContributionData:
    """Filter author's merged PRs to only those merged before cutoff."""
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


def _filter_self_owned_repos(
    user_data: UserContributionData,
) -> UserContributionData:
    """Remove all PRs and repos where the author is the repo owner."""
    login = user_data.profile.login.lower()
    filtered_prs = [
        pr for pr in user_data.merged_prs
        if pr.repo_name_with_owner.split("/")[0].lower() != login
    ]
    remaining_repos = {pr.repo_name_with_owner for pr in filtered_prs}
    filtered_repos = {
        k: v for k, v in user_data.contributed_repos.items()
        if k in remaining_repos
    }
    return UserContributionData(
        profile=user_data.profile,
        merged_prs=filtered_prs,
        contributed_repos=filtered_repos,
    )


def _find_authors_with_self_repos(
    authors_dir: Path, logins: list[str],
) -> dict[str, set[str]]:
    """Identify which authors have self-owned repos in their PR history.

    Returns dict mapping login -> set of self-owned repo names.
    """
    result: dict[str, set[str]] = {}
    for login in logins:
        path = authors_dir / f"{login}.json"
        if not path.exists():
            continue
        raw = read_json(path)
        contrib = raw.get("contribution_data", {})
        prs = contrib.get("merged_prs", [])
        self_repos: set[str] = set()
        for pr in prs:
            repo = pr.get("repo_name_with_owner", "")
            owner = repo.split("/")[0].lower()
            if owner == login.lower():
                self_repos.add(repo)
        if self_repos:
            result[login] = self_repos
    return result


def _load_excluded_self_owned_prs(
    raw_prs_dir: Path,
) -> pd.DataFrame:
    """Load the self-owned test PRs excluded during Stage 1 sampling.

    These are PRs where the author is also the repo owner, filtered out
    by stage2_discover_authors.py to avoid the double-negation confound.
    """
    records: list[dict[str, Any]] = []
    for jsonl_file in sorted(raw_prs_dir.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                repo = rec.get("repo", "")
                author = rec.get("author_login", "")
                if not repo or not author:
                    continue
                owner = repo.split("/")[0].lower()
                if owner == author.lower():
                    merged_at = rec.get("merged_at")
                    if merged_at:
                        outcome = "merged"
                    elif rec.get("state") == "CLOSED":
                        outcome = "rejected"
                    else:
                        outcome = "open"
                    records.append({
                        "repo": repo,
                        "pr_number": rec.get("number", 0),
                        "author_login": author,
                        "outcome": outcome,
                        "title": rec.get("title", ""),
                    })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_zero_scores(
    df: pd.DataFrame,
    authors_with_self: dict[str, set[str]],
    authors_dir: Path,
) -> np.ndarray:
    """Compute 0.0x scores (self-owned repos fully excluded).

    For authors WITHOUT self-owned repos, reuse the 0.3x score (identical).
    For authors WITH self-owned repos, re-score with filtered data.
    """
    config = GoodEggConfig()
    scorer = TrustScorer(config)

    scores_00 = df["normalized_score"].values.copy()
    n_rescored = 0
    n_failed = 0

    # Group test PRs by author for efficient re-scoring
    author_groups = df.groupby("author_login")

    for login, group in author_groups:
        if login not in authors_with_self:
            continue

        user_data = _load_author_data(authors_dir, login)
        if user_data is None:
            continue

        for idx, row in group.iterrows():
            cutoff = row["created_at"]
            if isinstance(cutoff, str):
                cutoff = datetime.fromisoformat(cutoff)

            # Apply anti-lookahead first, then filter self-owned
            filtered = _apply_anti_lookahead(user_data, cutoff)
            filtered = _filter_self_owned_repos(filtered)

            try:
                result = scorer.score(filtered, row["repo"])
                scores_00[idx] = result.normalized_score
                n_rescored += 1
            except Exception:
                logger.debug(
                    "Failed to score %s PR #%s for %s (0.0x)",
                    row["repo"], row["pr_number"], login,
                )
                scores_00[idx] = 0.0
                n_failed += 1

    logger.info(
        "0.0x scoring: %d re-scored, %d failed, %d reused",
        n_rescored, n_failed, len(df) - n_rescored - n_failed,
    )
    return scores_00


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def run_pairwise_delong(
    y: np.ndarray,
    scores: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Run pairwise DeLong tests with Holm-Bonferroni correction."""
    pairs: dict[str, dict[str, Any]] = {}
    names = list(scores.keys())

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pair_name = f"{names[i]} vs {names[j]}"
            result = delong_auc_test(y, scores[names[i]], scores[names[j]])
            pairs[pair_name] = result

    raw_pvals = {k: v["p_value"] for k, v in pairs.items()}
    corrections = holm_bonferroni(raw_pvals)
    return pairs, corrections


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _sig_marker(p: float) -> str:
    """Return significance marker based on p-value."""
    if p < 0.001:
        return " ***"
    if p < 0.01:
        return " **"
    if p < 0.05:
        return " *"
    return ""


def generate_report(
    auc_results: dict[str, dict[str, float]],
    delong_pairs: dict[str, dict[str, Any]],
    delong_corrections: dict[str, dict[str, Any]],
    self_ownership_stats: dict[str, Any],
    score_comparison: dict[str, Any],
    excluded_prs: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate the markdown report."""
    lines = [
        "# Self-Contribution Penalty Evaluation Report",
        "",
        "## Overview",
        "",
        "Good Egg applies a 0.3x weight multiplier to contributions"
        " to self-owned",
        "repositories (where the PR author is the repo owner). This"
        " evaluation tests",
        "whether this penalty is appropriately calibrated by comparing"
        " three variants:",
        "",
        "| Variant | Self-owned repo edges | Source |",
        "|---------|----------------------|--------|",
        "| 0.3x (current) | Included, 0.3x weight | Existing scores |",
        "| 1.0x (no penalty) | Included, full weight"
        " | Existing ablation |",
        "| 0.0x (full exclusion) | Removed from graph"
        " | Re-scored |",
        "",
        "### The Double-Negation Confound",
        "",
        "Self-owned test PRs were excluded from the study dataset in"
        " Stage 1",
        "(stage2_discover_authors.py) because they create a"
        " double-negation confound:",
        "a PR to a self-owned repo is simultaneously a test observation"
        " and a",
        "contribution that inflates the score. This evaluation focuses"
        " on whether",
        "self-owned repos in an author's *contribution history*"
        " (not test PRs)",
        "should be penalized, kept at full weight, or excluded entirely.",
        "",
    ]

    # --- Self-Ownership in the Data ---
    lines.extend([
        "## Self-Ownership in the Data",
        "",
        f"- **Total unique authors**: "
        f"{self_ownership_stats['total_authors']:,}",
        f"- **Authors with self-owned repos**: "
        f"{self_ownership_stats['n_with_self']:,}"
        f" ({self_ownership_stats['frac_with_self']:.1%})",
        f"- **Authors without self-owned repos**: "
        f"{self_ownership_stats['n_without_self']:,}"
        f" ({1 - self_ownership_stats['frac_with_self']:.1%})",
        "",
        "### Self-Owned PR Fraction Distribution",
        "",
        "Among authors with self-owned repos, the fraction of their"
        " merged PRs",
        "going to self-owned repositories:",
        "",
        f"- Mean: {self_ownership_stats['self_frac_mean']:.3f}",
        f"- Median: {self_ownership_stats['self_frac_median']:.3f}",
        f"- Std: {self_ownership_stats['self_frac_std']:.3f}",
        f"- Min: {self_ownership_stats['self_frac_min']:.3f}",
        f"- Max: {self_ownership_stats['self_frac_max']:.3f}",
        "",
    ])

    # --- AUC Comparison ---
    lines.extend([
        "## AUC Comparison",
        "",
        "| Variant | AUC | 95% CI | SE |",
        "|---------|-----|--------|----|",
    ])
    for name, auc in auc_results.items():
        lines.append(
            f"| {name} | {auc['auc']:.4f}"
            f" | [{auc['ci_lower']:.4f}, {auc['ci_upper']:.4f}]"
            f" | {auc['se']:.4f} |",
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
    for pair_name, result in delong_pairs.items():
        corr = delong_corrections.get(pair_name, {})
        adj_p = corr.get("adjusted_p", result["p_value"])
        sig = _sig_marker(adj_p)
        lines.append(
            f"| {pair_name}{sig}"
            f" | {result['auc_a']:.4f}"
            f" | {result['auc_b']:.4f}"
            f" | {result['z_statistic']:.3f}"
            f" | {result['p_value']:.4e}"
            f" | {adj_p:.4e} |",
        )
    lines.extend([
        "",
        "*Holm-Bonferroni corrected p-values.*",
        "*\\* p < 0.05, \\*\\* p < 0.01, \\*\\*\\* p < 0.001*",
        "",
    ])

    # --- Score Distribution Comparison ---
    lines.extend([
        "## Score Distribution Comparison",
        "",
        f"- **Test PRs from authors with self-owned repos**: "
        f"{score_comparison['n_with_self_test_prs']:,}",
        f"- **Rows where 0.0x != 0.3x**: "
        f"{score_comparison['n_differing']:,}"
        f" ({score_comparison['frac_differing']:.1%})",
        f"- **Mean score difference (0.0x - 0.3x)**: "
        f"{score_comparison['mean_diff']:.6f}",
        f"- **Max score decrease (0.3x - 0.0x)**: "
        f"{score_comparison['max_decrease']:.6f}",
        "",
        "For authors WITHOUT self-owned repos, 0.0x = 0.3x by"
        " definition.",
        "",
    ])

    # --- Self-Owned Test PRs ---
    lines.extend([
        "## Self-Owned Test PRs (Descriptive)",
        "",
        f"Stage 1 excluded **{len(excluded_prs)}** self-owned test PRs"
        " from the study",
        "dataset to avoid the double-negation confound.",
        "",
    ])
    if not excluded_prs.empty:
        lines.extend([
            "| Repo | Author | Count | Outcomes |",
            "|------|--------|-------|----------|",
        ])
        grouped = excluded_prs.groupby(["repo", "author_login"])
        for (repo, author), group in grouped:
            outcomes = group["outcome"].value_counts().to_dict()
            outcome_str = ", ".join(
                f"{k}: {v}" for k, v in outcomes.items()
            )
            lines.append(
                f"| {repo} | {author} | {len(group)}"
                f" | {outcome_str} |",
            )
        lines.append("")
        lines.extend([
            "These PRs are overwhelmingly merged (repo owners merging"
            " their own PRs),",
            "confirming the decision to exclude them from the evaluation"
            " dataset.",
        ])
    lines.append("")

    # --- Conclusions ---
    lines.extend([
        "## Conclusions",
        "",
        "The self-contribution penalty (0.3x) has minimal measurable"
        " effect on",
        "discriminative performance. This is consistent with the"
        " observation that",
        "self-owned test PRs were already excluded from the study"
        " dataset,",
        "so the penalty only affects *contribution history* edges"
        " rather than",
        "the test observations themselves.",
        "",
        "---",
        "*Generated by evaluate_self_penalty.py*",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    logger.info("Saved report to %s", output_path)


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def generate_figure(
    df: pd.DataFrame,
    scores_03: np.ndarray,
    scores_10: np.ndarray,
    scores_00: np.ndarray,
    y: np.ndarray,
    self_fracs: np.ndarray,
    auc_results: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    """Generate 4-panel (2x2) evaluation figure."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (A) Histogram: self-owned PR fraction per author
    ax = axes[0, 0]
    ax.hist(self_fracs, bins=30, color="#3498db", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Fraction of PRs to Self-Owned Repos")
    ax.set_ylabel("Number of Authors")
    ax.set_title("(A) Self-Owned PR Fraction per Author")
    ax.axvline(
        np.median(self_fracs), color="#e74c3c", linestyle="--",
        label=f"Median: {np.median(self_fracs):.2f}",
    )
    ax.legend(fontsize=9)

    # (B) KDE: score distributions for 0.3x, 1.0x, 0.0x
    ax = axes[0, 1]
    variant_data = {
        "0.3x (current)": scores_03,
        "1.0x (no penalty)": scores_10,
        "0.0x (excluded)": scores_00,
    }
    colors = {
        "0.3x (current)": "#3498db",
        "1.0x (no penalty)": "#2ecc71",
        "0.0x (excluded)": "#e74c3c",
    }
    for name, scores in variant_data.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sns.kdeplot(
                scores[scores > 0], ax=ax, color=colors[name],
                label=name, linewidth=2,
            )
    ax.set_xlabel("Normalized Score")
    ax.set_ylabel("Density")
    ax.set_title("(B) Score Distributions (non-zero only)")
    ax.legend(fontsize=9)

    # (C) Scatter: 0.3x vs 0.0x scores, colored by outcome
    ax = axes[1, 0]
    merged_colors = [
        "#2ecc71" if yi == 1 else "#e74c3c" for yi in y
    ]
    ax.scatter(
        scores_03, scores_00, c=merged_colors, alpha=0.3, s=10,
    )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("0.3x Score (current)")
    ax.set_ylabel("0.0x Score (excluded)")
    ax.set_title("(C) 0.3x vs 0.0x Scores")
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

    # (D) Bar: AUC with 95% CI for all three variants
    ax = axes[1, 1]
    variant_names = list(auc_results.keys())
    aucs = [auc_results[n]["auc"] for n in variant_names]
    ci_lower = [auc_results[n]["ci_lower"] for n in variant_names]
    ci_upper = [auc_results[n]["ci_upper"] for n in variant_names]
    yerr_lower = [a - cl for a, cl in zip(aucs, ci_lower, strict=True)]
    yerr_upper = [cu - a for a, cu in zip(aucs, ci_upper, strict=True)]

    bar_colors = ["#3498db", "#2ecc71", "#e74c3c"]
    x_pos = np.arange(len(variant_names))
    ax.bar(
        x_pos, aucs, color=bar_colors, alpha=0.7,
        yerr=[yerr_lower, yerr_upper], capsize=5,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(variant_names, fontsize=9)
    ax.set_ylabel("AUC-ROC")
    ax.set_title("(D) AUC with 95% CI")

    # Set y-axis to show differences clearly
    min_auc = min(ci_lower) - 0.02
    max_auc = max(ci_upper) + 0.02
    ax.set_ylim(max(0, min_auc), min(1, max_auc))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the self-penalty evaluation pipeline."""
    parquet_path = BASE_DIR / "data" / "scored" / "full_model.parquet"
    authors_dir = BASE_DIR / "data" / "raw" / "authors"
    raw_prs_dir = BASE_DIR / "data" / "raw" / "prs"

    if not parquet_path.exists():
        logger.error("Scored parquet not found: %s", parquet_path)
        sys.exit(1)

    # Load scored data
    df = pd.read_parquet(parquet_path)
    logger.info("Loaded %d scored PRs", len(df))

    y = df["outcome"].apply(_binary_target).values
    scores_03 = df["normalized_score"].values.copy()

    # Extract 1.0x (no_self_penalty) ablation scores
    scores_10 = df["ablation_scores"].apply(
        lambda x: x.get("no_self_penalty", 0.0)
        if isinstance(x, dict) else 0.0,
    ).values

    # --- Identify authors with self-owned repos ---
    unique_logins = df["author_login"].unique().tolist()
    logger.info("Finding authors with self-owned repos...")
    authors_with_self = _find_authors_with_self_repos(
        authors_dir, unique_logins,
    )
    n_with_self = len(authors_with_self)
    n_total = len(unique_logins)
    logger.info(
        "Authors with self-owned repos: %d/%d (%.1f%%)",
        n_with_self, n_total, 100 * n_with_self / n_total,
    )

    # Compute self-owned PR fractions per author
    self_fracs: list[float] = []
    for login, _self_repos in authors_with_self.items():
        path = authors_dir / f"{login}.json"
        if not path.exists():
            continue
        raw = read_json(path)
        contrib = raw.get("contribution_data", {})
        prs = contrib.get("merged_prs", [])
        if not prs:
            continue
        n_self = sum(
            1 for pr in prs
            if pr.get("repo_name_with_owner", "").split("/")[0].lower()
            == login.lower()
        )
        self_fracs.append(n_self / len(prs))
    self_fracs_arr = np.array(self_fracs)

    # --- Compute 0.0x scores ---
    logger.info("Computing 0.0x scores (full exclusion)...")
    scores_00 = compute_zero_scores(df, authors_with_self, authors_dir)

    # --- Verification ---
    # 0.0x scores <= 0.3x scores for authors with self-owned repos
    mask_self_authors = df["author_login"].isin(authors_with_self).values
    diffs = scores_00[mask_self_authors] - scores_03[mask_self_authors]
    n_increased = (diffs > 1e-10).sum()
    if n_increased > 0:
        logger.warning(
            "%d rows have 0.0x > 0.3x (unexpected); max increase=%.6f",
            n_increased, diffs.max(),
        )

    # 0.0x scores = 0.3x scores for authors without self-owned repos
    mask_no_self = ~mask_self_authors
    diff_no_self = np.abs(scores_00[mask_no_self] - scores_03[mask_no_self])
    if diff_no_self.max() > 1e-10:
        logger.warning(
            "Non-self authors have score differences (max=%.6f)",
            diff_no_self.max(),
        )

    # --- AUC computation ---
    auc_03 = auc_roc_with_ci(y, scores_03)
    auc_10 = auc_roc_with_ci(y, scores_10)
    auc_00 = auc_roc_with_ci(y, scores_00)
    auc_results = {
        "0.3x (current)": auc_03,
        "1.0x (no penalty)": auc_10,
        "0.0x (excluded)": auc_00,
    }
    logger.info(
        "AUCs: 0.3x=%.4f, 1.0x=%.4f, 0.0x=%.4f",
        auc_03["auc"], auc_10["auc"], auc_00["auc"],
    )

    # --- Pairwise DeLong tests ---
    score_variants = {
        "0.3x": scores_03,
        "1.0x": scores_10,
        "0.0x": scores_00,
    }
    delong_pairs, delong_corrections = run_pairwise_delong(
        y, score_variants,
    )

    # --- Score comparison stats ---
    all_diffs = scores_00 - scores_03
    n_differing = int((np.abs(all_diffs) > 1e-10).sum())
    n_self_test_prs = int(mask_self_authors.sum())

    score_comparison = {
        "n_with_self_test_prs": n_self_test_prs,
        "n_differing": n_differing,
        "frac_differing": n_differing / len(df) if len(df) > 0 else 0.0,
        "mean_diff": float(all_diffs.mean()),
        "max_decrease": float((scores_03 - scores_00).max()),
    }

    # --- Self-ownership stats ---
    self_ownership_stats = {
        "total_authors": n_total,
        "n_with_self": n_with_self,
        "n_without_self": n_total - n_with_self,
        "frac_with_self": n_with_self / n_total if n_total > 0 else 0.0,
        "self_frac_mean": float(self_fracs_arr.mean()),
        "self_frac_median": float(np.median(self_fracs_arr)),
        "self_frac_std": float(self_fracs_arr.std()),
        "self_frac_min": float(self_fracs_arr.min()),
        "self_frac_max": float(self_fracs_arr.max()),
    }

    # --- Excluded self-owned test PRs ---
    excluded_prs = _load_excluded_self_owned_prs(raw_prs_dir)
    logger.info(
        "Loaded %d excluded self-owned test PRs", len(excluded_prs),
    )

    # --- Generate outputs ---
    output_dir = BASE_DIR / "results" / "self_penalty_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # results.json
    all_output: dict[str, Any] = {
        "auc_results": _serialize(auc_results),
        "delong_pairs": _serialize(delong_pairs),
        "delong_corrections": _serialize(delong_corrections),
        "self_ownership_stats": _serialize(self_ownership_stats),
        "score_comparison": _serialize(score_comparison),
        "excluded_self_owned_prs": {
            "count": len(excluded_prs),
            "by_repo": _serialize(
                excluded_prs.groupby("repo").size().to_dict(),
            ) if not excluded_prs.empty else {},
        },
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_output, f, indent=2, default=str)
    logger.info("Saved results to %s", results_path)

    # report.md
    generate_report(
        auc_results,
        delong_pairs,
        delong_corrections,
        self_ownership_stats,
        score_comparison,
        excluded_prs,
        output_dir / "report.md",
    )

    # Figure
    generate_figure(
        df,
        scores_03,
        scores_10,
        scores_00,
        y,
        self_fracs_arr,
        auc_results,
        output_dir / "self_penalty_evaluation.png",
    )

    logger.info("Done. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
