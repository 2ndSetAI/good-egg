from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, roc_curve

logger = logging.getLogger(__name__)

# Consistent style
sns.set_theme(style="whitegrid", font_scale=1.1)
FIGSIZE = (8, 6)
DPI = 150


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    auc_ci: tuple[float, float] | None = None,
    title: str = "ROC Curve",
    output_path: Path | None = None,
) -> None:
    """Plot ROC curve with optional CI band."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    label = f"GE Score (AUC = {roc_auc:.3f})"
    if auc_ci:
        label += f"\n95% CI: [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]"
    ax.plot(fpr, tpr, lw=2, label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
        logger.info("Saved ROC curve to %s", output_path)
    plt.close(fig)


def plot_score_distributions(
    scores_by_outcome: dict[str, np.ndarray],
    title: str = "GE Score Distribution by Outcome",
    output_path: Path | None = None,
) -> None:
    """Plot score distributions for each outcome class."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    colors = {
        "merged": "#2ecc71",
        "rejected": "#e74c3c",
        "pocket_veto": "#f39c12",
    }

    for outcome, scores in scores_by_outcome.items():
        color = colors.get(outcome)
        ax.hist(
            scores, bins=30, alpha=0.5, label=outcome.replace("_", " ").title(),
            color=color, density=True,
        )
        ax.axvline(
            np.median(scores), color=color or "gray",
            linestyle="--", alpha=0.7,
        )

    ax.set_xlabel("GE Normalized Score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
        logger.info("Saved score distributions to %s", output_path)
    plt.close(fig)


def plot_ablation_forest(
    ablation_results: dict[str, dict[str, float]],
    title: str = "Ablation Study: AUC-ROC Impact",
    output_path: Path | None = None,
) -> None:
    """Forest plot showing AUC change for each ablation.

    ablation_results: dict mapping ablation name to
        {auc: float, auc_diff: float, p_value: float}
    """
    names = list(ablation_results.keys())
    diffs = [ablation_results[n]["auc_diff"] for n in names]
    p_values = [ablation_results[n].get("p_value", 1.0) for n in names]

    # Sort by absolute effect size
    order = np.argsort([abs(d) for d in diffs])[::-1]
    names = [names[i] for i in order]
    diffs = [diffs[i] for i in order]
    p_values = [p_values[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.5)))
    y_pos = np.arange(len(names))

    colors = ["#e74c3c" if p < 0.05 else "#95a5a6" for p in p_values]
    ax.barh(y_pos, diffs, color=colors, alpha=0.7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [n.replace("_", " ").title() for n in names], fontsize=9,
    )
    ax.set_xlabel("AUC-ROC Change (ablation - full model)")
    ax.set_title(title)
    ax.invert_yaxis()

    # Annotate significance
    for i, (d, p) in enumerate(zip(diffs, p_values, strict=True)):
        marker = " *" if p < 0.05 else ""
        ax.text(
            d + 0.002 if d >= 0 else d - 0.002,
            i,
            f"{d:+.4f}{marker}",
            va="center",
            ha="left" if d >= 0 else "right",
            fontsize=8,
        )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
        logger.info("Saved ablation forest plot to %s", output_path)
    plt.close(fig)


def plot_calibration(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Plot",
    output_path: Path | None = None,
) -> None:
    """Plot calibration curve (reliability diagram)."""
    fraction_pos, mean_predicted = calibration_curve(
        y_true, y_scores, n_bins=n_bins, strategy="uniform",
    )

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(
        mean_predicted, fraction_pos, "o-", lw=2,
        label="GE Score",
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly Calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
        logger.info("Saved calibration plot to %s", output_path)
    plt.close(fig)


def plot_pocket_veto_by_trust(
    trust_levels: list[str],
    pocket_veto_rates: list[float],
    rejection_rates: list[float],
    title: str = "Outcome by Trust Level",
    output_path: Path | None = None,
) -> None:
    """Bar chart of pocket veto and rejection rates by trust level."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(trust_levels))
    width = 0.35

    ax.bar(
        x - width / 2, rejection_rates, width,
        label="Explicit Rejection", color="#e74c3c", alpha=0.7,
    )
    ax.bar(
        x + width / 2, pocket_veto_rates, width,
        label="Pocket Veto", color="#f39c12", alpha=0.7,
    )

    ax.set_xlabel("Trust Level")
    ax.set_ylabel("Rate")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(trust_levels)
    ax.legend()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
        logger.info("Saved pocket veto bar chart to %s", output_path)
    plt.close(fig)


def plot_baseline_comparison(
    labels: list[str],
    aucs: list[float],
    ci_lowers: list[float],
    ci_uppers: list[float],
    ge_auc: float,
    title: str = "AUC-ROC: GE Score vs. Baselines",
    output_path: Path | None = None,
) -> None:
    """Forest plot comparing GE score AUC against baselines."""
    fig, ax = plt.subplots(
        figsize=(10, max(4, len(labels) * 0.5)),
    )
    y_pos = np.arange(len(labels))
    errors = [
        [a - lo for a, lo in zip(aucs, ci_lowers, strict=True)],
        [hi - a for a, hi in zip(aucs, ci_uppers, strict=True)],
    ]

    colors = [
        "#3498db" if "GE" in lbl else "#95a5a6"
        for lbl in labels
    ]
    ax.barh(
        y_pos, aucs, xerr=errors, color=colors, alpha=0.7,
        capsize=3,
    )
    ax.axvline(ge_auc, color="#e74c3c", linewidth=1.5, linestyle="--",
               label=f"GE Score AUC = {ge_auc:.3f}", alpha=0.8)
    ax.axvline(0.5, color="gray", linewidth=0.8, linestyle=":",
               label="Random (0.500)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("AUC-ROC")
    ax.set_title(title)
    ax.set_xlim([0.4, max(max(aucs) + 0.05, ge_auc + 0.05)])
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
        logger.info("Saved baseline comparison to %s", output_path)
    plt.close(fig)


def plot_feature_importance(
    feature_names: list[str],
    importances: list[float],
    title: str = "Feature Importance",
    output_path: Path | None = None,
) -> None:
    """Horizontal bar chart of feature importances (coefficients)."""
    order = np.argsort(np.abs(importances))[::-1]
    names = [feature_names[i] for i in order]
    imps = [importances[i] for i in order]

    fig, ax = plt.subplots(
        figsize=(8, max(4, len(names) * 0.4)),
    )
    y_pos = np.arange(len(names))
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in imps]

    ax.barh(y_pos, imps, color=colors, alpha=0.7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Coefficient")
    ax.set_title(title)
    ax.invert_yaxis()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
        logger.info("Saved feature importance plot to %s", output_path)
    plt.close(fig)
