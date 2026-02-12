"""Compare similarity methods for PR-README text pairs.

Evaluates four similarity approaches (Gemini embeddings, TF-IDF, MiniLM,
Jaccard) on the same 1,293 PR-README text pairs and performs statistical
analysis to determine whether the choice of similarity method materially
affects the H4 finding.
"""
from __future__ import annotations

import json
import logging
import random
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

# Add parent directory for local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checkpoint import read_jsonl  # noqa: E402
from embedding import (  # noqa: E402
    _EMBEDDING_CACHE_DIR,
    _cache_key,
)
from embedding import (  # noqa: E402
    cosine_similarity as vec_cosine_similarity,
)
from stats import (  # noqa: E402
    auc_roc_with_ci,
    delong_auc_test,
    likelihood_ratio_test,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Visual style constants (matching plots.py)
sns.set_theme(style="whitegrid", font_scale=1.1)
FIGSIZE = (8, 6)
DPI = 150

BASE_DIR = Path(__file__).resolve().parents[1]
SEED = 42


# ---------------------------------------------------------------------------
# Stage 6 log_loss_manual (exact copy for LRT methodology match)
# ---------------------------------------------------------------------------

def log_loss_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean log loss (matches stage6_analyze.py implementation)."""
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -float(np.mean(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    ))


# ---------------------------------------------------------------------------
# Step 1: Reconstruct the 1,293 text pairs
# ---------------------------------------------------------------------------

def load_text_pairs(
    df: pd.DataFrame,
) -> tuple[list[str], list[str], pd.DataFrame]:
    """Load README and PR texts for rows with embedding_similarity.

    Returns (readme_texts, pr_texts, filtered_df) aligned by index.
    """
    repos_dir = BASE_DIR / "data" / "raw" / "repos"
    classified_dir = BASE_DIR / "data" / "raw" / "prs_classified"

    # Filter to rows with valid embedding similarity
    valid = df["embedding_similarity"].notna()
    df_sub = df.loc[valid].copy().reset_index(drop=True)
    logger.info(
        "Rows with embedding_similarity: %d", len(df_sub),
    )

    # Load repo READMEs
    readme_texts_map: dict[str, str] = {}
    for repo_name in df_sub["repo"].unique():
        owner, name = repo_name.split("/", 1)
        readme_path = repos_dir / f"{owner}__{name}_readme.md"
        if readme_path.exists():
            text = readme_path.read_text(errors="replace").strip()
            if text:
                readme_texts_map[repo_name] = text[:4000]

    # Load PR bodies from JSONL (needed for title fallback)
    pr_body_map: dict[tuple[str, int], str] = {}
    pr_title_map: dict[tuple[str, int], str] = {}
    for repo_name in df_sub["repo"].unique():
        owner, name = repo_name.split("/", 1)
        cpath = classified_dir / f"{owner}__{name}.jsonl"
        if cpath.exists():
            for rec in read_jsonl(cpath):
                key = (repo_name, rec["number"])
                pr_body_map[key] = rec.get("body", "")
                pr_title_map[key] = rec.get("title", "")

    # Build aligned text lists
    readme_texts: list[str] = []
    pr_texts: list[str] = []
    keep_mask: list[bool] = []

    for _, row in df_sub.iterrows():
        repo_name = row["repo"]
        pr_key = (repo_name, row["pr_number"])

        readme = readme_texts_map.get(repo_name)
        body = pr_body_map.get(pr_key, "")
        title = pr_title_map.get(pr_key, "")

        pr_text = body.strip() if body else ""
        if not pr_text:
            pr_text = title.strip()
        if pr_text:
            pr_text = pr_text[:2000]

        if readme and pr_text:
            readme_texts.append(readme)
            pr_texts.append(pr_text)
            keep_mask.append(True)
        else:
            keep_mask.append(False)

    df_sub = df_sub.loc[keep_mask].reset_index(drop=True)
    logger.info("Text pairs reconstructed: %d", len(readme_texts))
    return readme_texts, pr_texts, df_sub


def sanity_check_gemini(
    readme_texts: list[str],
    pr_texts: list[str],
    df_sub: pd.DataFrame,
    n_samples: int = 10,
) -> None:
    """Verify Gemini embeddings from cache match parquet values."""
    cache_dir = _EMBEDDING_CACHE_DIR / "gemini-embedding-001"
    if not cache_dir.exists():
        logger.warning("Gemini cache dir not found, skipping sanity check")
        return

    rng = random.Random(SEED)
    indices = rng.sample(range(len(df_sub)), min(n_samples, len(df_sub)))
    model = "gemini-embedding-001"
    mismatches = 0

    for idx in indices:
        readme = readme_texts[idx]
        pr_text = pr_texts[idx]
        expected = df_sub.iloc[idx]["embedding_similarity"]

        # Load cached embeddings
        readme_key = _cache_key(readme, model)
        pr_key = _cache_key(pr_text, model)
        readme_cache = cache_dir / f"{readme_key}.json"
        pr_cache = cache_dir / f"{pr_key}.json"

        if not readme_cache.exists() or not pr_cache.exists():
            logger.warning(
                "Cache miss for idx %d, skipping", idx,
            )
            continue

        readme_emb = np.array(
            json.loads(readme_cache.read_text())["embedding"],
            dtype=np.float32,
        )
        pr_emb = np.array(
            json.loads(pr_cache.read_text())["embedding"],
            dtype=np.float32,
        )
        recomputed = vec_cosine_similarity(readme_emb, pr_emb)

        if abs(recomputed - expected) > 1e-6:
            logger.warning(
                "Mismatch at idx %d: expected=%.8f, got=%.8f",
                idx, expected, recomputed,
            )
            mismatches += 1

    if mismatches == 0:
        logger.info(
            "Sanity check passed: %d samples match parquet values",
            len(indices),
        )
    else:
        logger.warning(
            "Sanity check: %d/%d mismatches", mismatches, len(indices),
        )


# ---------------------------------------------------------------------------
# Step 2: Compute similarities
# ---------------------------------------------------------------------------

def compute_tfidf_similarities(
    readme_texts: list[str],
    pr_texts: list[str],
) -> np.ndarray:
    """Compute TF-IDF cosine similarities for each text pair."""
    all_texts = readme_texts + pr_texts
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        sublinear_tf=True,
        min_df=2,
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    n = len(readme_texts)

    sims = np.zeros(n)
    for i in range(n):
        sim = sklearn_cosine(
            tfidf_matrix[i], tfidf_matrix[n + i],
        )
        sims[i] = float(sim[0, 0])
    return sims


def compute_minilm_similarities(
    readme_texts: list[str],
    pr_texts: list[str],
) -> np.ndarray | None:
    """Compute MiniLM ONNX similarities for each text pair.

    Returns None if onnxruntime is not available.
    """
    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except ImportError:
        logger.warning(
            "onnxruntime or tokenizers not installed, skipping MiniLM",
        )
        return None

    model_dir = BASE_DIR / "data" / "models" / "all-MiniLM-L6-v2"
    model_path = model_dir / "model.onnx"
    tokenizer_path = model_dir / "tokenizer.json"

    if not model_path.exists() or not tokenizer_path.exists():
        logger.warning(
            "MiniLM model files not found at %s, skipping", model_dir,
        )
        return None

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=128)
    tokenizer.enable_padding(length=128)

    session = ort.InferenceSession(str(model_path))

    def _embed_batch(texts: list[str], batch_size: int = 64) -> np.ndarray:
        all_embeddings: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encodings = tokenizer.encode_batch(batch)

            input_ids = np.array(
                [e.ids for e in encodings], dtype=np.int64,
            )
            attention_mask = np.array(
                [e.attention_mask for e in encodings], dtype=np.int64,
            )
            token_type_ids = np.zeros_like(input_ids)

            outputs = session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                },
            )
            # Mean pooling over token embeddings
            token_embeddings = outputs[0]  # (batch, seq_len, hidden)
            mask_expanded = attention_mask[
                :, :, np.newaxis
            ].astype(np.float32)
            summed = np.sum(token_embeddings * mask_expanded, axis=1)
            counts = np.sum(mask_expanded, axis=1).clip(min=1e-9)
            mean_pooled = summed / counts

            # L2 normalize
            norms = np.linalg.norm(
                mean_pooled, axis=1, keepdims=True,
            ).clip(min=1e-9)
            normalized = mean_pooled / norms
            all_embeddings.append(normalized)

        return np.vstack(all_embeddings)

    logger.info("Computing MiniLM embeddings for %d texts...", len(readme_texts) * 2)
    readme_embs = _embed_batch(readme_texts)
    pr_embs = _embed_batch(pr_texts)

    # Cosine similarity (already L2-normed, so dot product suffices)
    sims = np.sum(readme_embs * pr_embs, axis=1)
    return sims


def compute_jaccard_similarities(
    readme_texts: list[str],
    pr_texts: list[str],
) -> np.ndarray:
    """Compute Jaccard similarity on lowercased word sets."""
    n = len(readme_texts)
    sims = np.zeros(n)
    for i in range(n):
        tokens_a = set(readme_texts[i].lower().split())
        tokens_b = set(pr_texts[i].lower().split())
        union = tokens_a | tokens_b
        if len(union) == 0:
            sims[i] = 0.0
        else:
            sims[i] = len(tokens_a & tokens_b) / len(union)
    return sims


# ---------------------------------------------------------------------------
# Step 3: Statistical analysis
# ---------------------------------------------------------------------------

def _binary_target(outcome: str) -> int:
    """Convert outcome to binary: merged=1, else=0."""
    return 1 if outcome == "merged" else 0


def run_h4_lrt(
    y: np.ndarray,
    ge_scores: np.ndarray,
    similarities: np.ndarray,
) -> dict[str, Any]:
    """Run H4-style LRT: LR(GE+sim) vs LR(GE) alone.

    Matches stage6_analyze.py methodology using log_loss_manual.
    """
    x_base = ge_scores.reshape(-1, 1)
    x_full = np.column_stack([ge_scores, similarities])

    lr_base = LogisticRegression(
        penalty=None, max_iter=1000, random_state=SEED,
    )
    lr_base.fit(x_base, y)
    ll_base = -log_loss_manual(
        y, lr_base.predict_proba(x_base)[:, 1],
    ) * len(y)

    lr_full = LogisticRegression(
        penalty=None, max_iter=1000, random_state=SEED,
    )
    lr_full.fit(x_full, y)
    ll_full = -log_loss_manual(
        y, lr_full.predict_proba(x_full)[:, 1],
    ) * len(y)

    lrt = likelihood_ratio_test(ll_base, ll_full, df_diff=1)
    return lrt


def run_marginal_improvement(
    y: np.ndarray,
    ge_scores: np.ndarray,
    merge_rate: np.ndarray,
    age: np.ndarray,
    similarities: np.ndarray,
) -> dict[str, Any]:
    """Marginal improvement: AUC of LR(GE+MR+age+sim) vs LR(GE+MR+age).

    Uses log_loss_manual for LRT and paired DeLong for AUC comparison.
    """
    x_base = np.column_stack([ge_scores, merge_rate, age])
    x_full = np.column_stack([ge_scores, merge_rate, age, similarities])

    lr_base = LogisticRegression(
        penalty=None, max_iter=1000, random_state=SEED,
    )
    lr_base.fit(x_base, y)
    proba_base = lr_base.predict_proba(x_base)[:, 1]

    lr_full = LogisticRegression(
        penalty=None, max_iter=1000, random_state=SEED,
    )
    lr_full.fit(x_full, y)
    proba_full = lr_full.predict_proba(x_full)[:, 1]

    auc_base = auc_roc_with_ci(y, proba_base)
    auc_full = auc_roc_with_ci(y, proba_full)
    delong = delong_auc_test(y, proba_base, proba_full)

    # Also compute LRT for the marginal model
    ll_base = -log_loss_manual(y, proba_base) * len(y)
    ll_full = -log_loss_manual(y, proba_full) * len(y)
    lrt = likelihood_ratio_test(ll_base, ll_full, df_diff=1)

    return {
        "auc_base": auc_base["auc"],
        "auc_full": auc_full["auc"],
        "auc_diff": auc_full["auc"] - auc_base["auc"],
        "delong_z": delong["z_statistic"],
        "delong_p": delong["p_value"],
        "lrt_statistic": lrt["lr_statistic"],
        "lrt_p": lrt["p_value"],
    }


def analyze_method(
    name: str,
    similarities: np.ndarray,
    y: np.ndarray,
    ge_scores: np.ndarray,
    merge_rate: np.ndarray,
    age: np.ndarray,
    gemini_sims: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run full statistical analysis for one similarity method."""
    results: dict[str, Any] = {"name": name, "n": len(y)}

    # H4-style LRT
    lrt = run_h4_lrt(y, ge_scores, similarities)
    results["lrt_statistic"] = lrt["lr_statistic"]
    results["lrt_p_value"] = lrt["p_value"]

    # Standalone AUC-ROC with DeLong CI
    auc_ci = auc_roc_with_ci(y, similarities)
    results["standalone_auc"] = auc_ci["auc"]
    results["standalone_auc_ci_lower"] = auc_ci["ci_lower"]
    results["standalone_auc_ci_upper"] = auc_ci["ci_upper"]
    results["standalone_auc_se"] = auc_ci["se"]

    # Correlation with Gemini (for non-Gemini methods)
    if gemini_sims is not None:
        pearson_r, pearson_p = sp_stats.pearsonr(
            similarities, gemini_sims,
        )
        spearman_r, spearman_p = sp_stats.spearmanr(
            similarities, gemini_sims,
        )
        results["pearson_r_gemini"] = float(pearson_r)
        results["pearson_p_gemini"] = float(pearson_p)
        results["spearman_r_gemini"] = float(spearman_r)
        results["spearman_p_gemini"] = float(spearman_p)

    # Marginal improvement in combined model
    marginal = run_marginal_improvement(
        y, ge_scores, merge_rate, age, similarities,
    )
    results["marginal"] = marginal

    return results


# ---------------------------------------------------------------------------
# Step 4: Generate outputs
# ---------------------------------------------------------------------------

def generate_figure(
    method_results: dict[str, dict[str, Any]],
    method_sims: dict[str, np.ndarray],
    y: np.ndarray,
    output_path: Path,
) -> None:
    """Generate 4-panel comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (A) Forest plot: LRT statistics with significance markers
    ax = axes[0, 0]
    names = list(method_results.keys())
    lrt_stats = [method_results[n]["lrt_statistic"] for n in names]
    lrt_ps = [method_results[n]["lrt_p_value"] for n in names]
    y_pos = np.arange(len(names))
    colors = [
        "#e74c3c" if p < 0.05 else "#95a5a6" for p in lrt_ps
    ]
    ax.barh(y_pos, lrt_stats, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("LRT Statistic")
    ax.set_title("(A) H4 Likelihood Ratio Test by Method")
    ax.invert_yaxis()
    for i, (stat, p) in enumerate(zip(lrt_stats, lrt_ps, strict=True)):
        marker = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""
        ax.text(
            stat + max(lrt_stats) * 0.02, i,
            f"{stat:.2f}{marker}",
            va="center", fontsize=8,
        )

    # (B) Similarity distributions (KDE) overlaid by method
    ax = axes[0, 1]
    palette = {
        "Gemini": "#3498db",
        "TF-IDF": "#2ecc71",
        "MiniLM": "#e74c3c",
        "Jaccard": "#f39c12",
    }
    for name, sims in method_sims.items():
        color = palette.get(name, "#95a5a6")
        ax.hist(
            sims, bins=50, alpha=0.3, color=color, density=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sns.kdeplot(sims, ax=ax, color=color, label=name, linewidth=2)
    ax.set_xlabel("Similarity Score")
    ax.set_ylabel("Density")
    ax.set_title("(B) Similarity Distributions by Method")
    ax.legend(fontsize=8)

    # (C) Scatter: TF-IDF vs Gemini
    ax = axes[1, 0]
    if "TF-IDF" in method_sims and "Gemini" in method_sims:
        merged_colors = [
            "#2ecc71" if yi == 1 else "#e74c3c" for yi in y
        ]
        ax.scatter(
            method_sims["Gemini"], method_sims["TF-IDF"],
            c=merged_colors, alpha=0.3, s=10,
        )
        r = method_results.get("TF-IDF", {}).get(
            "pearson_r_gemini", float("nan"),
        )
        ax.set_xlabel("Gemini Similarity")
        ax.set_ylabel("TF-IDF Similarity")
        ax.set_title(f"(C) TF-IDF vs Gemini (r={r:.3f})")
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#2ecc71", markersize=6,
                   label="Merged"),
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#e74c3c", markersize=6,
                   label="Not Merged"),
        ]
        ax.legend(handles=legend_elements, fontsize=8)

    # (D) Scatter: MiniLM vs Gemini
    ax = axes[1, 1]
    if "MiniLM" in method_sims and "Gemini" in method_sims:
        ax.scatter(
            method_sims["Gemini"], method_sims["MiniLM"],
            c=merged_colors, alpha=0.3, s=10,
        )
        r = method_results.get("MiniLM", {}).get(
            "pearson_r_gemini", float("nan"),
        )
        ax.set_xlabel("Gemini Similarity")
        ax.set_ylabel("MiniLM Similarity")
        ax.set_title(f"(D) MiniLM vs Gemini (r={r:.3f})")
        ax.legend(handles=legend_elements, fontsize=8)
    elif "Jaccard" in method_sims and "Gemini" in method_sims:
        # Fallback: Jaccard vs Gemini if MiniLM unavailable
        ax.scatter(
            method_sims["Gemini"], method_sims["Jaccard"],
            c=merged_colors, alpha=0.3, s=10,
        )
        r = method_results.get("Jaccard", {}).get(
            "pearson_r_gemini", float("nan"),
        )
        ax.set_xlabel("Gemini Similarity")
        ax.set_ylabel("Jaccard Similarity")
        ax.set_title(f"(D) Jaccard vs Gemini (r={r:.3f})")
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#2ecc71", markersize=6,
                   label="Merged"),
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#e74c3c", markersize=6,
                   label="Not Merged"),
        ]
        ax.legend(handles=legend_elements, fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved comparison figure to %s", output_path)


def generate_report(
    method_results: dict[str, dict[str, Any]],
    pairwise_delong: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate markdown comparison report."""
    lines = [
        "# Similarity Method Comparison Report",
        "",
        "## Overview",
        "",
        "This report compares four text similarity methods applied to the",
        "same 1,293 PR body / repo README text pairs used in the H4",
        "semantic similarity hypothesis test.",
        "",
        "## Method Summary",
        "",
        "| Method | LRT Stat | LRT p-value | Standalone AUC"
        " | AUC 95% CI | Gemini r (Pearson) |",
        "|--------|----------|-------------|---------------"
        "|------------|-------------------|",
    ]

    for name, res in method_results.items():
        lrt_s = f"{res['lrt_statistic']:.3f}"
        lrt_p = f"{res['lrt_p_value']:.4e}"
        auc = f"{res['standalone_auc']:.4f}"
        ci = (
            f"[{res['standalone_auc_ci_lower']:.4f},"
            f" {res['standalone_auc_ci_upper']:.4f}]"
        )
        gem_r = res.get("pearson_r_gemini")
        gem_str = f"{gem_r:.4f}" if gem_r is not None else "---"
        sig = " *" if res["lrt_p_value"] < 0.05 else ""
        lines.append(
            f"| {name}{sig} | {lrt_s} | {lrt_p} | {auc}"
            f" | {ci} | {gem_str} |",
        )

    lines.extend([
        "",
        "*\\* p < 0.05*",
        "",
        "## Marginal Improvement (GE + merge_rate + age + sim)",
        "",
        "| Method | Base AUC | Full AUC | AUC Diff"
        " | DeLong p | LRT p |",
        "|--------|----------|----------|----------"
        "|---------| ------|",
    ])

    for name, res in method_results.items():
        m = res.get("marginal", {})
        if m:
            lines.append(
                f"| {name} | {m['auc_base']:.4f}"
                f" | {m['auc_full']:.4f}"
                f" | {m['auc_diff']:+.4f}"
                f" | {m['delong_p']:.4e}"
                f" | {m['lrt_p']:.4e} |",
            )

    lines.extend([
        "",
        "## Pairwise DeLong Tests (Standalone AUCs)",
        "",
        "| Comparison | AUC A | AUC B | z | p-value |",
        "|------------|-------|-------|---|---------|",
    ])

    for pair_name, res in pairwise_delong.items():
        sig = " *" if res["p_value"] < 0.05 else ""
        lines.append(
            f"| {pair_name}{sig} | {res['auc_a']:.4f}"
            f" | {res['auc_b']:.4f}"
            f" | {res['z_statistic']:.3f}"
            f" | {res['p_value']:.4e} |",
        )

    lines.extend([
        "",
        "## Decision Criteria Evaluation",
        "",
        "The key question is whether the H4 finding (semantic similarity",
        "adds predictive value beyond GE score alone) is robust to the",
        "choice of similarity method.",
        "",
        "**Criteria:**",
        "1. If all methods yield significant LRT p-values (< 0.05),"
        " the finding is robust.",
        "2. If only Gemini is significant, the finding may be an artifact"
        " of the embedding model.",
        "3. If no method is significant, the finding is likely spurious.",
        "",
    ])

    # Automated evaluation
    significant = [
        name for name, res in method_results.items()
        if res["lrt_p_value"] < 0.05
    ]
    total = len(method_results)
    if len(significant) == total:
        lines.append(
            f"**Result:** All {total} methods yield significant LRT"
            " (p < 0.05). The H4 finding is robust to similarity method"
            " choice.",
        )
    elif len(significant) > 1:
        lines.append(
            f"**Result:** {len(significant)}/{total} methods yield"
            f" significant LRT: {', '.join(significant)}. The H4 finding"
            " is partially robust.",
        )
    elif len(significant) == 1:
        lines.append(
            f"**Result:** Only {significant[0]} yields a significant LRT."
            " The H4 finding may be method-specific.",
        )
    else:
        lines.append(
            "**Result:** No method yields a significant LRT."
            " The H4 finding is not supported.",
        )

    lines.extend([
        "",
        "## Known Limitations",
        "",
        "- Gemini similarities were computed during feature engineering"
        " (not recomputed here); other methods are computed fresh.",
        "- MiniLM uses 128-token truncation vs. Gemini's larger context"
        " window, which may disadvantage it on longer texts.",
        "- TF-IDF is fitted on the full corpus (all 2,586 texts),"
        " which could introduce minor data leakage in a strict"
        " train/test sense, but this is acceptable for a comparison"
        " study where all methods see the same data.",
        "- Jaccard is a bag-of-words baseline with no semantic"
        " understanding.",
        "",
        "---",
        "*Generated by compare_similarity_methods.py*",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    logger.info("Saved report to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full comparison pipeline."""
    features_path = BASE_DIR / "data" / "features" / "features.parquet"
    if not features_path.exists():
        logger.error("Features file not found: %s", features_path)
        sys.exit(1)

    df = pd.read_parquet(features_path)
    logger.info("Loaded %d feature rows", len(df))

    # Step 1: Reconstruct text pairs
    readme_texts, pr_texts, df_sub = load_text_pairs(df)

    if len(df_sub) == 0:
        logger.error("No valid text pairs found")
        sys.exit(1)

    # Sanity check Gemini embeddings
    sanity_check_gemini(readme_texts, pr_texts, df_sub)

    # Prepare common arrays
    y = df_sub["outcome"].apply(_binary_target).values
    ge_scores = df_sub["normalized_score"].values
    gemini_sims = df_sub["embedding_similarity"].values

    # For marginal improvement analysis, we need merge_rate and age
    # on the same subset. Handle missing values by requiring all present.
    has_extras = (
        df_sub["author_merge_rate"].notna()
        & df_sub["log_account_age_days"].notna()
        & (df_sub["log_account_age_days"] > 0)
    )
    merge_rate = df_sub["author_merge_rate"].values
    age = df_sub["log_account_age_days"].values

    # Fill NaN for marginal analysis (will be masked per-method)
    merge_rate_clean = np.where(has_extras, merge_rate, 0.0)
    age_clean = np.where(has_extras, age, 0.0)

    # Step 2: Compute similarities
    logger.info("Computing TF-IDF similarities...")
    tfidf_sims = compute_tfidf_similarities(readme_texts, pr_texts)

    logger.info("Computing MiniLM similarities...")
    minilm_sims = compute_minilm_similarities(readme_texts, pr_texts)

    logger.info("Computing Jaccard similarities...")
    jaccard_sims = compute_jaccard_similarities(readme_texts, pr_texts)

    # Build method dictionary
    method_sims: dict[str, np.ndarray] = {
        "Gemini": gemini_sims,
        "TF-IDF": tfidf_sims,
    }
    if minilm_sims is not None:
        method_sims["MiniLM"] = minilm_sims
    method_sims["Jaccard"] = jaccard_sims

    # Step 3: Statistical analysis
    method_results: dict[str, dict[str, Any]] = {}

    for name, sims in method_sims.items():
        logger.info("Analyzing method: %s", name)

        gem_ref = gemini_sims if name != "Gemini" else None

        # Use the valid-extras subset for marginal analysis
        # but full subset for LRT and standalone AUC
        result = analyze_method(
            name=name,
            similarities=sims,
            y=y,
            ge_scores=ge_scores,
            merge_rate=merge_rate_clean,
            age=age_clean,
            gemini_sims=gem_ref,
        )

        # Rerun marginal with clean subset only
        if has_extras.sum() > 50:
            mask = has_extras.values
            marginal = run_marginal_improvement(
                y[mask],
                ge_scores[mask],
                merge_rate[mask],
                age[mask],
                sims[mask],
            )
            result["marginal"] = marginal

        method_results[name] = result

    # Pairwise DeLong tests between standalone AUCs
    pairwise_delong: dict[str, dict[str, Any]] = {}
    method_names = list(method_sims.keys())
    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            name_a = method_names[i]
            name_b = method_names[j]
            delong = delong_auc_test(
                y,
                method_sims[name_a],
                method_sims[name_b],
            )
            pairwise_delong[f"{name_a} vs {name_b}"] = delong

    # Step 4: Generate outputs
    output_dir = (
        BASE_DIR / "results" / "similarity_comparison"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # results.json
    all_output: dict[str, Any] = {
        "methods": {
            k: _serialize(v) for k, v in method_results.items()
        },
        "pairwise_delong": {
            k: _serialize(v) for k, v in pairwise_delong.items()
        },
        "n_pairs": len(df_sub),
        "n_with_extras": int(has_extras.sum()),
    }
    results_json_path = output_dir / "results.json"
    with open(results_json_path, "w") as f:
        json.dump(all_output, f, indent=2, default=str)
    logger.info("Saved results to %s", results_json_path)

    # comparison_report.md
    generate_report(
        method_results, pairwise_delong,
        output_dir / "comparison_report.md",
    )

    # similarity_comparison.png
    generate_figure(
        method_results, method_sims, y,
        output_dir / "similarity_comparison.png",
    )

    logger.info("Done. Outputs in %s", output_dir)


def _serialize(obj: Any) -> Any:
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


if __name__ == "__main__":
    main()
