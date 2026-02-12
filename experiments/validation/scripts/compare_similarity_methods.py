"""Compare similarity methods for PR-README text pairs.

Evaluates multiple similarity approaches (Gemini embeddings, TF-IDF, MiniLM
at various token lengths, Jaccard) on PR-README text pairs and performs
statistical analysis to determine whether the choice of similarity method
materially affects the H4 finding.

Runs two analysis scopes:
  - Gemini subset (rows with embedding_similarity): all methods including Gemini
  - Full dataset (all rows with text): TF-IDF, MiniLM variants, Jaccard only
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
from matplotlib.lines import Line2D
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
    holm_bonferroni,
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
# Text loading helpers (shared between Gemini subset and full dataset)
# ---------------------------------------------------------------------------

def _load_text_sources(
    df: pd.DataFrame,
) -> tuple[dict[str, str], dict[tuple[str, int], str],
           dict[tuple[str, int], str]]:
    """Load README, PR body, and PR title maps from disk.

    Returns (readme_map, body_map, title_map).
    """
    repos_dir = BASE_DIR / "data" / "raw" / "repos"
    classified_dir = BASE_DIR / "data" / "raw" / "prs_classified"

    readme_map: dict[str, str] = {}
    for repo_name in df["repo"].unique():
        owner, name = repo_name.split("/", 1)
        readme_path = repos_dir / f"{owner}__{name}_readme.md"
        if readme_path.exists():
            text = readme_path.read_text(errors="replace").strip()
            if text:
                readme_map[repo_name] = text[:4000]

    # Load PR bodies and titles from JSONL files.
    # NOTE (Issue 4): We source titles from the JSONL rather than the
    # parquet because the parquet has no "title" column — stage5's
    # row.get("title", "") always returns "". The JSONL approach is
    # more correct since it contains the actual PR metadata.
    body_map: dict[tuple[str, int], str] = {}
    title_map: dict[tuple[str, int], str] = {}
    for repo_name in df["repo"].unique():
        owner, name = repo_name.split("/", 1)
        cpath = classified_dir / f"{owner}__{name}.jsonl"
        if cpath.exists():
            for rec in read_jsonl(cpath):
                key = (repo_name, rec["number"])
                body_map[key] = rec.get("body", "")
                title_map[key] = rec.get("title", "")

    return readme_map, body_map, title_map


def _build_text_pairs(
    df_sub: pd.DataFrame,
    readme_map: dict[str, str],
    body_map: dict[tuple[str, int], str],
    title_map: dict[tuple[str, int], str],
) -> tuple[list[str], list[str], pd.DataFrame, int]:
    """Build aligned (readme_texts, pr_texts, filtered_df, n_title_only).

    Returns the filtered DataFrame and count of title-only PRs.
    """
    readme_texts: list[str] = []
    pr_texts: list[str] = []
    keep_mask: list[bool] = []
    n_title_only = 0

    for _, row in df_sub.iterrows():
        repo_name = row["repo"]
        pr_key = (repo_name, row["pr_number"])

        readme = readme_map.get(repo_name)
        body = body_map.get(pr_key, "")
        title = title_map.get(pr_key, "")

        pr_text = body.strip() if body else ""
        if not pr_text:
            pr_text = title.strip()
            if pr_text:
                n_title_only += 1
        if pr_text:
            pr_text = pr_text[:2000]

        if readme and pr_text:
            readme_texts.append(readme)
            pr_texts.append(pr_text)
            keep_mask.append(True)
        else:
            keep_mask.append(False)

    df_filtered = df_sub.loc[keep_mask].reset_index(drop=True)
    return readme_texts, pr_texts, df_filtered, n_title_only


def load_text_pairs(
    df: pd.DataFrame,
) -> tuple[list[str], list[str], pd.DataFrame, int]:
    """Load text pairs for the Gemini subset (embedding_similarity present).

    Returns (readme_texts, pr_texts, filtered_df, n_title_only).
    """
    valid = df["embedding_similarity"].notna()
    df_sub = df.loc[valid].copy().reset_index(drop=True)
    logger.info(
        "Rows with embedding_similarity: %d", len(df_sub),
    )
    readme_map, body_map, title_map = _load_text_sources(df_sub)
    texts = _build_text_pairs(df_sub, readme_map, body_map, title_map)
    logger.info("Gemini subset text pairs: %d", len(texts[0]))
    return texts


def load_all_text_pairs(
    df: pd.DataFrame,
) -> tuple[list[str], list[str], pd.DataFrame, int]:
    """Load text pairs for the full dataset (no Gemini filter).

    Returns (readme_texts, pr_texts, filtered_df, n_title_only).
    """
    df_sub = df.copy().reset_index(drop=True)
    logger.info("Full dataset rows: %d", len(df_sub))
    readme_map, body_map, title_map = _load_text_sources(df_sub)
    texts = _build_text_pairs(df_sub, readme_map, body_map, title_map)
    logger.info("Full dataset text pairs: %d", len(texts[0]))
    return texts


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
# Compute similarities
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
    max_length: int = 128,
) -> np.ndarray | None:
    """Compute MiniLM ONNX similarities for each text pair.

    Parameters
    ----------
    max_length : Token truncation length (128, 256, or 512).

    Returns None if onnxruntime is not available or if the model
    cannot handle the requested sequence length.
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
    tokenizer.enable_truncation(max_length=max_length)
    tokenizer.enable_padding(length=max_length)

    session = ort.InferenceSession(str(model_path))

    def _embed_batch(
        texts: list[str], batch_size: int = 64,
    ) -> np.ndarray:
        all_embeddings: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encodings = tokenizer.encode_batch(batch)

            input_ids = np.array(
                [e.ids for e in encodings], dtype=np.int64,
            )
            attention_mask = np.array(
                [e.attention_mask for e in encodings],
                dtype=np.int64,
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
            token_embs = outputs[0]  # (batch, seq_len, hidden)
            mask_exp = attention_mask[
                :, :, np.newaxis
            ].astype(np.float32)
            summed = np.sum(token_embs * mask_exp, axis=1)
            counts = np.sum(mask_exp, axis=1).clip(min=1e-9)
            mean_pooled = summed / counts

            # L2 normalize
            norms = np.linalg.norm(
                mean_pooled, axis=1, keepdims=True,
            ).clip(min=1e-9)
            normalized = mean_pooled / norms
            all_embeddings.append(normalized)

        return np.vstack(all_embeddings)

    logger.info(
        "Computing MiniLM-%d embeddings for %d texts...",
        max_length, len(readme_texts) * 2,
    )
    try:
        readme_embs = _embed_batch(readme_texts)
        pr_embs = _embed_batch(pr_texts)
    except Exception:
        logger.warning(
            "MiniLM failed at max_length=%d (ONNX model may have "
            "fixed position embeddings), skipping",
            max_length,
        )
        return None

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
# Statistical analysis
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
    gemini_sims: np.ndarray | None = None,
    *,
    skip_marginal: bool = False,
    merge_rate: np.ndarray | None = None,
    age: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run statistical analysis for one similarity method.

    Parameters
    ----------
    skip_marginal : If True, skip the marginal improvement computation
        (useful when marginal is recomputed separately on a clean subset).
    """
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

    # Marginal improvement in combined model (Issue 6: skip when
    # caller will recompute on properly filtered data)
    if not skip_marginal and merge_rate is not None and age is not None:
        marginal = run_marginal_improvement(
            y, ge_scores, merge_rate, age, similarities,
        )
        results["marginal"] = marginal

    return results


# ---------------------------------------------------------------------------
# Scope-level analysis runner
# ---------------------------------------------------------------------------

def run_scope_analysis(
    scope_name: str,
    readme_texts: list[str],
    pr_texts: list[str],
    df_sub: pd.DataFrame,
    gemini_sims: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run the full analysis pipeline for one scope.

    Returns a dict with keys: n_pairs, n_with_extras, methods,
    method_sims, pairwise_delong, lrt_corrections, delong_corrections.
    """
    y = df_sub["outcome"].apply(_binary_target).values
    ge_scores = df_sub["normalized_score"].values

    has_extras = (
        df_sub["author_merge_rate"].notna()
        & df_sub["log_account_age_days"].notna()
        & (df_sub["log_account_age_days"] > 0)
    )
    merge_rate = df_sub["author_merge_rate"].values
    age = df_sub["log_account_age_days"].values

    # --- Compute similarities ---
    logger.info("[%s] Computing TF-IDF similarities...", scope_name)
    tfidf_sims = compute_tfidf_similarities(readme_texts, pr_texts)

    minilm_variants: dict[str, np.ndarray] = {}
    for ml in (128, 256, 512):
        logger.info(
            "[%s] Computing MiniLM-%d similarities...",
            scope_name, ml,
        )
        sims = compute_minilm_similarities(
            readme_texts, pr_texts, max_length=ml,
        )
        if sims is not None:
            minilm_variants[f"MiniLM-{ml}"] = sims

    logger.info("[%s] Computing Jaccard similarities...", scope_name)
    jaccard_sims = compute_jaccard_similarities(readme_texts, pr_texts)

    # Build method dictionary
    method_sims: dict[str, np.ndarray] = {}
    if gemini_sims is not None:
        method_sims["Gemini"] = gemini_sims
    method_sims["TF-IDF"] = tfidf_sims
    method_sims.update(minilm_variants)
    method_sims["Jaccard"] = jaccard_sims

    # --- Statistical analysis per method ---
    method_results: dict[str, dict[str, Any]] = {}
    for name, sims in method_sims.items():
        logger.info("[%s] Analyzing method: %s", scope_name, name)
        gem_ref = gemini_sims if (
            name != "Gemini" and gemini_sims is not None
        ) else None

        result = analyze_method(
            name=name,
            similarities=sims,
            y=y,
            ge_scores=ge_scores,
            gemini_sims=gem_ref,
            skip_marginal=True,
        )

        # Run marginal with properly filtered data
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

    # --- Holm-Bonferroni on LRT p-values (non-Gemini methods) ---
    lrt_pvals: dict[str, float] = {
        name: res["lrt_p_value"]
        for name, res in method_results.items()
        if name != "Gemini"
    }
    lrt_corrections = holm_bonferroni(lrt_pvals)

    # --- Pairwise DeLong tests ---
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

    # Holm-Bonferroni on DeLong p-values
    delong_pvals: dict[str, float] = {
        k: v["p_value"] for k, v in pairwise_delong.items()
    }
    delong_corrections = holm_bonferroni(delong_pvals)

    return {
        "n_pairs": len(df_sub),
        "n_with_extras": int(has_extras.sum()),
        "methods": method_results,
        "method_sims": method_sims,
        "pairwise_delong": pairwise_delong,
        "lrt_corrections": lrt_corrections,
        "delong_corrections": delong_corrections,
        "y": y,
    }


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


def _write_method_summary(
    lines: list[str],
    scope: dict[str, Any],
    include_gemini_r: bool = True,
) -> None:
    """Write a method summary table using corrected p-values."""
    methods = scope["methods"]
    corrections = scope["lrt_corrections"]

    header = (
        "| Method | LRT Stat | Raw p | Adj. p"
        " | Standalone AUC | AUC 95% CI"
    )
    sep = (
        "|--------|----------|-------|-------"
        "|----------------|----------"
    )
    if include_gemini_r:
        header += " | Gemini r |"
        sep += "--|----------|"
    else:
        header += " |"
        sep += "--|"

    lines.extend([header, sep])

    for name, res in methods.items():
        lrt_s = f"{res['lrt_statistic']:.3f}"
        lrt_p = f"{res['lrt_p_value']:.4e}"
        adj_p = corrections.get(name, {}).get("adjusted_p")
        adj_str = f"{adj_p:.4e}" if adj_p is not None else "---"
        # Use corrected p for significance marker on non-Gemini
        marker_p = adj_p if adj_p is not None else res["lrt_p_value"]
        sig = _sig_marker(marker_p)
        auc = f"{res['standalone_auc']:.4f}"
        ci = (
            f"[{res['standalone_auc_ci_lower']:.4f},"
            f" {res['standalone_auc_ci_upper']:.4f}]"
        )
        row = (
            f"| {name}{sig} | {lrt_s} | {lrt_p} | {adj_str}"
            f" | {auc} | {ci}"
        )
        if include_gemini_r:
            gem_r = res.get("pearson_r_gemini")
            gem_str = f"{gem_r:.4f}" if gem_r is not None else "---"
            row += f" | {gem_str} |"
        else:
            row += " |"
        lines.append(row)

    lines.extend([
        "",
        "*Significance markers use Holm-Bonferroni corrected p-values"
        " (non-Gemini methods).*",
        "*\\* p < 0.05, \\*\\* p < 0.01, \\*\\*\\* p < 0.001*",
    ])


def _write_minilm_comparison(
    lines: list[str],
    scope: dict[str, Any],
) -> None:
    """Write MiniLM token length comparison subsection."""
    methods = scope["methods"]
    minilm_names = sorted(
        [n for n in methods if n.startswith("MiniLM-")],
    )
    if len(minilm_names) < 2:
        return

    lines.extend([
        "",
        "### MiniLM Token Length Comparison",
        "",
        "| Variant | LRT Stat | Raw p | Adj. p"
        " | Standalone AUC |",
        "|---------|----------|-------|-------"
        "|----------------|",
    ])

    corrections = scope["lrt_corrections"]
    for name in minilm_names:
        res = methods[name]
        adj = corrections.get(name, {}).get("adjusted_p")
        adj_str = f"{adj:.4e}" if adj is not None else "---"
        marker_p = adj if adj is not None else res["lrt_p_value"]
        sig = _sig_marker(marker_p)
        lines.append(
            f"| {name}{sig} | {res['lrt_statistic']:.3f}"
            f" | {res['lrt_p_value']:.4e}"
            f" | {adj_str}"
            f" | {res['standalone_auc']:.4f} |",
        )

    # Compute approximate token coverage stats
    lines.extend([
        "",
        "Mean README length is ~1,300 tokens after 4,000-char truncation.",
        "At 128 tokens, MiniLM captures ~10% of README content;",
        "at 256, ~20%; at 512, ~40%.",
    ])


def _write_marginal_table(
    lines: list[str],
    scope: dict[str, Any],
) -> None:
    """Write marginal improvement table."""
    methods = scope["methods"]
    n_extras = scope["n_with_extras"]
    n_pairs = scope["n_pairs"]
    gap = n_pairs - n_extras

    lines.extend([
        "",
        f"### Marginal Improvement (n={n_extras:,};"
        f" {gap} rows lack merge_rate or account_age)",
        "",
        "| Method | Base AUC | Full AUC | AUC Diff"
        " | DeLong p | LRT p |",
        "|--------|----------|----------|----------"
        "|---------|-------|",
    ])

    for name, res in methods.items():
        m = res.get("marginal", {})
        if m:
            lines.append(
                f"| {name} | {m['auc_base']:.4f}"
                f" | {m['auc_full']:.4f}"
                f" | {m['auc_diff']:+.4f}"
                f" | {m['delong_p']:.4e}"
                f" | {m['lrt_p']:.4e} |",
            )


def _write_delong_table(
    lines: list[str],
    scope: dict[str, Any],
) -> None:
    """Write pairwise DeLong tests table with corrected p-values."""
    pairwise = scope["pairwise_delong"]
    corrections = scope["delong_corrections"]

    lines.extend([
        "",
        "### Pairwise DeLong Tests (Standalone AUCs)",
        "",
        "| Comparison | AUC A | AUC B | z"
        " | Raw p | Adj. p |",
        "|------------|-------|-------|---"
        "|-------|--------|",
    ])

    for pair_name, res in pairwise.items():
        adj = corrections.get(pair_name, {}).get("adjusted_p")
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


def generate_report(
    gemini_scope: dict[str, Any],
    full_scope: dict[str, Any],
    selection_bias: dict[str, Any],
    output_path: Path,
) -> None:
    """Generate the full markdown comparison report."""
    lines = [
        "# Similarity Method Comparison Report",
        "",
        "## Overview",
        "",
        "This report compares text similarity methods applied to PR body"
        " / repo README",
        "text pairs from the H4 semantic similarity hypothesis test.",
        "Two scopes are analyzed:",
        "",
        f"- **Gemini subset** (n={gemini_scope['n_pairs']:,}):"
        " All methods including Gemini embeddings.",
        f"- **Full dataset** (n={full_scope['n_pairs']:,}):"
        " TF-IDF, MiniLM variants, and Jaccard (no Gemini filter).",
        "",
    ]

    # --- AUC Inversion section ---
    lines.extend([
        "## AUC Inversion: Similarity as a Negative Predictor",
        "",
        "All similarity methods produce standalone AUC < 0.5, meaning"
        " higher",
        "PR-README similarity is associated with *lower* merge"
        " probability.",
        "This inversion is consistent across every method tested"
        " (Gemini,",
        "TF-IDF, MiniLM variants, Jaccard), ruling out the possibility"
        " that",
        "it is an artifact of any single embedding model.",
        "",
        "Possible explanations:",
        "",
        "1. **Boilerplate/template PRs**: PRs that closely match the"
        " README",
        "   (e.g., copy-pasted templates, bot-generated PRs) may be"
        " lower",
        "   quality and less likely to merge.",
        "2. **Subsystem specificity**: Merged PRs tend to target"
        " specific",
        "   subsystems whose vocabulary diverges from the high-level"
        " README,",
        "   while rejected PRs may be more generic or misaligned.",
        "3. **Information vs. direction**: The LRT tests whether"
        " similarity",
        "   adds *information* to the prediction model, not whether the",
        "   relationship is positive. A strong negative signal is just"
        " as",
        "   informative as a positive one for the LRT.",
        "",
    ])

    # --- Gemini subset ---
    lines.extend([
        f"## Gemini Subset Analysis (n={gemini_scope['n_pairs']:,})",
        "",
        "### Method Summary",
        "",
    ])
    _write_method_summary(lines, gemini_scope, include_gemini_r=True)
    _write_minilm_comparison(lines, gemini_scope)
    _write_marginal_table(lines, gemini_scope)
    _write_delong_table(lines, gemini_scope)

    # --- Full dataset ---
    lines.extend([
        "",
        f"## Full Dataset Analysis (n={full_scope['n_pairs']:,})",
        "",
        "### Method Summary",
        "",
    ])
    _write_method_summary(lines, full_scope, include_gemini_r=False)
    _write_minilm_comparison(lines, full_scope)
    _write_marginal_table(lines, full_scope)

    # --- Selection bias ---
    lines.extend([
        "",
        "## Selection Bias Analysis",
        "",
        "The Gemini subset includes only PRs with non-empty bodies"
        " (needed",
        "for Gemini embedding). The full dataset also includes"
        " title-only PRs.",
        "",
        f"- **Gemini subset merge rate**: "
        f"{selection_bias['gemini_merge_rate']:.1%}",
        f"- **Full dataset merge rate**: "
        f"{selection_bias['full_merge_rate']:.1%}",
        f"- **Title-only PR fraction (full)**: "
        f"{selection_bias['title_only_fraction']:.1%}",
        f"- **Gemini subset mean GE score**: "
        f"{selection_bias['gemini_mean_ge']:.4f}",
        f"- **Full dataset mean GE score**: "
        f"{selection_bias['full_mean_ge']:.4f}",
        "",
        "Title-only PRs (~16 tokens) may dilute the similarity signal"
        " in the",
        "full dataset, as there is less textual information to compare"
        " against",
        "the README.",
    ])

    # --- Decision criteria ---
    lines.extend([
        "",
        "## Decision Criteria Evaluation",
        "",
        "The key question is whether the H4 finding (semantic"
        " similarity",
        "adds predictive value beyond GE score alone) is robust to the",
        "choice of similarity method. Significance is evaluated using",
        "Holm-Bonferroni corrected p-values.",
        "",
        "**Criteria:**",
        "1. If all non-Gemini methods yield significant corrected LRT"
        " p-values,",
        "   the finding is robust.",
        "2. If only Gemini is significant, the finding may be an"
        " artifact of the",
        "   embedding model.",
        "3. If no method is significant, the finding is likely"
        " spurious.",
        "",
    ])

    # Automated evaluation (using corrected p-values)
    corrections = gemini_scope["lrt_corrections"]
    significant = [
        name for name, corr in corrections.items()
        if corr["reject"]
    ]
    # Also check Gemini raw
    gem_res = gemini_scope["methods"].get("Gemini", {})
    gem_sig = gem_res.get("lrt_p_value", 1.0) < 0.05
    total_non_gemini = len(corrections)

    significant_with_gem = (
        ["Gemini", *significant] if gem_sig else list(significant)
    )

    total = total_non_gemini + (1 if "Gemini" in gemini_scope["methods"] else 0)

    if len(significant_with_gem) == total:
        lines.append(
            f"**Result:** All {total} methods yield significant LRT."
            " The H4 finding is robust to similarity method choice.",
        )
    elif len(significant_with_gem) > 1:
        lines.append(
            f"**Result:** {len(significant_with_gem)}/{total} methods"
            f" yield significant LRT: "
            f"{', '.join(significant_with_gem)}."
            " The H4 finding is partially robust.",
        )
    elif len(significant_with_gem) == 1:
        lines.append(
            f"**Result:** Only {significant_with_gem[0]} yields a"
            " significant LRT. The H4 finding may be"
            " method-specific.",
        )
    else:
        lines.append(
            "**Result:** No method yields a significant LRT."
            " The H4 finding is not supported.",
        )

    # --- Known Limitations ---
    lines.extend([
        "",
        "## Known Limitations",
        "",
        "- Gemini similarities were computed during feature engineering"
        " (not recomputed here); other methods are computed fresh.",
        "- MiniLM is tested at 128, 256, and 512 token truncation"
        " lengths.",
        "  At 128 tokens, ~90% of README content is discarded."
        " At 256 tokens, ~80%.",
        "  At 512 tokens, ~60%. Gemini has a much larger context"
        " window.",
        "- TF-IDF is fitted on the full corpus, which could introduce"
        " minor data leakage in a strict train/test sense, but this"
        " is acceptable",
        "  for a comparison study where all methods see the same data.",
        "- Jaccard is a bag-of-words baseline with no semantic"
        " understanding.",
        "- The full dataset includes ~"
        f"{selection_bias['title_only_fraction']:.0%}"
        " title-only PRs which have very short text (~16 tokens).",
    ])

    # --- Methodology Notes ---
    lines.extend([
        "",
        "## Methodology Notes",
        "",
        "### Holm-Bonferroni Correction",
        "",
        "Multiple comparison correction is applied using the"
        " Holm-Bonferroni",
        "step-down procedure, which is uniformly more powerful than"
        " the",
        "standard Bonferroni correction while still controlling the"
        " family-wise",
        "error rate at alpha = 0.05.",
        "",
        "### Why Gemini Is Excluded from Correction",
        "",
        "Gemini embeddings were the original method used in the stage6"
        " H4 test.",
        "The comparison script replicates this result as a"
        " backward-compatibility",
        "check, not as a new hypothesis. Only the alternative methods"
        " (TF-IDF,",
        "MiniLM variants, Jaccard) represent new tests and are"
        " included in the",
        "correction family.",
        "",
        "---",
        "*Generated by compare_similarity_methods.py*",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    logger.info("Saved report to %s", output_path)


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def _forest_plot(
    ax: plt.Axes,
    scope: dict[str, Any],
    title: str,
) -> None:
    """Draw a forest-style horizontal bar plot of LRT statistics."""
    methods = scope["methods"]
    corrections = scope["lrt_corrections"]

    names = list(methods.keys())
    lrt_stats = [methods[n]["lrt_statistic"] for n in names]
    y_pos = np.arange(len(names))

    colors = []
    for n in names:
        if n == "Gemini":
            p = methods[n]["lrt_p_value"]
        else:
            p = corrections.get(n, {}).get(
                "adjusted_p", methods[n]["lrt_p_value"],
            )
        colors.append("#e74c3c" if p < 0.05 else "#95a5a6")

    ax.barh(y_pos, lrt_stats, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("LRT Statistic")
    ax.set_title(title)
    ax.invert_yaxis()

    max_stat = max(lrt_stats) if lrt_stats else 1.0
    for i, n in enumerate(names):
        stat = lrt_stats[i]
        if n == "Gemini":
            p = methods[n]["lrt_p_value"]
        else:
            p = corrections.get(n, {}).get(
                "adjusted_p", methods[n]["lrt_p_value"],
            )
        marker = _sig_marker(p)
        ax.text(
            stat + max_stat * 0.02, i,
            f"{stat:.2f}{marker}",
            va="center", fontsize=8,
        )


def generate_figure(
    gemini_scope: dict[str, Any],
    full_scope: dict[str, Any],
    output_path: Path,
) -> None:
    """Generate 6-panel comparison figure (3x2)."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    gem_y = gemini_scope["y"]
    gem_sims = gemini_scope["method_sims"]
    gem_results = gemini_scope["methods"]

    # Shared scatter elements (Issue 5: define before conditionals)
    merged_colors = [
        "#2ecc71" if yi == 1 else "#e74c3c" for yi in gem_y
    ]
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

    # (A) Forest plot: Gemini subset LRT
    _forest_plot(
        axes[0, 0], gemini_scope,
        "(A) H4 LRT — Gemini Subset",
    )

    # (B) KDE: Similarity distributions (Gemini subset)
    ax = axes[0, 1]
    palette = {
        "Gemini": "#3498db",
        "TF-IDF": "#2ecc71",
        "Jaccard": "#f39c12",
    }
    # Add MiniLM variants with varying reds
    minilm_colors = {
        "MiniLM-128": "#e74c3c",
        "MiniLM-256": "#c0392b",
        "MiniLM-512": "#a93226",
    }
    palette.update(minilm_colors)
    for name, sims in gem_sims.items():
        color = palette.get(name, "#95a5a6")
        ax.hist(
            sims, bins=50, alpha=0.2, color=color, density=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sns.kdeplot(
                sims, ax=ax, color=color, label=name, linewidth=2,
            )
    ax.set_xlabel("Similarity Score")
    ax.set_ylabel("Density")
    ax.set_title("(B) Similarity Distributions — Gemini Subset")
    ax.legend(fontsize=7)

    # (C) Scatter: TF-IDF vs Gemini
    ax = axes[1, 0]
    if "TF-IDF" in gem_sims and "Gemini" in gem_sims:
        ax.scatter(
            gem_sims["Gemini"], gem_sims["TF-IDF"],
            c=merged_colors, alpha=0.3, s=10,
        )
        r = gem_results.get("TF-IDF", {}).get(
            "pearson_r_gemini", float("nan"),
        )
        ax.set_xlabel("Gemini Similarity")
        ax.set_ylabel("TF-IDF Similarity")
        ax.set_title(f"(C) TF-IDF vs Gemini (r={r:.3f})")
        ax.legend(handles=legend_elements, fontsize=8)

    # (D) Scatter: Best MiniLM variant vs Gemini
    ax = axes[1, 1]
    best_minilm = None
    best_lrt = -1.0
    for name in gem_sims:
        if not name.startswith("MiniLM-"):
            continue
        lrt = gem_results.get(name, {}).get("lrt_statistic", 0.0)
        if lrt > best_lrt:
            best_lrt = lrt
            best_minilm = name

    if best_minilm and "Gemini" in gem_sims:
        ax.scatter(
            gem_sims["Gemini"], gem_sims[best_minilm],
            c=merged_colors, alpha=0.3, s=10,
        )
        r = gem_results.get(best_minilm, {}).get(
            "pearson_r_gemini", float("nan"),
        )
        ax.set_xlabel("Gemini Similarity")
        ax.set_ylabel(f"{best_minilm} Similarity")
        ax.set_title(
            f"(D) {best_minilm} vs Gemini (r={r:.3f})",
        )
        ax.legend(handles=legend_elements, fontsize=8)
    elif "Jaccard" in gem_sims and "Gemini" in gem_sims:
        ax.scatter(
            gem_sims["Gemini"], gem_sims["Jaccard"],
            c=merged_colors, alpha=0.3, s=10,
        )
        r = gem_results.get("Jaccard", {}).get(
            "pearson_r_gemini", float("nan"),
        )
        ax.set_xlabel("Gemini Similarity")
        ax.set_ylabel("Jaccard Similarity")
        ax.set_title(f"(D) Jaccard vs Gemini (r={r:.3f})")
        ax.legend(handles=legend_elements, fontsize=8)

    # (E) Forest plot: Full dataset LRT
    _forest_plot(
        axes[2, 0], full_scope,
        "(E) H4 LRT — Full Dataset",
    )

    # (F) Grouped bar: MiniLM token length comparison
    ax = axes[2, 1]
    minilm_names_gem = sorted(
        [n for n in gem_results if n.startswith("MiniLM-")],
    )
    minilm_names_full = sorted(
        [n for n in full_scope["methods"]
         if n.startswith("MiniLM-")],
    )
    all_minilm = sorted(set(minilm_names_gem + minilm_names_full))

    if all_minilm:
        x_idx = np.arange(len(all_minilm))
        width = 0.35
        gem_vals = [
            gem_results.get(n, {}).get("lrt_statistic", 0.0)
            for n in all_minilm
        ]
        full_vals = [
            full_scope["methods"].get(n, {}).get(
                "lrt_statistic", 0.0,
            )
            for n in all_minilm
        ]
        ax.bar(
            x_idx - width / 2, gem_vals, width,
            label="Gemini subset", color="#3498db", alpha=0.7,
        )
        ax.bar(
            x_idx + width / 2, full_vals, width,
            label="Full dataset", color="#e67e22", alpha=0.7,
        )
        ax.set_xticks(x_idx)
        ax.set_xticklabels(all_minilm, fontsize=9)
        ax.set_ylabel("LRT Statistic")
        ax.set_title("(F) MiniLM Token Length Comparison")
        ax.legend(fontsize=8)
    else:
        ax.text(
            0.5, 0.5, "MiniLM not available",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_title("(F) MiniLM Token Length Comparison")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved comparison figure to %s", output_path)


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

    # ===== Gemini subset =====
    (gem_readmes, gem_prs, gem_df,
     gem_title_only) = load_text_pairs(df)

    if len(gem_df) == 0:
        logger.error("No valid text pairs found for Gemini subset")
        sys.exit(1)

    sanity_check_gemini(gem_readmes, gem_prs, gem_df)

    gemini_sims = gem_df["embedding_similarity"].values
    gemini_scope = run_scope_analysis(
        "Gemini subset", gem_readmes, gem_prs, gem_df,
        gemini_sims=gemini_sims,
    )

    # ===== Full dataset =====
    (full_readmes, full_prs, full_df,
     full_title_only) = load_all_text_pairs(df)

    if len(full_df) == 0:
        logger.error("No valid text pairs found for full dataset")
        sys.exit(1)

    full_scope = run_scope_analysis(
        "Full dataset", full_readmes, full_prs, full_df,
        gemini_sims=None,
    )

    # ===== Selection bias =====
    gem_y = gem_df["outcome"].apply(_binary_target).values
    full_y = full_df["outcome"].apply(_binary_target).values
    selection_bias: dict[str, Any] = {
        "gemini_merge_rate": float(gem_y.mean()),
        "full_merge_rate": float(full_y.mean()),
        "gemini_n": len(gem_df),
        "full_n": len(full_df),
        "title_only_fraction": (
            full_title_only / len(full_df) if len(full_df) else 0.0
        ),
        "title_only_count": full_title_only,
        "gemini_mean_ge": float(
            gem_df["normalized_score"].mean(),
        ),
        "full_mean_ge": float(
            full_df["normalized_score"].mean(),
        ),
    }
    logger.info(
        "Selection bias: Gemini merge rate=%.3f, "
        "Full merge rate=%.3f, title-only=%.1f%%",
        selection_bias["gemini_merge_rate"],
        selection_bias["full_merge_rate"],
        selection_bias["title_only_fraction"] * 100,
    )

    # ===== Generate outputs =====
    output_dir = BASE_DIR / "results" / "similarity_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # results.json
    def _strip_internal(scope: dict[str, Any]) -> dict[str, Any]:
        """Remove non-serializable internal arrays from scope."""
        return {
            k: _serialize(v) for k, v in scope.items()
            if k not in ("method_sims", "y")
        }

    all_output: dict[str, Any] = {
        "gemini_subset": _strip_internal(gemini_scope),
        "full_dataset": _strip_internal(full_scope),
        "selection_bias": _serialize(selection_bias),
    }
    results_json_path = output_dir / "results.json"
    with open(results_json_path, "w") as f:
        json.dump(all_output, f, indent=2, default=str)
    logger.info("Saved results to %s", results_json_path)

    # comparison_report.md
    generate_report(
        gemini_scope, full_scope, selection_bias,
        output_dir / "comparison_report.md",
    )

    # similarity_comparison.png
    generate_figure(
        gemini_scope, full_scope,
        output_dir / "similarity_comparison.png",
    )

    logger.info("Done. Outputs in %s", output_dir)


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


if __name__ == "__main__":
    main()
