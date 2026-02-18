# Similarity Method Comparison Report

## Overview

This report compares text similarity methods applied to PR body / repo README
text pairs from the H4 semantic similarity hypothesis test.
Two scopes are analyzed:

- **Gemini subset** (n=1,569): All methods including Gemini embeddings.
- **Full dataset** (n=5,417): TF-IDF, MiniLM variants, and Jaccard (no Gemini filter).

## AUC Inversion: Similarity as a Negative Predictor

All similarity methods produce standalone AUC < 0.5, meaning higher
PR-README similarity is associated with *lower* merge probability.
This inversion is consistent across every method tested (Gemini,
TF-IDF, MiniLM variants, Jaccard), ruling out the possibility that
it is an artifact of any single embedding model.

Possible explanations:

1. **Boilerplate/template PRs**: PRs that closely match the README
   (e.g., copy-pasted templates, bot-generated PRs) may be lower
   quality and less likely to merge.
2. **Subsystem specificity**: Merged PRs tend to target specific
   subsystems whose vocabulary diverges from the high-level README,
   while rejected PRs may be more generic or misaligned.
3. **Information vs. direction**: The LRT tests whether similarity
   adds *information* to the prediction model, not whether the
   relationship is positive. A strong negative signal is just as
   informative as a positive one for the LRT.

## Gemini Subset Analysis (n=1,569)

### Method Summary

| Method | LRT Stat | Raw p | Adj. p | Standalone AUC | AUC 95% CI | Gemini r |
|--------|----------|-------|-------|----------------|------------|----------|
| Gemini *** | 35.200 | 2.9745e-09 | --- | 0.4110 | [0.3824, 0.4395] | --- |
| TF-IDF | 1.774 | 1.8289e-01 | 5.4866e-01 | 0.4499 | [0.4212, 0.4786] | 0.5708 |
| MiniLM-128 | 0.002 | 9.6445e-01 | 1.0000e+00 | 0.4949 | [0.4651, 0.5246] | 0.4385 |
| MiniLM-256 | 0.056 | 8.1247e-01 | 1.0000e+00 | 0.4834 | [0.4539, 0.5129] | 0.5251 |
| MiniLM-512 | 2.299 | 1.2948e-01 | 5.1793e-01 | 0.5238 | [0.4950, 0.5526] | 0.6103 |
| Jaccard *** | 25.498 | 4.4281e-07 | 2.2141e-06 | 0.4241 | [0.3954, 0.4527] | 0.5054 |

*Significance markers use Holm-Bonferroni corrected p-values (non-Gemini methods).*
*\* p < 0.05, \*\* p < 0.01, \*\*\* p < 0.001*

### MiniLM Token Length Comparison

| Variant | LRT Stat | Raw p | Adj. p | Standalone AUC |
|---------|----------|-------|-------|----------------|
| MiniLM-128 | 0.002 | 9.6445e-01 | 1.0000e+00 | 0.4949 |
| MiniLM-256 | 0.056 | 8.1247e-01 | 1.0000e+00 | 0.4834 |
| MiniLM-512 | 2.299 | 1.2948e-01 | 5.1793e-01 | 0.5238 |

Mean README length is ~1,300 tokens after 4,000-char truncation.
At 128 tokens, MiniLM captures ~10% of README content;
at 256, ~20%; at 512, ~40%.

### Marginal Improvement (n=1,495; 74 rows lack merge_rate or account_age)

| Method | Base AUC | Full AUC | AUC Diff | DeLong p | LRT p |
|--------|----------|----------|----------|---------|-------|
| Gemini | 0.6272 | 0.6539 | +0.0266 | 3.1018e-02 | 3.9133e-09 |
| TF-IDF | 0.6272 | 0.6311 | +0.0039 | 5.5072e-01 | 6.8803e-02 |
| MiniLM-128 | 0.6272 | 0.6268 | -0.0004 | 7.1910e-01 | 8.0230e-01 |
| MiniLM-256 | 0.6272 | 0.6261 | -0.0011 | 6.4930e-01 | 5.8457e-01 |
| MiniLM-512 | 0.6272 | 0.6272 | -0.0000 | 9.9749e-01 | 2.1478e-01 |
| Jaccard | 0.6272 | 0.6437 | +0.0165 | 1.7806e-01 | 4.7295e-07 |

### Pairwise DeLong Tests (Standalone AUCs)

| Comparison | AUC A | AUC B | z | Raw p | Adj. p |
|------------|-------|-------|---|-------|--------|
| Gemini vs TF-IDF ** | 0.4110 | 0.4499 | -3.235 | 1.2158e-03 | 8.8163e-03 |
| Gemini vs MiniLM-128 *** | 0.4110 | 0.4949 | -5.308 | 1.1105e-07 | 1.3326e-06 |
| Gemini vs MiniLM-256 *** | 0.4110 | 0.4834 | -5.004 | 5.6160e-07 | 6.1776e-06 |
| Gemini vs MiniLM-512 *** | 0.4110 | 0.5238 | -8.611 | 7.2237e-18 | 1.0836e-16 |
| Gemini vs Jaccard | 0.4110 | 0.4241 | -0.883 | 3.7724e-01 | 3.7724e-01 |
| TF-IDF vs MiniLM-128 * | 0.4499 | 0.4949 | -2.953 | 3.1450e-03 | 1.8870e-02 |
| TF-IDF vs MiniLM-256 | 0.4499 | 0.4834 | -2.419 | 1.5561e-02 | 7.7804e-02 |
| TF-IDF vs MiniLM-512 *** | 0.4499 | 0.5238 | -5.699 | 1.2069e-08 | 1.5690e-07 |
| TF-IDF vs Jaccard | 0.4499 | 0.4241 | 1.552 | 1.2077e-01 | 3.4783e-01 |
| MiniLM-128 vs MiniLM-256 | 0.4949 | 0.4834 | 1.572 | 1.1594e-01 | 3.4783e-01 |
| MiniLM-128 vs MiniLM-512 | 0.4949 | 0.5238 | -2.204 | 2.7528e-02 | 1.1011e-01 |
| MiniLM-128 vs Jaccard ** | 0.4949 | 0.4241 | 3.693 | 2.2161e-04 | 1.9945e-03 |
| MiniLM-256 vs MiniLM-512 ** | 0.4834 | 0.5238 | -3.750 | 1.7662e-04 | 1.7662e-03 |
| MiniLM-256 vs Jaccard ** | 0.4834 | 0.4241 | 3.263 | 1.1020e-03 | 8.8163e-03 |
| MiniLM-512 vs Jaccard *** | 0.5238 | 0.4241 | 6.234 | 4.5356e-10 | 6.3499e-09 |

## Full Dataset Analysis (n=5,417)

### Method Summary

| Method | LRT Stat | Raw p | Adj. p | Standalone AUC | AUC 95% CI |
|--------|----------|-------|-------|----------------|------------|
| TF-IDF *** | 85.914 | 1.8792e-20 | 7.5169e-20 | 0.4373 | [0.4203, 0.4542] |
| MiniLM-128 *** | 59.728 | 1.0894e-14 | 1.0894e-14 | 0.4486 | [0.4312, 0.4660] |
| MiniLM-256 *** | 61.342 | 4.7983e-15 | 9.5966e-15 | 0.4416 | [0.4243, 0.4588] |
| MiniLM-512 *** | 68.298 | 1.4056e-16 | 4.2168e-16 | 0.4400 | [0.4230, 0.4570] |
| Jaccard *** | 204.830 | 1.8449e-46 | 9.2246e-46 | 0.4138 | [0.3967, 0.4309] |

*Significance markers use Holm-Bonferroni corrected p-values (non-Gemini methods).*
*\* p < 0.05, \*\* p < 0.01, \*\*\* p < 0.001*

### MiniLM Token Length Comparison

| Variant | LRT Stat | Raw p | Adj. p | Standalone AUC |
|---------|----------|-------|-------|----------------|
| MiniLM-128 *** | 59.728 | 1.0894e-14 | 1.0894e-14 | 0.4486 |
| MiniLM-256 *** | 61.342 | 4.7983e-15 | 9.5966e-15 | 0.4416 |
| MiniLM-512 *** | 68.298 | 1.4056e-16 | 4.2168e-16 | 0.4400 |

Mean README length is ~1,300 tokens after 4,000-char truncation.
At 128 tokens, MiniLM captures ~10% of README content;
at 256, ~20%; at 512, ~40%.

### Marginal Improvement (n=5,129; 288 rows lack merge_rate or account_age)

| Method | Base AUC | Full AUC | AUC Diff | DeLong p | LRT p |
|--------|----------|----------|----------|---------|-------|
| TF-IDF | 0.6472 | 0.6903 | +0.0431 | 3.2863e-20 | 1.0882e-20 |
| MiniLM-128 | 0.6472 | 0.6778 | +0.0306 | 1.3312e-12 | 6.7623e-14 |
| MiniLM-256 | 0.6472 | 0.6807 | +0.0335 | 2.3947e-14 | 6.1824e-15 |
| MiniLM-512 | 0.6472 | 0.6816 | +0.0344 | 3.6059e-15 | 1.3785e-15 |
| Jaccard | 0.6472 | 0.7081 | +0.0609 | 8.6667e-26 | 4.4033e-42 |

## Selection Bias Analysis

The Gemini subset includes only PRs with non-empty bodies (needed
for Gemini embedding). The full dataset also includes title-only PRs.

- **Gemini subset merge rate**: 62.4%
- **Full dataset merge rate**: 71.5%
- **Title-only PR fraction (full)**: 71.0%
- **Gemini subset mean GE score**: 0.7147
- **Full dataset mean GE score**: 0.6033

Title-only PRs (~16 tokens) may dilute the similarity signal in the
full dataset, as there is less textual information to compare against
the README.

## Decision Criteria Evaluation

The key question is whether the H4 finding (semantic similarity
adds predictive value beyond GE score alone) is robust to the
choice of similarity method. Significance is evaluated using
Holm-Bonferroni corrected p-values.

**Criteria:**
1. If all non-Gemini methods yield significant corrected LRT p-values,
   the finding is robust.
2. If only Gemini is significant, the finding may be an artifact of the
   embedding model.
3. If no method is significant, the finding is likely spurious.

**Result:** 2/6 methods yield significant LRT: Gemini, Jaccard. The H4 finding is partially robust.

## Known Limitations

- Gemini similarities were computed during feature engineering (not recomputed here); other methods are computed fresh.
- MiniLM is tested at 128, 256, and 512 token truncation lengths.
  At 128 tokens, ~90% of README content is discarded. At 256 tokens, ~80%.
  At 512 tokens, ~60%. Gemini has a much larger context window.
- TF-IDF is fitted on the full corpus, which could introduce minor data leakage in a strict train/test sense, but this is acceptable
  for a comparison study where all methods see the same data.
- Jaccard is a bag-of-words baseline with no semantic understanding.
- The full dataset includes ~71% title-only PRs which have very short text (~16 tokens).

## Methodology Notes

### Holm-Bonferroni Correction

Multiple comparison correction is applied using the Holm-Bonferroni
step-down procedure, which is uniformly more powerful than the
standard Bonferroni correction while still controlling the family-wise
error rate at alpha = 0.05.

### Why Gemini Is Excluded from Correction

Gemini embeddings were the original method used in the stage6 H4 test.
The comparison script replicates this result as a backward-compatibility
check, not as a new hypothesis. Only the alternative methods (TF-IDF,
MiniLM variants, Jaccard) represent new tests and are included in the
correction family.

---
*Generated by compare_similarity_methods.py*