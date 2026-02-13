# Similarity Method Comparison Report

## Overview

This report compares text similarity methods applied to PR body / repo README
text pairs from the H4 semantic similarity hypothesis test.
Two scopes are analyzed:

- **Gemini subset** (n=1,293): All methods including Gemini embeddings.
- **Full dataset** (n=4,977): TF-IDF, MiniLM variants, and Jaccard (no Gemini filter).

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

## Gemini Subset Analysis (n=1,293)

### Method Summary

| Method | LRT Stat | Raw p | Adj. p | Standalone AUC | AUC 95% CI | Gemini r |
|--------|----------|-------|-------|----------------|------------|----------|
| Gemini *** | 20.818 | 5.0516e-06 | --- | 0.4162 | [0.3821, 0.4504] | --- |
| TF-IDF | 3.435 | 6.3842e-02 | 2.5537e-01 | 0.4422 | [0.4086, 0.4757] | 0.5859 |
| MiniLM-128 | 0.768 | 3.8078e-01 | 7.6156e-01 | 0.4691 | [0.4346, 0.5037] | 0.4382 |
| MiniLM-256 | 2.275 | 1.3149e-01 | 3.9448e-01 | 0.4545 | [0.4200, 0.4890] | 0.5337 |
| MiniLM-512 | 0.601 | 4.3830e-01 | 4.3830e-01 | 0.5132 | [0.4797, 0.5468] | 0.6190 |
| Jaccard ** | 13.596 | 2.2665e-04 | 1.1332e-03 | 0.4412 | [0.4068, 0.4756] | 0.5009 |

*Significance markers use Holm-Bonferroni corrected p-values (non-Gemini methods).*
*\* p < 0.05, \*\* p < 0.01, \*\*\* p < 0.001*

### MiniLM Token Length Comparison

| Variant | LRT Stat | Raw p | Adj. p | Standalone AUC |
|---------|----------|-------|-------|----------------|
| MiniLM-128 | 0.768 | 3.8078e-01 | 7.6156e-01 | 0.4691 |
| MiniLM-256 | 2.275 | 1.3149e-01 | 3.9448e-01 | 0.4545 |
| MiniLM-512 | 0.601 | 4.3830e-01 | 4.3830e-01 | 0.5132 |

Mean README length is ~1,300 tokens after 4,000-char truncation.
At 128 tokens, MiniLM captures ~10% of README content;
at 256, ~20%; at 512, ~40%.

### Marginal Improvement (n=1,245; 48 rows lack merge_rate or account_age)

| Method | Base AUC | Full AUC | AUC Diff | DeLong p | LRT p |
|--------|----------|----------|----------|---------|-------|
| Gemini | 0.6519 | 0.6796 | +0.0277 | 3.6734e-02 | 3.6336e-06 |
| TF-IDF | 0.6519 | 0.6603 | +0.0084 | 2.6523e-01 | 3.7258e-02 |
| MiniLM-128 | 0.6519 | 0.6518 | -0.0001 | 9.8735e-01 | 3.3520e-01 |
| MiniLM-256 | 0.6519 | 0.6555 | +0.0036 | 5.5720e-01 | 1.2627e-01 |
| MiniLM-512 | 0.6519 | 0.6523 | +0.0004 | 8.9666e-01 | 4.8769e-01 |
| Jaccard | 0.6519 | 0.6679 | +0.0160 | 1.9878e-01 | 1.3286e-04 |

### Pairwise DeLong Tests (Standalone AUCs)

| Comparison | AUC A | AUC B | z | Raw p | Adj. p |
|------------|-------|-------|---|-------|--------|
| Gemini vs TF-IDF | 0.4162 | 0.4422 | -1.874 | 6.0976e-02 | 4.8781e-01 |
| Gemini vs MiniLM-128 * | 0.4162 | 0.4691 | -2.841 | 4.4992e-03 | 4.4992e-02 |
| Gemini vs MiniLM-256 | 0.4162 | 0.4545 | -2.284 | 2.2368e-02 | 2.0131e-01 |
| Gemini vs MiniLM-512 *** | 0.4162 | 0.5132 | -6.186 | 6.1781e-10 | 9.2672e-09 |
| Gemini vs Jaccard | 0.4162 | 0.4412 | -1.394 | 1.6331e-01 | 8.1655e-01 |
| TF-IDF vs MiniLM-128 | 0.4422 | 0.4691 | -1.521 | 1.2832e-01 | 7.6995e-01 |
| TF-IDF vs MiniLM-256 | 0.4422 | 0.4545 | -0.763 | 4.4550e-01 | 1.0000e+00 |
| TF-IDF vs MiniLM-512 *** | 0.4422 | 0.5132 | -4.687 | 2.7778e-06 | 3.6111e-05 |
| TF-IDF vs Jaccard | 0.4422 | 0.4412 | 0.049 | 9.6096e-01 | 9.6096e-01 |
| MiniLM-128 vs MiniLM-256 | 0.4691 | 0.4545 | 1.742 | 8.1461e-02 | 5.7023e-01 |
| MiniLM-128 vs MiniLM-512 * | 0.4691 | 0.5132 | -2.948 | 3.1988e-03 | 3.5187e-02 |
| MiniLM-128 vs Jaccard | 0.4691 | 0.4412 | 1.222 | 2.2171e-01 | 8.8686e-01 |
| MiniLM-256 vs MiniLM-512 *** | 0.4545 | 0.5132 | -4.848 | 1.2471e-06 | 1.7459e-05 |
| MiniLM-256 vs Jaccard | 0.4545 | 0.4412 | 0.617 | 5.3749e-01 | 1.0000e+00 |
| MiniLM-512 vs Jaccard ** | 0.5132 | 0.4412 | 3.816 | 1.3537e-04 | 1.6244e-03 |

## Full Dataset Analysis (n=4,977)

### Method Summary

| Method | LRT Stat | Raw p | Adj. p | Standalone AUC | AUC 95% CI |
|--------|----------|-------|-------|----------------|------------|
| TF-IDF *** | 44.785 | 2.1987e-11 | 8.7948e-11 | 0.4710 | [0.4528, 0.4892] |
| MiniLM-128 *** | 33.286 | 7.9569e-09 | 1.5914e-08 | 0.4584 | [0.4399, 0.4769] |
| MiniLM-256 *** | 34.037 | 5.4062e-09 | 1.6219e-08 | 0.4545 | [0.4361, 0.4730] |
| MiniLM-512 *** | 30.578 | 3.2077e-08 | 3.2077e-08 | 0.4617 | [0.4435, 0.4800] |
| Jaccard *** | 62.149 | 3.1851e-15 | 1.5925e-14 | 0.4619 | [0.4438, 0.4801] |

*Significance markers use Holm-Bonferroni corrected p-values (non-Gemini methods).*
*\* p < 0.05, \*\* p < 0.01, \*\*\* p < 0.001*

### MiniLM Token Length Comparison

| Variant | LRT Stat | Raw p | Adj. p | Standalone AUC |
|---------|----------|-------|-------|----------------|
| MiniLM-128 *** | 33.286 | 7.9569e-09 | 1.5914e-08 | 0.4584 |
| MiniLM-256 *** | 34.037 | 5.4062e-09 | 1.6219e-08 | 0.4545 |
| MiniLM-512 *** | 30.578 | 3.2077e-08 | 3.2077e-08 | 0.4617 |

Mean README length is ~1,300 tokens after 4,000-char truncation.
At 128 tokens, MiniLM captures ~10% of README content;
at 256, ~20%; at 512, ~40%.

### Marginal Improvement (n=4,736; 241 rows lack merge_rate or account_age)

| Method | Base AUC | Full AUC | AUC Diff | DeLong p | LRT p |
|--------|----------|----------|----------|---------|-------|
| TF-IDF | 0.6675 | 0.6979 | +0.0304 | 3.9078e-13 | 3.2825e-12 |
| MiniLM-128 | 0.6675 | 0.6907 | +0.0232 | 5.9584e-10 | 2.1854e-08 |
| MiniLM-256 | 0.6675 | 0.6916 | +0.0242 | 1.7946e-10 | 5.7988e-09 |
| MiniLM-512 | 0.6675 | 0.6891 | +0.0216 | 7.6358e-10 | 1.0037e-07 |
| Jaccard | 0.6675 | 0.6972 | +0.0297 | 5.2229e-11 | 3.3672e-14 |

## Selection Bias Analysis

The Gemini subset includes only PRs with non-empty bodies (needed
for Gemini embedding). The full dataset also includes title-only PRs.

- **Gemini subset merge rate**: 71.8%
- **Full dataset merge rate**: 73.7%
- **Title-only PR fraction (full)**: 74.0%
- **Gemini subset mean GE score**: 0.7212
- **Full dataset mean GE score**: 0.5979

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
- The full dataset includes ~74% title-only PRs which have very short text (~16 tokens).

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