# Proximity-Based Suspension Detection — Results

This report summarizes the results of proximity-based methods for detecting
suspended GitHub accounts among authors with merged PRs.

## Methodology

### Population

- **Primary**: 19,598 authors with merged PRs (417 suspended, 19,181 active)
- **Replication**: 31,293 authors (739 suspended, 30,554 active)
- **Temporal cutoffs**: 2022-07-01 (58 susp), 2023-01-01 (92), 2024-01-01 (204)

### Feature Sets

- **F10** (10 core behavioral): merge_rate, total_prs, career_span_days,
  mean_title_length, median_additions, median_files_changed, total_repos,
  isolation_score, hub_score, bipartite_clustering
- **F16** (16 extended): F10 + rejection_rate, hour_entropy, empty_body_rate,
  title_spam_score, weekend_ratio, prs_per_active_day
- **F16_no_mr**: F16 minus merge_rate and rejection_rate (decontaminated)

### Methods

- **k-NN cosine/euclidean**: NearestNeighbors on StandardScaled features,
  score = negative mean distance to k nearest seeds
- **Jaccard max**: max repo-set Jaccard similarity to any suspended seed
- **Jaccard mean-k5**: mean of top-5 Jaccard similarities to seeds
- **PPR**: Personalized PageRank on bipartite author-repo graph with
  restart on suspended seeds
- **Combined LR**: Logistic regression on behavioral features + proximity score

### Holdout Strategies

- **Strategy A** (discovery-order): 44 original seeds → 373 expansion test set
- **Strategy B** (suspended-only CV): 5-fold on suspended, active in every fold
- **Strategy C** (temporal): features from pre-cutoff PRs, CV within each cutoff

### Statistical Tests

- DeLong paired test for AUC comparisons
- Holm-Bonferroni correction for multiple comparisons

## 1. k-NN Proximity Results

### Strategy A: Discovery-Order Holdout (k-NN)

| Method | AUC-ROC | AUC-PR | P@25 | P@50 |
|--------|---------|--------|------|------|
| baseline_lr_F16 | 0.6257 | 0.0267 | 0.0000 | 0.0200 |
| baseline_lr_F16_no_mr | 0.6256 | 0.0267 | 0.0000 | 0.0400 |
| baseline_lr_F10 | 0.6115 | 0.0255 | 0.0000 | 0.0000 |
| knn_F16_euclidean_k15 | 0.4413 | 0.0154 | 0.0000 | 0.0000 |
| knn_F16_euclidean_k10 | 0.4378 | 0.0153 | 0.0000 | 0.0000 |
| knn_F16_euclidean_k3 | 0.4360 | 0.0152 | 0.0000 | 0.0000 |
| knn_F16_euclidean_k5 | 0.4357 | 0.0152 | 0.0000 | 0.0000 |
| knn_F16_no_mr_euclidean_k15 | 0.4294 | 0.0151 | 0.0000 | 0.0000 |
| knn_F16_no_mr_euclidean_k10 | 0.4266 | 0.0150 | 0.0000 | 0.0000 |
| baseline_merge_rate | 0.4257 | 0.0178 | 0.0000 | 0.0200 |
| knn_F16_no_mr_euclidean_k5 | 0.4238 | 0.0149 | 0.0000 | 0.0000 |
| knn_F16_no_mr_cosine_k3 | 0.4230 | 0.0149 | 0.0000 | 0.0000 |
| knn_F16_no_mr_euclidean_k3 | 0.4228 | 0.0148 | 0.0000 | 0.0000 |
| knn_F10_euclidean_k15 | 0.4210 | 0.0149 | 0.0000 | 0.0000 |
| knn_F16_no_mr_cosine_k5 | 0.4208 | 0.0149 | 0.0000 | 0.0000 |
| knn_F10_euclidean_k5 | 0.4206 | 0.0148 | 0.0000 | 0.0000 |
| knn_F10_euclidean_k3 | 0.4199 | 0.0148 | 0.0000 | 0.0000 |
| knn_F10_euclidean_k10 | 0.4182 | 0.0148 | 0.0000 | 0.0000 |
| knn_F16_cosine_k3 | 0.4170 | 0.0147 | 0.0000 | 0.0000 |
| knn_F16_cosine_k5 | 0.4152 | 0.0147 | 0.0000 | 0.0000 |
| knn_F16_no_mr_cosine_k10 | 0.4138 | 0.0147 | 0.0000 | 0.0000 |
| knn_F16_cosine_k10 | 0.4102 | 0.0146 | 0.0000 | 0.0000 |
| knn_F16_no_mr_cosine_k15 | 0.4075 | 0.0145 | 0.0000 | 0.0000 |
| knn_F16_cosine_k15 | 0.4064 | 0.0145 | 0.0000 | 0.0000 |
| knn_F10_cosine_k5 | 0.4051 | 0.0144 | 0.0000 | 0.0000 |
| knn_F10_cosine_k3 | 0.4045 | 0.0144 | 0.0000 | 0.0000 |
| knn_F10_cosine_k10 | 0.4005 | 0.0143 | 0.0000 | 0.0000 |
| knn_F10_cosine_k15 | 0.3980 | 0.0142 | 0.0000 | 0.0000 |

### Strategy B: Suspended-Only CV, Merged-PR Population (k-NN)

| Method | AUC-ROC | AUC-PR | P@25 | P@50 |
|--------|---------|--------|------|------|
| baseline_lr_F16 | 0.5727 | 0.0253 | 0.0000 | 0.0000 |
| baseline_lr_F16_no_mr | 0.5709 | 0.0255 | 0.0000 | 0.0400 |
| knn_F16_euclidean_k3 | 0.5698 | 0.0270 | 0.0400 | 0.0200 |
| knn_F16_euclidean_k5 | 0.5673 | 0.0266 | 0.0400 | 0.0400 |
| knn_F10_cosine_k10 | 0.5669 | 0.0264 | 0.0000 | 0.0000 |
| knn_F10_cosine_k5 | 0.5668 | 0.0264 | 0.0000 | 0.0200 |
| knn_F16_cosine_k3 | 0.5666 | 0.0274 | 0.0400 | 0.0200 |
| knn_F10_cosine_k15 | 0.5661 | 0.0266 | 0.0800 | 0.0600 |
| knn_F16_cosine_k5 | 0.5658 | 0.0271 | 0.0400 | 0.0200 |
| knn_F10_cosine_k3 | 0.5649 | 0.0263 | 0.0400 | 0.0200 |
| knn_F10_euclidean_k3 | 0.5639 | 0.0252 | 0.0000 | 0.0200 |
| knn_F16_euclidean_k10 | 0.5638 | 0.0261 | 0.0000 | 0.0600 |
| knn_F10_euclidean_k5 | 0.5635 | 0.0255 | 0.0000 | 0.0200 |
| knn_F16_no_mr_cosine_k3 | 0.5629 | 0.0273 | 0.0400 | 0.0400 |
| knn_F10_euclidean_k10 | 0.5622 | 0.0254 | 0.0000 | 0.0000 |
| knn_F16_cosine_k10 | 0.5622 | 0.0263 | 0.0000 | 0.0400 |
| knn_F16_euclidean_k15 | 0.5619 | 0.0257 | 0.0400 | 0.0400 |
| knn_F16_no_mr_euclidean_k3 | 0.5612 | 0.0266 | 0.0400 | 0.0200 |
| knn_F10_euclidean_k15 | 0.5610 | 0.0254 | 0.0000 | 0.0200 |
| knn_F16_no_mr_cosine_k5 | 0.5609 | 0.0269 | 0.0400 | 0.0200 |
| knn_F16_cosine_k15 | 0.5606 | 0.0285 | 0.0800 | 0.0600 |
| baseline_lr_F10 | 0.5586 | 0.0241 | 0.0000 | 0.0200 |
| knn_F16_no_mr_euclidean_k5 | 0.5577 | 0.0262 | 0.0400 | 0.0400 |
| knn_F16_no_mr_cosine_k10 | 0.5571 | 0.0262 | 0.0000 | 0.0400 |
| knn_F16_no_mr_cosine_k15 | 0.5561 | 0.0284 | 0.0800 | 0.0600 |
| knn_F16_no_mr_euclidean_k10 | 0.5537 | 0.0257 | 0.0000 | 0.0600 |
| knn_F16_no_mr_euclidean_k15 | 0.5523 | 0.0253 | 0.0400 | 0.0400 |
| baseline_merge_rate | 0.4492 | 0.0205 | 0.0400 | 0.0400 |

### Strategy B: Suspended-Only CV, All Authors (Stage 12 Replication)

| Method | AUC-ROC | AUC-PR | P@25 | P@50 |
|--------|---------|--------|------|------|
| baseline_lr_F16 | 0.6214 | 0.0400 | 0.0800 | 0.1000 |
| knn_F16_cosine_k5 | 0.5573 | 0.0303 | 0.0800 | 0.0800 |
| baseline_merge_rate | 0.5137 | 0.0255 | 0.0800 | 0.1000 |

### Strategy C: Temporal Holdout, cutoff=2022-07-01 (k-NN)

| Method | AUC-ROC | AUC-PR | P@25 | P@50 |
|--------|---------|--------|------|------|
| knn_F10_cosine_k5 | 0.5348 | 0.0283 | 0.0000 | 0.0200 |
| knn_F10_cosine_k10 | 0.5200 | 0.0268 | 0.0000 | 0.0000 |
| baseline_merge_rate | 0.4831 | 0.0275 | 0.0000 | 0.0600 |
| baseline_lr_F16 | 0.4806 | 0.0370 | 0.0800 | 0.0400 |
| knn_F16_cosine_k10 | 0.4762 | 0.0240 | 0.0000 | 0.0000 |
| baseline_lr_F10 | 0.4725 | 0.0239 | 0.0400 | 0.0200 |
| knn_F16_cosine_k5 | 0.4714 | 0.0264 | 0.0400 | 0.0200 |

### Strategy C: Temporal Holdout, cutoff=2023-01-01 (k-NN)

| Method | AUC-ROC | AUC-PR | P@25 | P@50 |
|--------|---------|--------|------|------|
| baseline_lr_F16 | 0.5346 | 0.0270 | 0.0000 | 0.0000 |
| knn_F10_cosine_k10 | 0.5246 | 0.0269 | 0.0400 | 0.0200 |
| knn_F10_cosine_k5 | 0.5189 | 0.0252 | 0.0000 | 0.0000 |
| knn_F16_cosine_k10 | 0.5089 | 0.0255 | 0.0000 | 0.0400 |
| knn_F16_cosine_k5 | 0.5062 | 0.0245 | 0.0000 | 0.0000 |
| baseline_lr_F10 | 0.5007 | 0.0256 | 0.0000 | 0.0000 |
| baseline_merge_rate | 0.4616 | 0.0255 | 0.0000 | 0.0200 |

### Strategy C: Temporal Holdout, cutoff=2024-01-01 (k-NN)

| Method | AUC-ROC | AUC-PR | P@25 | P@50 |
|--------|---------|--------|------|------|
| baseline_lr_F10 | 0.5516 | 0.0307 | 0.0000 | 0.0400 |
| baseline_lr_F16 | 0.5497 | 0.0301 | 0.0000 | 0.0400 |
| knn_F10_cosine_k10 | 0.5376 | 0.0386 | 0.0800 | 0.0400 |
| knn_F10_cosine_k5 | 0.5365 | 0.0294 | 0.0000 | 0.0200 |
| knn_F16_cosine_k10 | 0.5248 | 0.0388 | 0.0800 | 0.0400 |
| knn_F16_cosine_k5 | 0.5233 | 0.0300 | 0.0000 | 0.0200 |
| baseline_merge_rate | 0.4586 | 0.0263 | 0.0400 | 0.0400 |

## 2. Graph Proximity Results

### Strategy A: Discovery-Order Holdout (Graph)

| Method | AUC-ROC | AUC-PR | P@25 | P@50 |
|--------|---------|--------|------|------|
| jaccard_max | 0.5465 | 0.0225 | 0.0000 | 0.0000 |
| jaccard_mean_k5 | 0.5314 | 0.0210 | 0.0000 | 0.0000 |
| ppr | 0.4607 | 0.0165 | 0.0000 | 0.0000 |
| baseline_merge_rate | 0.4257 | 0.0178 | 0.0000 | 0.0200 |

### Strategy B: Suspended-Only CV, Merged-PR Population (Graph)

| Method | AUC-ROC | AUC-PR | P@25 | P@50 |
|--------|---------|--------|------|------|
| jaccard_max | 0.5952 | 0.0266 | 0.0800 | 0.0600 |
| jaccard_mean_k5 | 0.5906 | 0.0266 | 0.0800 | 0.0600 |
| ppr | 0.4787 | 0.0191 | 0.0000 | 0.0000 |
| baseline_merge_rate | 0.4492 | 0.0205 | 0.0400 | 0.0400 |

### Strategy C: Temporal Holdout, cutoff=2022-07-01 (Graph)

| Method | AUC-ROC | AUC-PR | P@25 | P@50 |
|--------|---------|--------|------|------|
| jaccard_max | 0.6060 | 0.0371 | 0.0000 | 0.0000 |
| ppr | 0.5223 | 0.0294 | 0.0400 | 0.0200 |
| baseline_merge_rate | 0.4831 | 0.0275 | 0.0000 | 0.0600 |

### Strategy C: Temporal Holdout, cutoff=2023-01-01 (Graph)

| Method | AUC-ROC | AUC-PR | P@25 | P@50 |
|--------|---------|--------|------|------|
| jaccard_max | 0.5607 | 0.0301 | 0.0000 | 0.0000 |
| ppr | 0.4799 | 0.0243 | 0.0400 | 0.0200 |
| baseline_merge_rate | 0.4616 | 0.0255 | 0.0400 | 0.0200 |

### Strategy C: Temporal Holdout, cutoff=2024-01-01 (Graph)

| Method | AUC-ROC | AUC-PR | P@25 | P@50 |
|--------|---------|--------|------|------|
| jaccard_max | 0.5557 | 0.0302 | 0.1200 | 0.0600 |
| baseline_merge_rate | 0.4586 | 0.0263 | 0.0000 | 0.0400 |
| ppr | 0.4424 | 0.0226 | 0.0000 | 0.0000 |

## 3. Combined Model Results (H4)

### F10 Features

Behavioral LR baseline: AUC = 0.5586

| Model | AUC | Delta | DeLong p |
|-------|-----|-------|----------|
| LR + knn | 0.5646 | 0.0060 | 0.0857 |
| LR + graph | 0.6079 | 0.0493 | 0.0000 |
| LR + both | 0.5795 | 0.0209 | 0.0014 |

### F16 Features

Behavioral LR baseline: AUC = 0.5727

| Model | AUC | Delta | DeLong p |
|-------|-----|-------|----------|
| LR + knn | 0.5696 | -0.0030 | 0.2952 |
| LR + graph | 0.6062 | 0.0336 | 0.0000 |
| LR + both | 0.5875 | 0.0148 | 0.0011 |

## 4. Hypothesis Verdicts

**H1**: Profile proximity detects suspension in merged-PR population (k-NN AUC > 0.55)

**Verdict**: SUPPORTED — Best AUC = 0.5698 > 0.55 threshold. Mean AUC = 0.5623.


**H2**: Graph-based proximity captures structural signal (AUC > 0.55)

**Verdict**: SUPPORTED — Best AUC = 0.5952 > 0.55 threshold. Mean AUC = 0.5548.


**H3**: Proximity signal is robust to seed selection bias

**Verdict**: NOT SUPPORTED — Strategy A AUC (0.4413) is substantially lower than Strategy B (0.5698), suggesting seed selection bias affects results.


**H4**: Proximity adds incremental value to behavioral features

**Verdict**: SUPPORTED — F10+graph_combined: delta=+0.0493, p=0.0000; F10+both_combined: delta=+0.0209, p=0.0014; F16+graph_combined: delta=+0.0336, p=0.0000; F16+both_combined: delta=+0.0148, p=0.0011


## 5. LLM Scoring Results (H5)

Tests whether LLM (Gemini 3.1 Pro) scoring of PR titles and bodies adds signal
beyond behavioral and graph-proximity features. Uses temporal cutoffs (Strategy C)
to prevent lookahead bias: the LLM only sees PR titles and bodies created before
the cutoff date.

### Prompt Variants

- **V1** (title-only): Up to 20 merged PR titles, cheapest
- **V2** (titles + bodies): Titles + first 500 chars of body for up to 10 PRs
- **V3** (full profile): V2 + author metadata (total_prs, merge_rate, total_repos, career_span)

Model: `gemini/gemini-3.1-pro-preview`, temperature=1.0, 30,131 total API calls
across 3 cutoffs × 3 variants. Score failures dropped (not defaulted).

### Standalone LLM Scoring

| Cutoff | Population | V1 AUC | V2 AUC | V3 AUC |
|--------|-----------|--------|--------|--------|
| 2022-07-01 | 2,235 (58 susp) | 0.5444 | 0.5695 | 0.5209 |
| 2023-01-01 | 3,619 (92 susp) | 0.5168 | 0.5167 | 0.5045 |
| 2024-01-01 | 7,642 (204 susp) | 0.5372 | 0.5469 | 0.5408 |

V2 (titles + bodies) is the best or tied-best variant at every cutoff. Standalone
AUC ranges 0.50–0.57, comparable to behavioral feature baselines but not strong
on its own. V3 (full profile) does not improve over V2, suggesting the metadata
block does not help the LLM beyond what it can infer from PR text.

### Combined Models (LR + LLM ± Jaccard, F10 features)

| Cutoff | LR Baseline | + Jaccard | + LLM (V2) | + LLM + Jaccard |
|--------|-------------|-----------|------------|-----------------|
| 2022-07-01 | 0.4600 | 0.4612 | 0.4914 | 0.4958 |
| 2023-01-01 | 0.5277 | 0.5321 | 0.5160 | 0.5222 |
| 2024-01-01 | 0.5357 | 0.5510 | 0.5630 | 0.5771 |

At the 2024-01-01 cutoff (largest population), LLM+Jaccard combined reaches
AUC 0.577, a +0.026 improvement over Jaccard alone (0.551). Three of 24 DeLong
tests are significant after Holm-Bonferroni correction — all from the 2024-01-01
cutoff. At earlier cutoffs with smaller populations, no tests reach significance.

### Second-Phase Re-ranking

LLM re-ranking of top-N candidates from the first-phase model (LR+Jaccard) was
tested at top-100, top-200, and top-500 with alpha sweeps blending first-phase
and LLM scores (z-normalized). Results are uniformly negative: LLM re-ranking
does not improve precision at any operating point across any cutoff or variant.

### H5 Verdict

**H5**: LLM scoring of PR text adds signal beyond behavioral + graph features

**Verdict**: WEAKLY SUPPORTED — On the largest population (2024-01-01, 7,642
authors), LLM combined with Jaccard achieves the best single-cutoff AUC (0.577)
and 3/24 DeLong tests survive Holm-Bonferroni correction. However, the effect
is small (+0.026 over Jaccard alone), does not replicate at earlier cutoffs with
smaller populations, and second-phase re-ranking is ineffective. The LLM provides
marginal incremental value as a combined LR feature but is not useful as a
standalone detector or re-ranker on the merged-PR population.

## 6. Summary

**Best overall method**: jaccard_max (Strategy B merged (Graph)), AUC = 0.5952

**Best combined method**: LR(F10) + LLM(V2) + Jaccard at 2024-01-01 cutoff,
AUC = 0.5771

The best methods exceed the AUC > 0.55 threshold, suggesting proximity-based
detection has *some* signal on the merged-PR population. However, the practical
value is limited — precision at operational thresholds (P@25, P@50) remains low
across all methods. LLM scoring provides marginal incremental value when combined
with Jaccard but is not useful standalone or for re-ranking.

**Stage 12 replication** (all-authors, F16, cosine, k=5): AUC = 0.5573 (original stage 12: 0.595)
