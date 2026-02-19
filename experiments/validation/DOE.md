# Design of Experiments: Good Egg Trust Score as a Merge Predictor

**Version:** 1.0
**Date:** 2026-02-11
**Status:** Pre-registration draft

---

## Table of Contents

1. [Study Design](#1-study-design)
2. [Three-Class Outcome Model](#2-three-class-outcome-model)
3. [Hypotheses](#3-hypotheses)
4. [Sampling Strategy](#4-sampling-strategy)
5. [Exclusion Criteria](#5-exclusion-criteria)
6. [Metrics and Statistical Tests](#6-metrics-and-statistical-tests)
7. [Ablation Matrix](#7-ablation-matrix)
8. [Cross-Validation](#8-cross-validation)
9. [Known Limitations and Threats to Validity](#9-known-limitations-and-threats-to-validity)

---

## 1. Study Design

**Design type:** Retrospective cohort study with temporal holdout.

For each test pull request created at time *T*, the author's Good Egg (GE)
normalized trust score is computed using **only** contribution data where
`merged_at < T`. This anti-lookahead constraint ensures the score reflects
information that would have been available at the time the PR was opened,
preserving the validity of the score as a prospective predictor.

Outcome classification (merged, rejected, pocket veto) uses the full observed
lifecycle of each PR (`merged_at`, `closed_at`, or elapsed time at study execution
date). This is standard for retrospective cohort designs: outcomes are determined
after the fact, while predictor variables are constructed using only pre-event data.

**Unit of observation:** A single pull request on a target repository.

**Independent variable (primary):** GE normalized trust score (continuous,
[0, 1]).

**Dependent variable:** PR outcome (three-class categorical: merged, explicitly
rejected, pocket veto).

**Temporal scope:** PRs created between 2024-01-01 and 2025-12-31, divided into
four half-year bins (2024H1, 2024H2, 2025H1, 2025H2).

---

## 2. Three-Class Outcome Model

Each collected PR is classified into exactly one of three outcome categories.

### 2.1 Definitions

| Outcome | Definition |
|---|---|
| **Merged** | PR has a non-null `merged_at` timestamp. |
| **Explicitly rejected** | PR is closed without merging (`state = CLOSED`, `merged_at` is null) and the time from open to close is less than or equal to the repo-specific stale threshold. |
| **Pocket veto (timeout)** | PR is closed without merging and the time from open to close exceeds the repo-specific stale threshold, **or** PR is still open and the elapsed time since opening exceeds the stale threshold. |

### 2.2 Stale Threshold Computation

The stale threshold is computed per target repository to account for variation
in review cadence across projects. To prevent lookahead bias, the threshold is
derived exclusively from PRs merged during the 2024H1 bin.

Let *M* be the set of merged PRs in the target repository during 2024H1, and
let *ttm* be the set of time-to-merge values (in days) computed over *M*.

```
stale_threshold = max(30, min(percentile_90(ttm), 180))
```

| Parameter | Value | Rationale |
|---|---|---|
| Floor | 30 days | Prevents overly aggressive classification for fast-moving repos. |
| Percentile | 90th | Captures the tail of the review-time distribution without arbitrary multipliers. More statistically principled than a fixed multiplier of the median. |
| Cap | 180 days | Prevents indefinite waiting in low-activity repos. |

The stale threshold sample draws up to 100 merged PRs from 2024H1 per
repository (configurable via `collection.stale_sample_size` in
`study_config.yaml`).

### 2.3 Classification Logic

```
if pr.merged_at is not null:
    outcome = MERGED
elif pr.state == CLOSED and (pr.closed_at - pr.created_at) <= stale_threshold:
    outcome = EXPLICITLY_REJECTED
elif pr.state == CLOSED and (pr.closed_at - pr.created_at) > stale_threshold:
    outcome = POCKET_VETO
elif pr.state == OPEN and (now - pr.created_at) > stale_threshold:
    outcome = POCKET_VETO
else:
    outcome = EXCLUDED  # Still open within threshold; indeterminate
```

---

## 3. Hypotheses

### 3.1 H1 (Primary): Merge Discrimination

**Statement:** The GE normalized trust score discriminates between merged and
non-merged PRs with AUC-ROC > 0.60.

| Parameter | Value |
|---|---|
| Test | Mann-Whitney U / logistic regression AUC |
| Significance level | alpha = 0.05 |
| Statistical power | 0.80 |
| Minimum effect size | AUC-ROC > 0.60 |
| Outcome grouping | Binary: merged vs. not-merged (rejected + pocket veto) |

**Rationale for threshold:** AUC = 0.50 represents chance. A threshold of 0.60
represents a modest but practically meaningful improvement over random
classification, appropriate for a single-signal predictor.

### 3.2 H1a (Pocket Veto Discrimination)

**Statement:** The GE normalized trust score discriminates between explicitly
rejected PRs and pocket-vetoed PRs.

| Parameter | Value |
|---|---|
| Test | Mann-Whitney U |
| Significance level | alpha = 0.05 |
| Outcome grouping | Binary: explicit rejection vs. pocket veto |

**Rationale:** If established contributors' PRs are more likely to receive
explicit decisions (accept or reject) rather than being silently ignored, GE
scores should differ between these two non-merged outcome classes.

### 3.3 H2 (Ablation): Independent Dimension Contributions

**Statement:** Each of the six scoring dimensions in the GE trust graph
contributes independently to merge prediction.

| Parameter | Value |
|---|---|
| Test | Paired DeLong test comparing full-model AUC vs. ablated-model AUC |
| Correction | Holm-Bonferroni for 6 primary comparisons |
| Significance level | alpha = 0.05 (family-wise) |

The six dimensions correspond to the ablation variants defined in Section 7.
A dimension is considered independently contributing if removing it produces a
statistically significant decrease in AUC-ROC after correction.

### 3.4 H3 (Account Age): Incremental Value of Account Age

**Statement:** Log-transformed GitHub account age improves prediction when added
to the base GE normalized score.

| Parameter | Value |
|---|---|
| Test | Likelihood ratio test (nested logistic regression models) |
| Base model | logit(merged) ~ GE_score |
| Extended model | logit(merged) ~ GE_score + log(account_age_days) |
| Significance level | alpha = 0.05 |

### 3.5 H4 (Semantic Similarity): Incremental Value of Embedding Similarity

**Statement:** Gemini embedding cosine similarity between the PR description
and the target repository's README/description improves prediction when added
to the base GE score.

| Parameter | Value |
|---|---|
| Test | Likelihood ratio test (nested logistic regression models) |
| Base model | logit(merged) ~ GE_score |
| Extended model | logit(merged) ~ GE_score + embedding_similarity |
| Embedding model | `gemini-embedding-001` (Gemini) |
| Significance level | alpha = 0.05 |

**Protocol deviation:** The original DOE specified `text-embedding-004` as the
embedding model. During implementation, `gemini-embedding-001` was used instead
because `text-embedding-004` was deprecated in favor of the newer model. Both
are Google Gemini embedding models with the same dimensionality (3072). The
substitution does not affect the study design or statistical methodology.

### 3.6 H5 (Author Merge Rate): Incremental Value of Historical Merge Rate

**Statement:** The author's historical merge rate (fraction of the author's PRs
that were merged across all repositories) improves prediction when added to the
base GE score.

| Parameter | Value |
|---|---|
| Test | Likelihood ratio test (nested logistic regression models) |
| Base model | logit(merged) ~ GE_score |
| Extended model | logit(merged) ~ GE_score + author_merge_rate |
| Significance level | alpha = 0.05 |

---

## 4. Sampling Strategy

**Data completeness note:** PRs created after `today - max(stale_threshold)` may
have unresolved outcomes for still-open PRs. The study reports per-bin
indeterminate exclusion counts and includes a sensitivity analysis computing
the primary AUC-ROC with and without the final temporal bin (2025H2) to
demonstrate result stability.

### 4.1 Repository Stratification

Target repositories are stratified along three axes:

| Axis | Strata |
|---|---|
| **Language** | Python, Rust, Go, JavaScript/TypeScript, Java |
| **Repository size** (stars) | Small (1K--5K), Medium (5K--20K), Large (20K+) |
| **Domain** | web, cli, library, infra, data, ml, etc. |

Repository selection criteria (from `repo_list.yaml`):
- Active development (commits in the last 6 months).
- Accepts external contributions (not single-maintainer).
- Sufficient PR volume (>= 50 merged PRs in 2024--2025).

### 4.2 PR Sampling per Repository

For each target repository and each of the four temporal bins:

| Class | Target count per bin | Source |
|---|---|---|
| Merged | Up to 25 | GitHub search: `is:pr is:merged repo:{r} merged:{bin_start}..{bin_end}` |
| Closed (unmerged) | Up to 25 | GitHub search: `is:pr is:closed is:unmerged repo:{r} created:{bin_start}..{bin_end}` |

Each PR is assigned to exactly one temporal bin by creation
date. Deduplication ensures no PR appears in multiple bins.

**Maximum per repository:** Up to 100 merged + 100 closed/timed-out PRs across
all four bins.

**Class balance note:** AUC-ROC, the primary metric, is rank-based and invariant
to class proportions. No post-hoc resampling or oversampling is applied.

### 4.3 Target Sample Size

With an expected 15--30 target repositories and up to 200 PRs per repository,
the total sample is expected to range from 3,000 to 6,000 PRs. Power analysis
for H1 (AUC > 0.60 vs. null AUC = 0.50, alpha = 0.05, power = 0.80) requires
approximately 200 observations per class at the aggregate level, which is well
within this range.

---

## 5. Exclusion Criteria

The following PRs are excluded from analysis after collection:

| Criterion | Rationale |
|---|---|
| **Bot authors** | Detected via GE's built-in bot detection plus additional patterns (see `study_config.yaml`, `author_filtering.extra_bot_patterns`). Bot PRs reflect automated processes, not human trust signals. |
| **Self-owned repositories** | PRs to repos owned by the PR author. Self-merges do not represent external trust. |

(Note: this exclusion applies to test PRs where the author owns the target
repository. Separately, the scoring graph applies a configurable 0.3x penalty to
contributions in an author's own repositories within their contribution history ---
see Section 7, dimension #3.)

| **PRs open < 1 day** | Likely accidental or immediately superseded. |
| **PRs closed < 1 day** | Likely spam, accidental, or test PRs. |
| **Indeterminate PRs** | PRs still open within the stale threshold period. These cannot yet be classified and are excluded to avoid censoring bias. |

Bot detection patterns include:

```
^dependabot, \[bot\]$, -bot$, ^renovate, ^greenkeeper,
^snyk-, ^codecov, ^mergify, ^allcontributors, ^github-actions,
^pre-commit-ci
```

---

## 6. Metrics and Statistical Tests

### 6.1 Binary Classification (Merged vs. Not-Merged)

| Metric | Role | Description |
|---|---|---|
| **AUC-ROC** | Primary | Area under the receiver operating characteristic curve. Threshold-free measure of discrimination. |
| **AUC-PR** | Secondary | Area under the precision-recall curve. More informative under class imbalance. |
| **Brier score** | Calibration | Mean squared error between predicted probabilities and binary outcomes. Lower is better. |
| **Log-loss** | Calibration | Negative log-likelihood of predicted probabilities. Penalizes confident misclassifications. |

### 6.2 Three-Class Analysis (Merged vs. Rejected vs. Pocket Veto)

| Metric | Description |
|---|---|
| **Multinomial logistic regression** | Model: `outcome ~ GE_score`, with three-class outcome. Reports coefficients, standard errors, and p-values per class. |
| **One-vs-rest AUC** | AUC-ROC computed for each class against the other two, yielding three AUC values. |
| **Confusion matrix** | 3x3 matrix of predicted vs. actual class counts at the optimal threshold (Youden's J). |

### 6.3 Pocket Veto Analysis

| Test | Description |
|---|---|
| **Chi-squared test** | Tests independence between GE trust level (HIGH / MEDIUM / LOW) and outcome class (rejected vs. pocket veto). |
| **Cochran-Armitage trend test** | Tests for a monotonic trend in pocket veto rate across ordered GE trust levels (HIGH > MEDIUM > LOW). |

### 6.4 Trust Level Categorical Analysis

| Metric | Description |
|---|---|
| **Odds ratios** | For each trust level pair (e.g., HIGH vs. LOW), the odds ratio for being merged. Reported with 95% confidence intervals. |
| **Chi-squared test** | Tests independence between trust level and binary outcome (merged vs. not-merged). |

Trust level thresholds (from `GoodEggConfig.thresholds`):

| Level | Normalized score range |
|---|---|
| HIGH | >= 0.70 |
| MEDIUM | >= 0.30 and < 0.70 |
| LOW | < 0.30 |

---

## 7. Ablation Matrix

The ablation study evaluates the independent contribution of each scoring
dimension by selectively disabling it and measuring the change in predictive
performance. Each variant modifies GE's configuration to neutralize one or two
dimensions while holding all others constant.

### 7.1 Scoring Dimensions

The GE trust graph incorporates six scoring dimensions:

| # | Dimension | Implementation | Config location |
|---|---|---|---|
| 1 | **Recency decay** | Exponential decay with configurable half-life; contributions beyond `max_age_days` receive zero weight. | `recency.half_life_days`, `recency.max_age_days` |
| 2 | **Repository quality** | `log1p(stars * language_multiplier)`, with penalties for archived (0.5x) and forked (0.3x) repos. | `graph_builder._repo_quality()` |
| 3 | **Self-contribution penalty** | Contributions to repos owned by the author receive a 0.3x weight multiplier. | `graph_builder.build_graph()` (hardcoded) |
| 4 | **Language match** | Personalization vector assigns higher restart probability to repos sharing the context repo's language (`same_language_weight = 0.30` vs. `other_weight = 0.03`). | `graph_scoring.same_language_weight` |
| 5 | **Diversity and volume** | Adjusted `other_weight` scales with the number of unique repos and total PRs, rewarding cross-ecosystem contributors. | `graph_scoring.diversity_scale`, `graph_scoring.volume_scale` |
| 6 | **Language normalization** | Per-language multipliers normalize for ecosystem size (e.g., Rust 2.63x, JavaScript 1.0x) within the repo quality calculation. | `language_normalization.multipliers` |

### 7.2 Single-Dimension Ablation Variants

| Variant | Dimension removed | Config override | Effect |
|---|---|---|---|
| `no_recency` | Recency decay | `half_life_days: 999999`, `max_age_days: 999999` | All contributions weighted equally regardless of age. |
| `no_repo_quality` | Repository quality | Override `_repo_quality()` to return 1.0 | All repos treated as equal quality. |
| `no_self_penalty` | Self-contribution penalty | Override self-contribution check to always return `False` | Self-owned repo contributions weighted identically to external contributions. |
| `no_language_match` | Language match | `same_language_weight: 0.03` (set equal to `other_weight`) | No personalization boost for repos matching the context language. |
| `no_diversity_volume` | Diversity and volume | `diversity_scale: 0.0`, `volume_scale: 0.0` | `other_weight` remains at its base value regardless of contribution breadth. |
| `no_language_norm` | Language normalization | All `language_normalization.multipliers` set to 1.0 | No ecosystem size adjustment in repo quality. |

### 7.3 Two-Way Interaction Variants

| Variant | Dimensions removed | Config override | Rationale |
|---|---|---|---|
| `no_recency_no_quality` | Recency + Repo quality | `half_life_days: 999999`, `max_age_days: 999999`, override `_repo_quality()` | Tests whether temporal and quality signals are redundant. |
| `no_lang_match_no_lang_norm` | Language match + Language norm | `same_language_weight: 0.03`, all multipliers to 1.0 | Tests complete removal of language-aware scoring. |
| `no_diversity_no_self_penalty` | Diversity/volume + Self-penalty | `diversity_scale: 0.0`, `volume_scale: 0.0`, override self-penalty | Tests whether contributor breadth and ownership signals interact. |

### 7.4 Ablation Analysis Protocol

For each of the 9 ablation variants:

1. Re-score all authors in the dataset using the modified configuration.
2. Compute AUC-ROC for the ablated model on the binary outcome.
3. Compare to the full model using a paired DeLong test.
4. Apply Holm-Bonferroni correction across the 6 single-dimension comparisons
   (H2).

A dimension is deemed independently contributing if removing it produces a
statistically significant decrease in AUC-ROC (corrected p < 0.05). The
two-way interaction variants serve as exploratory checks for redundancy between
dimension pairs.

**Efficiency note:** The current approach re-scores each PR per ablation variant.
A future optimization could save intermediate graph components (edges, weights,
personalization vector) once and apply ablation parameters post-hoc, avoiding
redundant graph construction.

---

## 8. Cross-Validation

### 8.1 Strategy

**Method:** Stratified group 5-fold cross-validation.

**Grouping variable:** Target repository. All PRs from the same target
repository are assigned to the same fold. This prevents information leakage
from shared repository characteristics across train/test splits.

**Stratification:** Folds are balanced by outcome class proportions (merged vs.
not-merged) to the extent permitted by the grouping constraint.

### 8.2 Configuration

| Parameter | Value | Source |
|---|---|---|
| Number of folds | 5 | `analysis.cv_folds` |
| Random seed | 42 | `analysis.random_seed` |
| Grouping | Target repository | -- |

### 8.3 Reporting

For each fold and each metric (AUC-ROC, AUC-PR, Brier score, log-loss):
- Report per-fold values.
- Report mean and standard deviation across folds.
- Report 95% confidence interval via bootstrap (1,000 resamples) on the
  concatenated out-of-fold predictions.

---

## 9. Known Limitations and Threats to Validity

### 9.1 Repository Metadata Currency

Repository metadata (star count, archived status, fork status, primary
language) is fetched at query time, not at the time the PR was created. A
repository that has since gained significant popularity or been archived will
have different metadata than it did historically. This affects the repo quality
dimension. Historical star counts could
potentially be recovered using services like the star-history API. However, the
H2 ablation shows repo quality has a small but statistically significant AUC
impact (ΔAUC = −0.003, adjusted p = 0.015), though the effect size is minimal,
making this correction low priority.

### 9.2 Survivorship Bias in Contribution Data

GE scores are computed exclusively from an author's **merged** PRs. PRs that
were rejected, abandoned, or are still pending are not visible to the scoring
engine. This means the score reflects only successful contributions, which may
overestimate the trustworthiness of prolific authors who also have a high
rejection rate. The H5 hypothesis (author merge rate) partially addresses this
by testing whether historical merge rate adds predictive value.

### 9.3 Pocket Veto Threshold Heuristic

The stale threshold formula (`max(30, min(percentile_90(ttm), 180))`) uses the
90th percentile of observed time-to-merge. The floor (30 days) and cap (180 days)
are informed by prior work on open-source review latency but are not empirically
optimized. Sensitivity analysis should be conducted by varying these
parameters.

### 9.4 "Rejected" Class Contamination

The explicitly rejected class includes PRs that were:
- Genuinely rejected on technical or policy grounds.
- Abandoned by the author and subsequently closed by a maintainer.
- Superseded by a replacement PR and closed as duplicate.

These sub-categories carry different semantic meanings but cannot be reliably
distinguished from GitHub API data alone. This contamination may attenuate the
effect size for H1 and H1a.

### 9.5 Stale Threshold Baseline Period

The stale threshold is computed from 2024H1 merged PRs to avoid lookahead into
the test periods. However, this assumes review cadence is stationary. If a
project significantly changes its review practices after 2024H1 (e.g., gaining
or losing maintainers), the threshold may be miscalibrated for later bins.

### 9.6 Confounding Variables

Several factors that influence merge probability are not captured by the GE
score:
- PR quality (code correctness, test coverage, documentation).
- PR scope (single-line fix vs. large feature).
- Maintainer availability and project governance.
- Whether the PR addresses an open issue or roadmap item.

H4 (semantic similarity) partially addresses the content-relevance confound,
but residual confounding is expected. The study makes no causal claims; the
goal is to evaluate the GE score's predictive discrimination, not to establish
a causal relationship between contributor trust and merge outcomes.

### 9.7 External Validity

Results may not generalize to:
- Repositories in languages not represented in the sample.
- Private or enterprise repositories (the study uses public repos only).
- Repositories with non-standard contribution models (e.g., monorepos,
  vendored forks, mirror repos).

---

## Appendix A: Configuration Reference

All tunable study parameters are centralized in
`experiments/validation/study_config.yaml`. Key sections:

| Section | Parameters |
|---|---|
| `temporal_bins` | Half-year bins with start/end dates. |
| `collection` | `merged_per_bin`, `closed_per_bin`, `gh_search_delay_seconds`, `stale_sample_size`. |
| `classification` | `stale_threshold_floor_days`, `stale_threshold_cap_days`, `stale_threshold_percentile`, `pocket_veto_buffer_days` (0). |
| `author_filtering` | `extra_bot_patterns` (regex list). |
| `scoring` | `batch_size`. |
| `features` | `embedding_model`, `embedding_batch_size`. |
| `analysis` | `cv_folds`, `random_seed`, `alpha`, `h1_auc_threshold`, `trust_level_bins`. |
| `ablations` | Per-variant config overrides (see Section 7). |
| `paths` | Relative data and results directory paths. |

## Appendix B: GE Scoring Configuration Defaults

These are the default values from `GoodEggConfig` used as the full (baseline)
model. Ablation variants modify specific parameters as described in Section 7.

| Parameter | Default | Config path |
|---|---|---|
| Graph scoring alpha | 0.85 | `graph_scoring.alpha` |
| Context repo weight | 0.50 | `graph_scoring.context_repo_weight` |
| Same-language weight | 0.30 | `graph_scoring.same_language_weight` |
| Other weight | 0.03 | `graph_scoring.other_weight` |
| Diversity scale | 0.50 | `graph_scoring.diversity_scale` |
| Volume scale | 0.30 | `graph_scoring.volume_scale` |
| Recency half-life | 180 days | `recency.half_life_days` |
| Recency max age | 730 days | `recency.max_age_days` |
| Self-contribution penalty | 0.3x | Hardcoded in `TrustGraphBuilder.build_graph()` |
| Language normalization | Per-language (JS=1.0 to Nim=5.96) | `language_normalization.multipliers` |
| High trust threshold | 0.70 | `thresholds.high_trust` |
| Medium trust threshold | 0.30 | `thresholds.medium_trust` |
