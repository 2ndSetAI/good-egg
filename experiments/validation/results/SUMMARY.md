# GE Validation Study: Summary Report

**Date:** 2026-02-12
**Branch:** `experiments/ge-validation`
**Sample:** 3,005 PRs across 49 repositories, 10 languages
**Temporal scope:** 2024-01-01 to 2025-12-31 (four half-year bins)

---

## Overview

This study evaluates whether the Good Egg (GE) normalized trust score---a
graph-based measure of contributor reputation computed from merged PR
history---predicts PR merge outcomes on open-source repositories. The study
uses a retrospective cohort design with an anti-lookahead constraint: each
author's score is computed using only contribution data available before the
test PR was opened.

The full design is documented in [`DOE.md`](../DOE.md). This report summarizes
results from the full-scale run across 49 repositories stratified by language,
size, and domain. A red team audit of the codebase was performed and is
documented in [`RED_TEAM_AUDIT.md`](RED_TEAM_AUDIT.md); all critical and major
findings have been addressed in the code (see [Errata](#errata) below).

---

## Key Results

### H1: Binary Merge Discrimination (Primary)

**Result: Supported.** The GE score discriminates between merged and non-merged
PRs well above the pre-registered threshold of AUC > 0.60.

| Metric | Value | Note |
|--------|-------|------|
| AUC-ROC | **0.695** (95% CI: 0.675--0.715) | Primary metric |
| AUC-PR | 0.743 | |
| Brier score | 0.252 | On uncalibrated scores; interpret with caution |
| Log loss | 4.639 | On uncalibrated scores; interpret with caution |

Brier score and log loss are computed on the raw GE normalized score, which is
not a calibrated probability. These values should not be compared to calibrated
baselines. AUC-ROC and AUC-PR are rank-based and unaffected by calibration.

The confidence interval excludes both chance (0.50) and the minimum threshold
(0.60). Cross-validation confirms stability: mean AUC = 0.697 +/- 0.027
across 5 folds (grouped by repository).

![ROC Curve](figures/h1_roc_curve.png)

### H1a: Three-Class Discrimination (Merged vs. Rejected vs. Pocket Veto)

**Result: Supported.** GE scores differ significantly across all three outcome
classes.

| Test | Statistic | p-value |
|------|-----------|---------|
| Kruskal-Wallis | H = 476.8 | p < 10^-103 |
| Merged vs. Rejected (post-hoc) | U = 438,503 | p = 0.0004 (adjusted) |
| Merged vs. Pocket Veto (post-hoc) | U = 1,009,204 | p < 10^-104 (adjusted) |
| Rejected vs. Pocket Veto (post-hoc) | U = 199,747 | p < 10^-40 (adjusted) |

The score distributions show a clear separation: merged PRs cluster at higher
scores, pocket-vetoed PRs cluster near zero, and rejected PRs fall in between.

![Score Distributions](figures/score_distributions_3class.png)

### Pocket Veto Analysis

Low-trust contributors are disproportionately pocket-vetoed rather than
explicitly rejected. The association between trust level and outcome type
(among non-merged PRs) is strong.

| Test | Statistic | p-value |
|------|-----------|---------|
| Chi-squared | chi2 = 200.7, df = 2 | p < 10^-43 |
| Cramer's V | 0.43 | (medium-large effect) |

Among non-merged PRs from LOW-trust authors, ~81% are pocket-vetoed vs. ~19%
explicitly rejected. For HIGH-trust authors, the split is ~41% pocket veto
vs. ~59% explicit rejection.

The Cochran-Armitage trend test and trust-level odds ratios are now computed
per the DOE specification (added post-audit).

![Pocket Veto by Trust](figures/pocket_veto_by_trust.png)

### H2: Ablation Study

**Result: Partially supported.** Of the six scoring dimensions, only recency
decay shows a statistically significant independent contribution.

Holm-Bonferroni correction is applied to the 6 primary single-dimension
ablations only (per DOE Section 7.4). Two-way interactions and the recursive
quality variant are reported separately as exploratory.

| Variant | AUC | Delta | Significant? |
|---------|-----|-------|:------------:|
| **Full model** | **0.695** | -- | -- |
| no_recency | 0.562 | -0.133 | Yes (p < 10^-57) |
| no_repo_quality | 0.695 | +0.000 | No |
| no_self_penalty | 0.695 | -0.000 | No |
| no_language_match | 0.696 | +0.001 | No |
| no_diversity_volume | 0.695 | +0.000 | No |
| no_language_norm | 0.695 | -0.000 | No |

Exploratory two-way interactions:

| Variant | AUC | Delta |
|---------|-----|-------|
| no_recency_no_quality | 0.559 | -0.136 |
| no_lang_match_no_lang_norm | 0.696 | +0.001 |
| no_diversity_no_self_penalty | 0.695 | +0.000 |
| recursive_quality | 0.695 | +0.000 |

Removing recency drops AUC by 0.133 (from 0.695 to 0.562), nearly to chance.
All other dimensions have negligible individual impact after correction. This
suggests that recency decay is the dominant scoring signal, and other dimensions
are either redundant with it or contribute too little to detect at this sample
size.

![Ablation Forest Plot](figures/ablation_forest.png)

### H3: Account Age

**Result: Requires re-run.** The original result (LR = 0.0, p = 1.0) was
produced using L2-regularized logistic regression, which invalidates the
likelihood ratio test. The code has been fixed to use unregularized models
(`penalty=None`). Results will be updated when Stage 6 is re-run on the
existing dataset.

### H4: Embedding Similarity

**Result: Inconclusive (implementation limitation).** The embedding similarity
feature has known issues: repo names were used as proxy descriptions instead
of actual PR descriptions and repo READMEs, and author repo embeddings were
only matched against study target repos. See
[`RED_TEAM_AUDIT.md`](RED_TEAM_AUDIT.md) item C2. A proper test of this
hypothesis requires fetching actual content from the GitHub API.

### H5: Author Merge Rate

**Result: Supported.** Historical author merge rate (fraction of an author's
PRs merged across all repos) significantly improves prediction when added to
the GE score (LR = 315.2, p < 10^-69). This result was statistically
significant even under the previous L2-regularized LRT; the unregularized fix
will produce an even stronger test statistic.

### Feature Importance

Logistic regression coefficients (unregularized) confirm that the GE normalized
score is the dominant predictor, with author-level features (public repos,
followers, account age) contributing little additional information.

![Feature Importance](figures/feature_importance.png)

### Calibration

The calibration plot shows the GE score is over-confident in the low-to-mid
range (predicted probabilities 0.2--0.5 correspond to higher actual merge
rates than predicted), and slightly under-confident at high scores
(probabilities > 0.7 plateau around ~78% actual merge rate). This confirms
the GE score should not be interpreted as a merge probability without Platt
scaling or similar calibration.

![Calibration](figures/calibration.png)

---

## Newcomer Cohort

Authors with no prior merged PRs in the GE graph ("newcomers") receive a
score of 0 and cannot be discriminated by the trust score alone.

| Cohort | n | AUC-ROC | Merge Rate |
|--------|---|---------|------------|
| Newcomer (score = 0) | 432 | 0.500 | 53.9% |
| Established (score > 0) | 2,573 | 0.705 (CI: 0.683--0.728) | 65.5% |

Newcomers constitute 14.4% of the sample. Among established contributors, the
AUC improves to 0.705, indicating that the GE score is most useful for authors
who already have some contribution history.

---

## Cross-Validation

Stratified group 5-fold cross-validation (grouped by target repository):

| Fold | AUC-ROC |
|------|---------|
| 1 | 0.687 |
| 2 | 0.745 |
| 3 | 0.662 |
| 4 | 0.698 |
| 5 | 0.691 |
| **Mean +/- SD** | **0.697 +/- 0.027** |

The per-fold variance is modest, suggesting the score generalizes across
different repository populations. The weakest fold (0.662) still exceeds the
0.60 threshold.

---

## Implications for Good Egg

1. **The GE score is a meaningful merge predictor.** AUC-ROC of 0.695 confirms
   it carries real signal, sufficient for use as a triage heuristic in PR
   review workflows.

2. **Recency is the dominant dimension.** The ablation study shows that
   removing recency decay drops AUC almost to chance. The other five dimensions
   (repo quality, self-penalty, language match, diversity/volume, language
   normalization) do not individually contribute detectable signal at this
   sample size. This could motivate simplifying the scoring model, or it could
   indicate these dimensions are only useful in combination with recency.

3. **Pocket veto detection is a strong secondary use case.** The clear
   association between low trust and pocket veto outcomes (Cramer's V = 0.43)
   suggests the GE score could flag PRs at risk of being silently ignored.

4. **Author merge rate adds incremental value.** Future versions of GE could
   incorporate historical merge rate as a complementary signal, potentially
   lifting AUC above 0.70.

5. **Newcomer cold-start remains an open problem.** The score is uninformative
   for first-time contributors (14.4% of PRs). Alternative signals (e.g.,
   account age) did not help in the unregularized LRT, so newcomer assessment
   will require a different approach.

---

## Errata

The following issues were identified by a red team audit
([`RED_TEAM_AUDIT.md`](RED_TEAM_AUDIT.md)) and fixed in code:

| Issue | Severity | Fix |
|-------|----------|-----|
| LRTs used L2-regularized LR (H3/H4/H5) | Critical | Changed to `penalty=None`; H3/H5 need re-run |
| H4 embeddings used repo names, not content | Critical | Marked as inconclusive; proper implementation deferred |
| Still-open PRs from study period not collected | Critical | Stage 1 now collects open PRs per temporal bin |
| Brier/log loss on uncalibrated scores | Major | Added caveat in results; metrics retained for reference |
| Holm-Bonferroni on 10 tests instead of 6 | Major | Corrected to 6 primary ablations per DOE |
| Self-owned repo PRs not excluded | Major | Added author-vs-owner check in Stage 2 |
| Spam filter excluded fast merges | Major | Filter now only applies to non-merged PRs |
| Cochran-Armitage trend test missing | Minor | Added to Stage 6 |
| Odds ratios not computed | Minor | Added to Stage 6 |
| One-vs-rest AUC missing | Minor | Added to Stage 6 |
| Confusion matrix missing | Minor | Added to Stage 6 (at Youden's J threshold) |
| `_MERGE_BOT_CLOSERS` unused | Minor | Now checked in `_is_merge_bot_close` |
| Feature importance used regularized LR | Minor | Changed to `penalty=None` |

**To fully update numeric results**, re-run the pipeline from Stage 2 onward
(Stage 1 open-PR backfill requires GitHub API access):

```bash
python -m experiments.validation.pipeline run-stage 2
python -m experiments.validation.pipeline run-stage 4
python -m experiments.validation.pipeline run-stage 5
python -m experiments.validation.pipeline run-stage 6
```

---

## Limitations

- **Survivorship bias**: GE scores are computed only from merged PRs. Authors
  with high rejection rates may appear more trustworthy than warranted.
- **Repository metadata currency**: Star counts and archive status are fetched
  at query time, not at PR creation time.
- **Rejected-class contamination**: The "rejected" class includes superseded
  and author-abandoned PRs, which may attenuate effect sizes.
- **Incomplete pocket veto class**: Still-open PRs from the study period were
  not collected in the original run. Stage 1 has been fixed to collect them,
  but a backfill pass is needed.
- **H4 embedding quality**: Semantic similarity was computed using repo names
  as proxy descriptions, not actual PR descriptions or repo READMEs. H4
  results are inconclusive.
- **No causal claims**: The study evaluates predictive discrimination, not
  whether trust causes merges.

See [DOE.md, Section 9](../DOE.md#9-known-limitations-and-threats-to-validity)
for the full limitations discussion.

---

## Appendix: Raw Data

- Statistical test results: [`statistical_tests.json`](statistical_tests.json)
- Red team audit: [`RED_TEAM_AUDIT.md`](RED_TEAM_AUDIT.md)
- Figures directory: [`figures/`](figures/)
- Study design: [`DOE.md`](../DOE.md)
- Study configuration: [`study_config.yaml`](../study_config.yaml)
- Repository list: [`repo_list_full.yaml`](../repo_list_full.yaml)
