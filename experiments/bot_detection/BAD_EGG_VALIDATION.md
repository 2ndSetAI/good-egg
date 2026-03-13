# Bad Egg Feature Validation Results

## TL;DR

**The Bad Egg suspension advisory model has no discriminative power on the production-relevant population.** When restricted to authors who have merged PRs (the only users who would actually be scored), all 10 candidate features produce AUCs at or below chance (0.47–0.56) across three temporal cutoffs. No feature survives LOO ablation. The previous AUC of 0.643 was an artifact of including 279 suspended users with zero merged PRs — users whose trivially distinguishable features (merge_rate=0, total_prs=0) inflated apparent performance but who would never be scored in production.

## Background

### Previous work (PR 44, bot-detection branch)

The bot-detection experiments (stages 1–15) evaluated author-level features for predicting PR outcomes and account suspension. Stage 15 ablation found `{merge_rate, median_additions, isolation_score}` as the recommended 3-feature set for suspension classification, with merge_rate as the only feature surviving decontamination (AUC 0.693±0.110 with temporal holdout).

However, that analysis used the **full labeled population** (12,898 authors), of which 86% of suspended accounts had zero merged PRs.

### The population problem

The Bad Egg suspicion score only runs when `user_data.merged_prs` is non-empty — it's gated by `if self.config.bad_egg.enabled and user_data.merged_prs:` in both v1 and v2 scoring paths. This means the model never evaluates users without merged PRs. Training on zero-PR users and then deploying only to users with merged PRs creates a fundamental train/serve skew.

### Previous (flawed) validation

A validation script trained on all 12,898 labeled authors (323 suspended / 12,575 active) and reported CV AUC 0.637–0.643. This appeared to justify a 2-feature or 8-feature model. But:
- 279 of 323 suspended users (86%) had 0 merged PRs
- These users had merge_rate=0, total_prs=0, career_span_days=0 — trivially separable
- merge_rate appeared dispensable only because it had zero variance in 86% of positives
- The 2-feature trim (career_span_days + mean_title_length) was based on invalid evidence

## This Validation

### Ground truth expansion

Before running validation, we completed ground truth coverage by checking all unchecked PR authors against the GitHub API:

| Metric | Before | After |
|--------|--------|-------|
| Total authors in DB | 14,413 | 31,307 |
| Authors with status checked | 12,898 | 31,296 |
| Suspended accounts | 323 | 739 |
| **Suspended with merged PRs** | **44** | **417** |

The expansion found 416 new suspended accounts (78 + 338 across two runs, with one network error requiring restart). The script (`check_account_status.py`) queries `GET /users/{login}` at 2s spacing with 50% rate limit budget, is idempotent, and writes directly to DuckDB on each request.

### Methodology

Replicates PR 44 (stage 15) methodology with the **correct population**: only authors with at least 1 merged PR before each cutoff date.

**Cutoffs**: 2022-07-01, 2023-01-01, 2024-01-01

**Population per cutoff** (authors with ≥1 merged PR before cutoff, with known account status):

| Cutoff | Total | Suspended | Active | CV method |
|--------|-------|-----------|--------|-----------|
| 2022-07-01 | 2,235 | 58 | 2,177 | 5-fold stratified |
| 2023-01-01 | 3,619 | 92 | 3,527 | 5-fold stratified |
| 2024-01-01 | 7,642 | 204 | 7,438 | 5-fold stratified |

**10 candidate features** (all computable in a GitHub Action, no account_age):
1. merge_rate — merged / total PRs before cutoff
2. total_prs — count of all PRs (log-transformed)
3. career_span_days — max-min PR dates in days (log-transformed)
4. mean_title_length — average PR title length
5. median_additions — median lines added in merged PRs (log-transformed)
6. median_files_changed — median files changed in merged PRs (log-transformed)
7. total_repos — count of distinct repos
8. isolation_score — fraction of author's repos with no multi-repo contributor overlap
9. hub_score — degree centrality on bipartite author-repo graph
10. bipartite_clustering — bipartite clustering coefficient

**Excluded**: account_age (100% NaN for suspended accounts — profiles unavailable, leaked indicator), LLM-based features (no commercial API calls in GH Action), k-NN features (circular — uses suspended accounts as seeds).

**Model**: LogisticRegression(class_weight="balanced"), StandardScaler per fold, 5-fold stratified CV.

**Statistical tests**: DeLong paired test for AUC comparison, Holm-Bonferroni correction (alpha=0.05).

### Script

`scripts/validate_bad_egg_features.py` — runs both analyses and prints results.

## Results

### LOO Ablation: no feature survives

Every feature is DISPENSABLE at every cutoff after Holm-Bonferroni correction. Several features actually *hurt* the model when included (negative delta = removing improves AUC).

**Cutoff 2022-07-01** (Full 10f AUC: 0.470 — worse than random):

| Feature | Ablated AUC | Delta | p-value | adj-p | Verdict |
|---------|-------------|-------|---------|-------|---------|
| merge_rate | 0.456 | +0.014 | 0.519 | 1.000 | DISPENSABLE |
| median_additions | 0.465 | +0.005 | 0.615 | 1.000 | DISPENSABLE |
| isolation_score | 0.470 | +0.000 | 0.949 | 1.000 | DISPENSABLE |
| career_span_days | 0.504 | -0.034 | 0.023 | 0.182 | DISPENSABLE |
| median_files_changed | 0.488 | -0.018 | 0.202 | 1.000 | DISPENSABLE |

**Cutoff 2023-01-01** (Full 10f AUC: 0.491):

| Feature | Ablated AUC | Delta | p-value | adj-p | Verdict |
|---------|-------------|-------|---------|-------|---------|
| career_span_days | 0.486 | +0.005 | 0.447 | 1.000 | DISPENSABLE |
| merge_rate | 0.500 | -0.009 | 0.103 | 1.000 | DISPENSABLE |
| median_files_changed | 0.511 | -0.020 | 0.176 | 1.000 | DISPENSABLE |

**Cutoff 2024-01-01** (Full 10f AUC: 0.533):

| Feature | Ablated AUC | Delta | p-value | adj-p | Verdict |
|---------|-------------|-------|---------|-------|---------|
| career_span_days | 0.515 | +0.018 | 0.053 | 0.367 | DISPENSABLE |
| median_additions | 0.542 | -0.010 | 0.002 | 0.023 | DISPENSABLE* |
| bipartite_clustering | 0.543 | -0.010 | 0.034 | 0.275 | DISPENSABLE |

\* median_additions has significant adjusted p at 2024-01-01 but removing it *improves* AUC (delta is negative), so it's correctly classified as DISPENSABLE.

**Summary across cutoffs**: 0/10 features marked KEEP at any cutoff.

### Forward Selection: AUC degrades with more features

At every cutoff, AUC peaks with 1-3 features and declines as more are added — classic overfitting on noise.

**Cutoff 2022-07-01**: bipartite_clustering starts at 0.541, degrades to 0.470 with all 10.

**Cutoff 2023-01-01**: career_span_days starts at 0.541, degrades to 0.491 with all 10.

**Cutoff 2024-01-01**: career_span_days starts at 0.562, peaks at 0.582 with {career_span_days, mean_title_length}, degrades to 0.533 with all 10.

No addition step achieves a significant DeLong p-value.

### Full 10f vs 3f (PR 44 recommended set) Comparison

| Cutoff | 10f AUC | 3f AUC | Delta | p-value |
|--------|---------|--------|-------|---------|
| 2022-07-01 | 0.465 | 0.435 | +0.030 | 0.489 |
| 2023-01-01 | 0.478 | 0.451 | +0.026 | 0.421 |
| 2024-01-01 | 0.557 | 0.536 | +0.021 | 0.258 |

Neither model meaningfully outperforms chance. The differences are not significant.

### Refit attempt (all data, 3-feature)

Refitting on all data through 2026 (18,795 authors, 415 suspended):
- **CV AUC: 0.552** — barely above chance
- Probability range: [0.31, 0.55] — max probability can't even reach 0.60
- At t=0.50: flags 59% of users with 2.5% precision
- At t=0.55: flags 1% of users with 1.6% precision
- At t=0.60+: flags nobody

The model cannot produce actionable thresholds.

## Why the Signal Disappeared

The original model's apparent AUC of 0.643 was driven almost entirely by the **zero-PR suspended accounts**. These 279 users (86% of suspended) had:
- merge_rate = 0 (no merged PRs, some had only closed/rejected PRs)
- total_prs typically very low
- career_span_days = 0 (single PR or none)
- All other features at zero or near-zero

Active users with merged PRs have non-zero values for these features by definition. This made separation trivial — but only for a population that would never be scored.

When we restrict to users who have merged PRs (the production population), suspended accounts are **behaviorally indistinguishable** from active ones. Suspended accounts that have merged PRs look like normal contributors — they got PRs merged into real repos, which requires passing code review. Their merge rates, PR sizes, title lengths, career spans, and network positions are all within normal ranges.

## Implications

1. **The Bad Egg model should not ship.** No feature combination produces actionable discrimination on the production-relevant population. The 8-feature model in config.py is fit on noise.

2. **The feature set is exhausted.** All 10 candidate features that can be computed in a GitHub Action have been tested. None works. This includes graph features (hub_score, bipartite_clustering) that had promising coefficients in the inflated-population model.

3. **Account age would help but is unavailable.** Suspended accounts have their profiles removed, making `created_at` inaccessible via the API. This is the strongest predictor of suspension (young accounts are disproportionately suspended) but it's a leaked indicator — it's unavailable precisely for the accounts we want to detect.

4. **The fundamental asymmetry**: Suspension correlates with account-level metadata (age, profile completeness, activity patterns across all of GitHub) rather than PR-level behavioral features. Users who get PRs merged have already passed a human filter (code review), making their PR behavior look legitimate.

## Appendix: Raw Output

Full output from `scripts/validate_bad_egg_features.py` is reproduced above. The script can be re-run with:

```bash
uv run python scripts/validate_bad_egg_features.py
```

Data: `experiments/bot_detection/data/bot_detection.duckdb` (31,307 authors, 200,172 PRs, 96 repos).
