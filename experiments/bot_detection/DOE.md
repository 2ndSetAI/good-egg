# Bot Detection Experiment: Design of Experiments

Pre-registered hypotheses and analysis plan for Issue #38.

## 1. Overview

Human contributors using AI coding tools can submit hundreds of PRs across dozens of repos in a single day. Issue #38 asks whether bot-like behavioral signals -- computed from cross-repo activity at PR submission time -- predict non-merge outcomes.

The experiment treats each PR as a binary classification instance: merged vs. non-merged (rejected or pocket-vetoed). All features are computed from the author's activity on *other* repos *before* the PR timestamp, preventing lookahead contamination.

## 2. Data Sources

**Primary corpus: neoteny DuckDB (998 MB)**
- 52 repos, 32,374 PRs, 3,308 unique authors
- Reviews present on ~78% of PRs, commits on 100%
- Date range: 2014--2026

**Secondary: neoteny main DuckDB (81 MB)**
- Currently locked; not included in initial analysis

**PR 27 dataset**
- 49 repos, 5,417 PRs, 2,540 authors

**After deduplication across sources:** ~35K PRs across ~55 unique repos.

## 3. Anti-Lookahead Protocol

For a PR opened on repo X at time T:

1. Compute all features from the author's activity on repos *other than X*, at times strictly before T (using `<`, not `<=`).
2. The test repo X is excluded from the feature window entirely.
3. Any feature that uses information from time >= T or from repo X is a bug.

This simulates what a reviewer could know about the author *at the moment the PR appears*, without any information from the PR itself or the target repo's history with that author.

## 4. Hypotheses

### H1: Burstiness Predicts Non-Merge

Authors who submit many PRs to many repos in a short window are more likely to have their PRs rejected.

**Features:**
- `burst_count_1h`: PRs opened by author in the 1 hour before T (excluding repo X)
- `burst_repos_1h`: distinct repos targeted in that 1-hour window
- `burst_count_24h`: PRs opened in the 24 hours before T
- `burst_repos_24h`: distinct repos in the 24-hour window
- `burst_max_rate`: maximum PRs-per-hour in any sliding 1-hour window within the 24 hours before T

**Parameter sweep:**
- Window sizes: {1h, 4h, 12h, 24h}
- Minimum repos touched: {2, 3, 5}
- Minimum PRs in window: {3, 5, 10}
- Total configurations: 4 x 3 x 3 = 36

**Statistical tests:**
- AUC-ROC > 0.5 for each configuration (one-sided Mann-Whitney U)
- Holm-Bonferroni correction across all 36 configurations (alpha = 0.05)
- Report best configuration and its corrected p-value

### H2: Engagement Lifecycle Predicts Non-Merge

Authors who don't follow up on their PRs -- ignoring reviews, not responding to CI failures, abandoning PRs -- are more likely to have future PRs rejected.

**Features (computed from author's prior PRs on other repos):**
- `review_response_rate`: fraction of review comments that received an author reply
- `ci_failure_followup_rate`: fraction of CI failures followed by a new commit within 48h
- `avg_response_latency_hours`: median time from review comment to author response
- `abandoned_pr_rate`: fraction of author's prior PRs that were never merged and had no activity after the first week

**Statistical tests:**
- Logistic regression with all four features vs. intercept-only null model
- AUC-ROC > 0.5
- Nested likelihood ratio test (LRT), chi-squared with df = 4

### H3: Cross-Repo Fingerprinting Predicts Non-Merge

Authors who submit near-identical PRs across repos (copy-paste contributions) are more likely to be rejected.

**H3a: TF-IDF + Cosine Similarity (zero API cost, fully reproducible)**
- TF-IDF vectorization of PR titles and body text
- Cosine similarity between the current PR and author's prior PRs on other repos

**H3b: Gemini Embeddings (semantic similarity, API cost)**
- Embed PR titles using Gemini text embedding API
- Cosine similarity in embedding space

**Features (both variants produce):**
- `max_title_similarity`: highest cosine similarity between this PR's title and any prior PR title by the same author on a different repo
- `language_entropy`: Shannon entropy of programming languages across author's prior repos
- `topic_coherence`: average pairwise similarity of PR titles within the author's history
- `duplicate_title_count`: number of prior PRs with title similarity > 0.9

**Statistical tests:**
- AUC-ROC > 0.5 for each variant independently
- DeLong test comparing H3a vs. H3b AUC-ROC (paired, two-sided)

### H4: Combined Model Outperforms Individual Signals

A model using features from H1 + H2 + H3 together outperforms any single hypothesis group.

**Statistical tests:**
- Nested LRT: combined model vs. each single-group model
- AUC-ROC comparison via DeLong test (combined vs. best single group)

### H5: Bot Signals Complement Good Egg Score

Adding bot-detection features to the existing Good Egg trust score improves prediction of non-merge outcomes.

**Statistical tests:**
- Nested LRT: (GE score + bot features) vs. (GE score alone)
- DeLong test comparing AUC-ROC of the two models
- Report marginal improvement in AUC-ROC with 95% CI

## 5. Statistical Methodology

**Classifier:** `LogisticRegression(penalty=None)` from scikit-learn for all models. No regularization, so coefficients are maximum likelihood estimates and LRT is valid.

**Cross-validation:** 5-fold `StratifiedGroupKFold`, grouped by repo. Each fold's test set contains entire repos, preventing leakage from repo-specific base rates. Stratification preserves the overall merge/non-merge ratio in each fold.

**AUC-ROC confidence intervals:** DeLong method (implemented in `stats.py`).

**Multiple comparison correction:** Holm-Bonferroni for the H1 parameter sweep (36 tests). Individual hypotheses (H2--H5) are tested at alpha = 0.05 without additional correction since they address distinct questions.

**Positive class:** Non-merge (rejected + pocket_veto). A higher predicted probability means the model thinks the PR is more likely to fail.

## 6. Outcome Classification

Each PR is labeled as one of three outcomes:

- **MERGED**: `merged_at` is not null.
- **REJECTED**: closed without merge, and time-to-close < stale threshold for that repo.
- **POCKET_VETO**: closed without merge and time-to-close >= stale threshold, or still open past the stale threshold.

**Stale threshold** (per repo):

```
stale_threshold = max(30 days, min(P90 of 2024-H1 merged time-to-merge, 180 days))
```

The P90 of merged TTM from the first half of 2024 captures each repo's normal merge cadence. The 30-day floor prevents overly aggressive classification in fast-moving repos; the 180-day cap prevents sluggish repos from hiding pocket vetoes.

For binary classification, REJECTED and POCKET_VETO are combined into a single non-merge class.

## 7. Ramp-Up Procedure

The experiment runs in three stages to catch problems early:

**Stage 1 -- Micro (2 repos):**
Prove the pipeline runs end-to-end. Verify feature extraction, anti-lookahead enforcement, and outcome classification on a minimal dataset. No statistical claims.

**Stage 2 -- Small (10 repos):**
Validate statistical power. Confirm that the dataset is large enough for the planned tests to detect a meaningful effect (AUC-ROC >= 0.60). Run all hypotheses and check for obvious data quality issues.

**Stage 3 -- Full (~55 repos):**
Final analysis on the complete deduplicated corpus. All results reported from this stage.

Each stage must pass its red team checkpoint (section 8) before proceeding.

## 8. Red Team Checkpoints

Four audit points where a second reviewer verifies correctness before the experiment proceeds.

**Checkpoint 1 -- Data Integrity (after Stage 1)**
- Verify PR counts match source databases
- Confirm no duplicate PRs across data sources
- Validate outcome labels against raw GitHub data for a random sample of 50 PRs
- Check date ranges and completeness

**Checkpoint 2 -- Anti-Lookahead (after Stage 1)**
- For 20 randomly sampled PRs, manually verify that no feature uses data from time >= T or repo X
- Check that burst windows are computed with strict `<` on timestamp
- Verify repo exclusion in cross-repo features

**Checkpoint 3 -- Statistical Methodology (after Stage 2)**
- Confirm GroupKFold splits contain entire repos (no repo appears in both train and test)
- Verify LRT degrees of freedom match feature counts
- Check Holm-Bonferroni implementation against a reference calculation
- Validate DeLong CI against bootstrap CI on 10 random folds

**Checkpoint 4 -- Scale-Up Validation (before Stage 3)**
- Compare Stage 2 results with Stage 3 for directional consistency
- Flag any hypothesis that flips direction between stages
- Verify no new repos introduced data quality issues

## 9. Baselines

Three baselines provide context for interpreting model performance:

**Good Egg trust score:**
AUC-ROC = 0.650 on the PR 27 validation dataset. This is the existing production signal.

**Simple heuristics:**
Account age, follower count, public repo count. These are the features a human reviewer might glance at. Expected AUC-ROC in the 0.52--0.58 range based on prior work.

**Random:**
AUC-ROC = 0.500. Any model that can't beat this is useless.
