# Bot Detection Experiment Results

## Dataset

- **42,536 PRs** across 97 repos, 10,638 distinct authors
- Sources: neoteny DuckDB caches (primary + secondary) and PR 27 JSONL data
- Outcome distribution: 31,736 merged (74.6%), 6,298 rejected (14.8%), 4,502 pocket veto (10.6%)
- Non-merge rate: 25.4%
- 184 known bot PRs filtered before analysis

## H1: Burstiness

**AUC-ROC: 0.479 [0.473, 0.486], Mann-Whitney p = 8.4e-11**

The signal is statistically significant but *inverted*: bursty authors are more likely to be merged, not less. Authors with burst_count_24h >= 3 have a 12.7% non-merge rate vs 25.9% for non-bursty authors.

This makes sense in hindsight. An author who opens PRs across many repos in a short window is likely an active, experienced contributor with an established track record. They're power users, not spammers. Only 3.9% of PRs come from bursty authors (24h count >= 3), and 99.3% of those bursty authors have non-zero GE scores.

The H1 parameter sweep found 12 of 22 configs significant after Holm-Bonferroni correction:

| Config | Adj. p-value |
|---|---|
| burst_count_24h >= 3 | 4.1e-32 |
| burst_count_24h >= 5 | 6.0e-28 |
| burst_count_24h >= 2 | 6.0e-28 |
| burst_count_24h >= 7 | 1.7e-26 |
| burst_count_24h >= 10 | 2.0e-24 |

The association is real and very strong statistically, just in the opposite direction from what the DOE hypothesized.

## H2: Engagement Lifecycle

**AUC-ROC: 0.498 [0.492, 0.504], Mann-Whitney p = 0.52**

No signal. This is primarily a data limitation: the neoteny cache stores only merged PRs (which is where most of our data comes from), so the "abandoned PR" and "fire-and-forget" signals have almost no non-merged prior PRs to compute from.

- `abandoned_pr_rate`: 100% NaN (no prior closed-without-merge PRs in neoteny data)
- `review_response_rate`: 90.8% NaN
- `ci_failure_followup_rate`: 97.4% NaN

A proper test of H2 would require PR data that includes non-merged PRs with their review and commit history.

## H3: Cross-Repo Fingerprinting

**AUC-ROC: 0.479 [0.473, 0.485], Mann-Whitney p = 3.1e-11**

Like H1, statistically significant but inverted. Authors with higher cross-repo title similarity and more repos in their history tend to be *more* likely to merge. This again reflects that cross-repo activity is a marker of experience, not spam.

H3b (with language entropy features) significantly outperforms H3a (title similarity only) per DeLong test (z=3.17, p=0.0015), but in both cases the signal predicts merge, not non-merge.

## H4: Combined Model

**AUC-ROC: 0.495 [0.489, 0.501]**

Combining H1+H2+H3 features does not produce a useful non-merge predictor. All nested LRTs are significant (p < 1e-18), confirming the individual signals capture distinct variance, but that variance collectively predicts merge rather than non-merge.

## H5: GE Score Complement

### GE v1
- GE-only AUC: 0.502
- GE+bot signals AUC: 0.503
- DeLong: z=0.41, p=0.68 (no significant difference)
- LRT: chi2=485.3, p=1.9e-95

### GE v2
- GE-only AUC: 0.507
- GE+bot signals AUC: 0.503
- DeLong: z=-1.97, p=0.049 (marginally significant, bot signals *hurt*)
- LRT: chi2=443.9, p=1.2e-86

The LRT says the bot features add statistically significant information beyond GE alone, but the DeLong test says this information doesn't improve AUC -- and for v2, it slightly degrades it. The bot signals and GE scores are measuring overlapping constructs (cross-repo activity = experience = trust). Adding redundant features to logistic regression adds noise.

## Baselines (Stage 4)

| Baseline | AUC-ROC | 95% CI |
|---|---|---|
| GE v2 | 0.536 | [0.532, 0.540] |
| GE v1 | 0.531 | [0.527, 0.534] |
| Random | 0.502 | [0.496, 0.509] |
| Account age < 30d | 0.501 | [0.500, 0.501] |
| Zero followers | 0.500 | [0.500, 0.500] |
| Zero repos | 0.500 | [0.500, 0.500] |

GE v2 significantly outperforms v1 (DeLong p=2.7e-12). Both significantly outperform random. Account age, zero followers, and zero repos are useless as standalone predictors on this dataset.

## Key Correlations

| Feature pair | Pearson r |
|---|---|
| burst_count_24h vs ge_score_v1 | 0.369 |
| burst_repos_24h vs ge_score_v1 | 0.503 |
| burst_count_24h vs ge_score_v2 | 0.366 |
| burst_repos_24h vs ge_score_v2 | 0.421 |

Burstiness and GE scores are measuring the same underlying construct: how active and connected an author is across repos. The r=0.50 correlation between burst_repos_24h and ge_score_v1 confirms this.

## Interpretation

The central finding is that the behavioral signals hypothesized to detect bot-like spam activity (burstiness, cross-repo similarity, low engagement) are, in this dataset, markers of experienced contributors. The "300 PRs to 100 repos in 24 hours" pattern described in Issue #38 does not appear in this historical data at meaningful scale. The bursty authors in our dataset are power users, not spammers.

This doesn't mean burstiness can't detect spam. It means:

1. **Base rate matters.** In this dataset, ~0% of the 42K PRs are spam. The burstiness signal works as designed (it identifies cross-repo volume) but the population it identifies happens to be benign.

2. **The signal is confounded with experience.** An author active across many repos in a short window has, by definition, an established cross-repo history. That's exactly what the GE score measures. The two signals are r=0.50 correlated.

3. **To validate against actual spam**, we'd need a dataset that includes the specific episode described in Issue #38, or labeled spam/non-spam ground truth. The current dataset draws from established open-source projects where spam volume is negligible.

4. **GE v2 is the best available predictor** of non-merge outcomes on this data (AUC=0.536), modestly but significantly better than v1 (0.531). Neither is strong -- the 0.536 AUC reflects that most non-merges are legitimate PRs that were rejected on technical merit, not spam.

## Data Limitations

- Neoteny cache contains only merged PRs, so H2 engagement signals (abandoned PR rate, review response patterns on rejected PRs) couldn't be computed
- Author metadata (account age, followers) only available for ~28% of authors (PR 27 subset)
- No labeled spam ground truth -- non-merge is a weak proxy for "bad PR"
- GE scores computed without repo metadata (language, stars) since this wasn't cached, slightly underestimating true GE discriminative power

## Pipeline Details

- Stage 1: 42,536 classified PRs from 97 repos
- Stage 2: 42,536 feature rows (13 behavioral + 2 GE + 3 author metadata columns)
- Stage 3: 5-fold StratifiedGroupKFold CV grouped by repo, LogisticRegression(C=inf)
- Stage 4: 6 baselines with DeLong pairwise comparisons
- All features respect anti-lookahead: computed from author's other-repo PRs with created_at < T
