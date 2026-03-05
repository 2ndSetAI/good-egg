# Bot Detection Experiment Results

## Dataset (Iteration 2)

- **38,502 PRs** across 97 repos, 10,617 distinct authors
- Sources: neoteny DuckDB caches (primary + secondary) and PR 27 JSONL data
- Outcome distribution: 28,661 merged (74.4%), 5,587 rejected (14.5%), 4,254 pocket veto (11.1%)
- Non-merge rate: 25.6%
- **4,218 bot PRs filtered** (31 bot authors) -- up from 184 in iteration 1 after adding `^app/` pattern to catch `app/*` accounts

### Iteration 2 Changes

- **Bot filter fix (WS1)**: Added `^app/` regex pattern, removing 4,034 additional `app/*` bot PRs (e.g. `app/copybara-service`, `app/dependabot`)
- **Interaction features (WS3/H6)**: `burst_no_prior_merge`, `burst_first_time_repo`, `burst_low_ge`, `burst_new_account`
- **Content homogeneity features (WS4/H7)**: `burst_size_cv`, `burst_file_pattern_entropy` (embedding similarity deferred -- requires Gemini API key)
- **Account status script (WS2)**: Ready to run but requires GITHUB_TOKEN (not executed in this iteration)

## H1: Burstiness

**AUC-ROC: 0.490 [0.483, 0.496], Mann-Whitney p = 0.0016**

The signal is statistically significant but *inverted*: bursty authors are more likely to be merged, not less. After the bot filter fix, the effect is weaker (AUC moved from 0.479 to 0.490, closer to 0.5), suggesting some of the previous inversion was from `app/*` bots that were merged automatically.

The H1 parameter sweep found 2 of 15 configs significant after Holm-Bonferroni correction:

| Config | Adj. p-value |
|---|---|
| burst_count_24h >= 1 | 0.0037 |
| burst_repos_24h >= 1 | 0.0037 |

With cleaner data, the burstiness signal is weaker. The pattern remains: bursty = experienced contributor, not spammer.

## H2: Engagement Lifecycle

**AUC-ROC: 0.500 [0.493, 0.506], Mann-Whitney p = 0.95**

No signal. Same data limitation as iteration 1: neoteny cache stores mainly merged PRs, so engagement signals computed from rejected PRs are overwhelmingly NaN.

## H3: Cross-Repo Fingerprinting

**AUC-ROC: 0.512 [0.506, 0.519], Mann-Whitney p = 0.00025**

Flipped from iteration 1 (was 0.479, now 0.512). With the `app/*` bots removed, cross-repo features now weakly predict non-merge rather than merge. The H3b model (with entropy features) significantly outperforms H3a (DeLong z=-10.74, p=6.7e-27).

This is the strongest single-hypothesis signal in the iteration 2 results.

## H4: Combined Model

**AUC-ROC: 0.518 [0.512, 0.524]**

Combining H1+H2+H3 improves on any single signal. All nested LRTs are significant (p < 1e-20). The combined AUC of 0.518 is a modest improvement over iteration 1 (was 0.495), driven by the H3 flip after bot filter cleanup.

## H5: GE Score Complement

### GE v1
- GE-only AUC: 0.511
- GE+bot signals AUC: 0.519
- DeLong: z=3.98, p=6.9e-05 (significant improvement)
- LRT: chi2=166.0, p=1.2e-28

### GE v2
- GE-only AUC: 0.518
- GE+bot signals AUC: 0.521
- DeLong: z=2.85, p=0.0044 (significant improvement)
- LRT: chi2=160.7, p=1.4e-27

Both GE models benefit from adding bot signals. In iteration 1, adding bot signals to GE v2 *hurt* performance (z=-1.97). Now that the `app/*` confounders are removed, the bot signals provide genuine complementary information. The improvement is modest (0.511 -> 0.519 for v1, 0.518 -> 0.521 for v2).

## H6: Interaction Features (Burstiness x Novelty)

**AUC-ROC: 0.493 [0.486, 0.499], Mann-Whitney p = 0.029**

The interaction features attempt to separate bursty power users from bursty spammers. A bursty author without a track record (no prior merges, first time at repo, low GE) is more suspicious than one with established history.

Feature distributions show the core problem: there are almost no suspicious bursty authors.

| Feature | Nonzero count | % of dataset |
|---|---|---|
| burst_no_prior_merge | 78 | 0.2% |
| burst_first_time_repo | 102 | 0.3% |
| burst_low_ge | 78 | 0.2% |
| burst_new_account | 7 | 0.0% |

The candidate spammers (yashwantbezawada, ayushm98, arrdel) all correctly trigger the interaction features, but 78 nonzero values out of 38K PRs isn't enough signal density to move the AUC.

## H7: Burst Content Homogeneity

**AUC-ROC: 0.492 [0.486, 0.499], Mann-Whitney p = 0.023**

Within-burst content features (size CV, repo entropy) weakly predict non-merge (AUC < 0.5 = inverted). Only 113 of 38K PRs have computable burst content features (requires >= 2 PRs in the 24h window). The embedding similarity feature (`burst_title_embedding_sim`) was not computed in this run (requires Gemini API key).

## Baselines (Stage 4)

| Baseline | AUC-ROC | 95% CI |
|---|---|---|
| GE v2 | 0.527 | [0.524, 0.531] |
| GE v1 | 0.519 | [0.516, 0.522] |
| Random | 0.503 | [0.496, 0.510] |
| Account age < 30d | 0.501 | [0.500, 0.502] |
| Zero followers | 0.500 | [0.500, 0.500] |
| Zero repos | 0.500 | [0.500, 0.500] |

GE v2 remains the best single predictor. With the cleaner dataset (bot filter fix), GE AUCs dropped slightly from iteration 1 (v2: 0.536 -> 0.527), suggesting the `app/*` bots were easy merges that inflated the previous GE numbers.

## Interpretation (Iteration 2)

The bot filter fix was the most impactful change. Removing 4,034 `app/*` bot PRs:

1. **Flipped H3** from inverted (0.479) to correctly-oriented (0.512). The `app/*` bots had uniform cross-repo patterns that looked like "experienced" contributors to the H3 model.
2. **Made H5 positive**: Bot signals now genuinely complement GE scores (both v1 and v2) instead of hurting them.
3. **Reduced noise** in the burstiness signal (H1 moved from 0.479 to 0.490).

The interaction features (H6) and content homogeneity features (H7) didn't move the needle. The core problem is signal density: only 78-102 out of 38K PRs trigger the interaction conditions. This dataset has almost no spam — the suspicious bursty-with-no-track-record pattern exists (3 confirmed candidates) but at a rate too low to train or evaluate a model.

**What we've established:**
- Burstiness alone cannot distinguish spammers from power users (both are bursty)
- The `burst × no_prior_merge` interaction correctly identifies the 3 known candidates but has near-zero base rate
- Cross-repo fingerprinting (H3) shows the most promise after cleaning, weakly predicting non-merge
- GE v2 remains the best single predictor (AUC=0.527)
- Bot signals genuinely complement GE when the data is clean

**What would help next:**
- Account status ground truth (WS2 script ready, needs GITHUB_TOKEN) — if suspended authors cluster in the bursty-low-GE quadrant, that validates the interaction features
- Gemini embeddings for title similarity within bursts — requires API key
- A dataset with actual labeled spam episodes

## Data Limitations

- Neoteny cache contains only merged PRs, so H2 engagement signals couldn't be computed
- Author metadata (account age, followers) only available for ~28% of authors (PR 27 subset)
- No labeled spam ground truth — non-merge is a weak proxy for "bad PR"
- Account status (suspended vs active) not yet collected — script ready but requires GITHUB_TOKEN
- Embedding similarity features require Gemini API key (not available in this run)

## Pipeline Details

- Stage 1: 38,502 classified PRs from 97 repos (down from 42,536 after bot filter fix)
- Stage 2: 38,502 feature rows (13 behavioral + 4 interaction + 3 content + 2 GE + 4 author metadata columns)
- Stage 3: 5-fold StratifiedGroupKFold CV grouped by repo, LogisticRegression(C=inf), 8 hypotheses (H1-H7)
- Stage 4: 6 baselines with DeLong pairwise comparisons
- All features respect anti-lookahead: computed from author's other-repo PRs with created_at < T
