# Bot Detection Experiment Results

## Dataset (Iteration 4)

- **200,172 PRs** across 96 repos, 31,296 distinct authors
- Primary source: OSS parquet files (187,534 PRs from 59 repos), gap-filled with neoteny DuckDB cache (8,127 PRs + 77K reviews + 150K commits) and PR 27 JSONL (4,511 PRs)
- 3 repos excluded at import due to low merge rate (<10%): apache/spark (0.0%), facebookresearch/detectron2 (0.0%), pytorch/pytorch (1.8%) -- all use cherry-pick/phabricator workflows
- Outcome distribution: 146,033 merged (73.0%), 23,245 rejected (11.6%), 30,894 pocket veto (15.4%)
- Non-merge rate: 27.0%
- **52,659 bot PRs filtered** (57 bot authors)
- Cross-repo coverage: **3,208 authors (10.3%) appear in 2+ repos** (up from 3.9% in iterations 1-3)

### Iteration 4 Changes (over Iteration 3)

- **Full parquet data**: Replaced DuckDB cache (~800 PRs/repo, 52 repos) with full neoteny parquet files (327K PRs, 62 repos). After merge-rate exclusions and bot filtering: 200K PRs vs previous 38.5K.
- **DuckDB indexes**: Added indexes on `(author, created_at)`, `(author, repo, created_at)`, reviews `(repo, pr_number)`, commits `(repo, pr_number)`. Stage 2 ran in ~74 min vs estimated hours without indexes.
- **Cross-repo coverage 2.6x better**: 10.3% of authors in 2+ repos (was 3.9%), 762 in 3+ repos, 114 in 5+ repos.
- **Neoteny cache demoted to gap-fill**: Parquet is imported first (INSERT OR IGNORE), so parquet data wins dedup. Neoteny provides reviews/commits not in parquet.

## Iterations 1-4: PR-Level Features (H1-H7)

All PR-level behavioral features produced AUC 0.479-0.503 against a "merged vs not-merged" target. Full details below.

### H1: Burstiness -- AUC 0.483

Inverted: bursty authors are more likely to get merged. Burstiness is a signal of experienced contributors, not spammers. The effect got stronger with more data.

### H2: Engagement Lifecycle -- AUC 0.481

Inverted. Higher engagement correlates with being merged. Responsive authors get their PRs accepted.

### H3: Cross-Repo Fingerprinting -- AUC 0.503

Essentially random. Cross-repo signal disappeared with better data. The previous 0.512 was noise amplified by the small DuckDB sample.

### H4: Combined Model -- AUC 0.501

Combining H1+H2+H3 produces random discrimination. All nested LRTs significant (p < 1e-4) but effects too small and partially cancelling.

### H5: GE Score Complement -- GE v2 AUC 0.521

GE v2 remained the strongest single predictor. Adding bot signals to GE v2 slightly *hurt* performance (0.521 -> 0.520).

### H6: Interaction Features -- AUC 0.480

Inverted like H1. Burst + no prior merge correlates with active contributors trying new repos, not spammers.

### H7: Burst Content Homogeneity -- AUC 0.479

Inverted. Within-burst content similarity is higher for merged PRs (legitimate refactoring series).

### Baselines (Stage 4)

| Baseline | AUC-ROC | 95% CI |
|---|---|---|
| GE v2 | 0.533 | [0.531, 0.535] |
| GE v1 | 0.512 | [0.511, 0.514] |
| Account age < 30d | 0.501 | [0.501, 0.501] |
| Random | 0.498 | [0.495, 0.501] |

### Post-Mortem: Three Compounding Flaws

1. **Wrong unit of analysis**: Spam is author-level; PR-level evaluation diluted 819 suspicious PRs across 200K rows
2. **Wrong target variable**: "Not merged" is 27% of data but <2% is actual spam; the rest is normal development friction
3. **Feature sparsity**: 59% of authors are single-repo, so all cross-repo features are identically zero

---

## Iteration 5: Author-Level Bot Detection (H8-H13)

### Design Changes

- **Unit of analysis**: Authors, not PRs. 31,296 authors scored holistically.
- **Target variable**: `account_status == 'suspended'` from GitHub API (ground truth). Secondary: `total_repos >= 3 AND merge_rate < 0.30` (heuristic).
- **Ground truth acquisition**: Checked all 3,216 multi-repo authors via GitHub API. Found **61 suspended accounts** (1.9% of checked). Unchecked single-repo authors (28,080) assumed active.
- **Evaluation**: Precision@k and recall@k (appropriate for rare-event detection) alongside AUC-ROC/AUC-PR. 5-fold stratified CV.

### Hypotheses

| ID | Name | Approach |
|----|------|----------|
| H8 | Author Aggregates | Per-author merge rate, rejection rate, PR volume, body/title stats |
| H9 | Time-Series Anomaly | Inter-PR timing, burstiness, dormancy, regularity |
| H10 | Network Analysis | Bipartite author-repo graph, degree centrality, clustering, isolation |
| H11 | LLM Content Analysis | Gemini classifies PR titles as spam-like (score 0-1) |
| H11-tfidf | TF-IDF Title Analysis | Local, deterministic title spam scoring (shortness, lexical poverty, homogeneity, template matching, cross-author commonality) |
| H12 | Campaign Detection | Time-clustered spam in anomalous repo-months |
| H13 | Semi-Supervised | k-NN from suspended seeds + Isolation Forest |

### Results: Multi-Repo Only (61 suspended / 3,208 authors)

These results evaluate only multi-repo authors (2+ repos), the population where all features are expressive.

| Hypothesis | P@10 | P@25 | P@50 | P@100 | AUC-ROC | AUC-PR |
|------------|------|------|------|-------|---------|--------|
| H8 (merge rate) | 0.00 | 0.00 | 0.00 | 0.03 | 0.565 | 0.030 |
| H9 (temporal) | 0.00 | 0.00 | 0.00 | 0.01 | 0.420 | 0.016 |
| H10 (network) | 0.10 | 0.08 | 0.04 | 0.03 | 0.523 | 0.023 |
| **H11 (LLM)** | **0.70** | **0.44** | **0.22** | **0.15** | **0.619** | **0.136** |
| H11-tfidf | 0.10 | 0.16 | 0.14 | 0.09 | 0.595 | 0.056 |
| H13 k-NN | 1.00 | 1.00 | 1.00 | 0.61 | 1.000 | 1.000 |
| H13 IF | 0.00 | 0.00 | 0.00 | 0.01 | 0.391 | 0.016 |
| **Combined** | **0.70** | **0.48** | **0.36** | **0.25** | **0.928** | **0.306** |

### Results: Auxiliary Target (98 suspicious / 31,296 authors)

**Contamination warning**: The auxiliary target is defined as `total_repos >= 3 AND merge_rate < 0.30`. Evaluating H8 (merge rate) or H10 (network, correlated with repo count) against this target is partially circular. These numbers are included for completeness but should not be interpreted as independent validation.

| Hypothesis | P@10 | P@25 | P@50 | P@100 | AUC-ROC | AUC-PR |
|------------|------|------|------|-------|---------|--------|
| H8 (merge rate) | 0.00 | 0.00 | 0.00 | 0.00 | 0.709 | 0.005 |
| H9 (temporal) | 0.00 | 0.00 | 0.02 | 0.01 | 0.736 | 0.016 |
| H10 (network) | 0.20 | 0.16 | 0.14 | 0.14 | 0.990 | 0.138 |
| **H11 (LLM)** | **0.40** | **0.44** | **0.30** | **0.22** | **0.990** | **0.184** |
| H13 k-NN | 0.00 | 0.04 | 0.02 | 0.03 | 0.187 | 0.004 |
| H13 IF | 0.00 | 0.04 | 0.02 | 0.07 | 0.982 | 0.082 |
| **Combined** | **0.60** | **0.48** | **0.36** | **0.23** | **0.994** | **0.278** |

### Campaign Detection (H12)

- **101 anomalous repo-months** flagged (rejection rate > 2 stdev above repo mean)
- **609 campaign authors** identified (only appear during anomalous months, 0% merge rate)
- Hacktoberfest 2019: 94.1% rejection rate (16 authors, 6 repos)
- Hacktoberfest 2020: 70.7% rejection rate (39 authors, 13 repos)
- 16 of 609 campaign authors are suspended (updated from iteration 5b ground truth expansion; the original iteration 5 reported 0 overlap because only 61 suspended accounts were known at that point)

### Analysis (Multi-Repo Population)

#### What worked

1. **The unit-of-analysis pivot was decisive.** Author-level features on the same 200K PR corpus produce AUC 0.619-1.000 against real ground truth. PR-level features on the same data produced AUC 0.479-0.503. Same data, different framing, completely different results.

2. **H11 (LLM content) is the strongest unsupervised signal for multi-repo authors.** P@10 = 0.70 (7 of the top 10 most suspicious authors by LLM score are actually suspended). AUC-ROC of 0.619 is moderate -- many suspended authors have titles that look legitimate in isolation. Cost: ~$0.15 for 820 authors. Note: the LLM population was pre-filtered by `merge_rate < 0.5` (see Lookahead Contamination section), but the LLM itself scores based on title content, not merge rate.

3. **The combined score is practical.** P@25 = 0.48, P@100 = 0.25, AUC 0.928. Rank all 3,208 multi-repo authors by this score, check the top 50, and 18 are confirmed suspended. However, this score inherits contamination from H8 and H13 components. (Combined scores shifted slightly with the addition of H11-tfidf to the score ensemble.)

4. **H13 k-NN is circular but useful.** Perfect AUC against primary target is expected -- seeds are the positives. Its value is in the combined score where it boosts features that correlate with known-bad accounts. Seeds were selected via biased sampling (merge rate ascending), which means the k-NN model may over-weight merge-rate-correlated features.

#### What didn't work

1. **H11-tfidf provides weaker signal than LLM but covers all authors.** AUC 0.595 on multi-repo (vs 0.619 for LLM), AUC 0.571 on single-repo (vs 0.500 degenerate for LLM). The advantage is coverage: it scores all 31,296 authors including single-repo, where LLM gives a constant 0.0. Precision is low (P@10 = 0.10 on multi-repo) but the signal is real and uncontaminated by merge rate.

2. **H8 (merge rate alone) is weak and contaminated.** AUC 0.565, P@k near zero. See Lookahead Contamination section -- the 0.565 includes the data being predicted.

3. **H9 (temporal) is inverted.** AUC 0.420, below chance. Suspended accounts have *less* variable timing than legitimate authors, opposite of the hypothesis. This result is not contaminated by merge rate.

4. **H10 (network) collapsed from 0.952 to 0.523.** The hub_score column is degree centrality (not HITS -- HITS was numerically unstable on bipartite graphs). The earlier result (Iteration 5, 27 seeds) was inflated by seed selection bias: the first 27 suspended accounts were the lowest-merge-rate multi-repo authors, who also happen to have extreme network topology. Checking all 3,216 multi-repo authors added 34 suspended accounts with more typical network profiles. The 0.523 is the honest number. Not directly contaminated by merge rate, but the initial 0.952 was an artifact of biased seed selection.

5. **H13 Isolation Forest detects anomalies but not the right ones.** AUC 0.391 (below chance). It finds statistical outliers, most of which are prolific legitimate contributors.

6. **Campaign detection (H12) finds a different population.** The 609 time-clustered campaign authors have 16 suspended accounts among them (2.6%), roughly matching the background rate.

### Pipeline Details (Multi-Repo)

- Ground truth: All 3,216 multi-repo authors checked via GitHub API (`--min-repos 2`). 61 suspended, 3,155 active. ~107 min.
- Stage 5: Author aggregate features (H8) + bipartite network graph (H10) for 31,296 authors. ~1m45s.
- Stage 6a: Time-series features (H9) for 31,296 authors. ~0.5s.
- Stage 6b: LLM content analysis (H11) for 820 pre-filtered authors (merge_rate < 0.5, repos >= 2) via Gemini 2.0 Flash. 820 cached. 4 parse failures (defaulted to 0.5). ~2s (cache hits).
- Stage 6c: Semi-supervised (H13) k-NN + Isolation Forest. 61 suspended seeds. ~instant.
- Stage 6d: TF-IDF title analysis (H11-tfidf) for all 31,296 authors. Local, deterministic. ~<1 min.
- Stage 7: Author-level evaluation against primary (suspended) and auxiliary (suspicious) targets. Precision@k, AUC-ROC, AUC-PR, Mann-Whitney U. ~4s.
- Stage 8: Campaign detection. 101 anomalous repo-months, 609 campaign authors. ~1s.

---

## Iteration 5b: Single-Repo Population

### Motivation

The multi-repo analysis (Iteration 5) established that author-level features work for the 3,208 authors who contribute to 2+ repos. But 28,088 authors (89.7%) only appear in a single repo. These are a structurally different population: network features are degenerate (every single-repo author has identical hub score), LLM pre-filtering excludes them, and cross-repo behavior doesn't exist. Do the same hypotheses work, or does this population need different approaches?

### Ground Truth Expansion

Checked 9,682 single-repo authors via GitHub API (`--min-repos 1 --limit 10000`), ordered by merge rate ascending. Found **262 suspended accounts** (2.7% of checked). Combined with the 61 multi-repo suspended, total ground truth is now **323 suspended / 12,898 checked**.

The single-repo suspension rate (2.7%) is higher than multi-repo (1.9%). This makes sense: accounts that get suspended often don't survive long enough to contribute to multiple repos.

**Sampling bias**: The merge-rate-ascending ordering means the checked population is enriched for low-merge-rate authors. The true suspension rate among all single-repo authors may be lower than 2.7%. 18,398 single-repo authors remain unchecked (higher merge rates, outside top 10K by suspicion).

### Results: Single-Repo (262 suspended / 28,088 authors)

| Hypothesis | P@10 | P@25 | P@50 | P@100 | AUC-ROC | AUC-PR |
|------------|------|------|------|-------|---------|--------|
| **H8 (merge rate)** | **0.10** | **0.08** | **0.08** | **0.05** | **0.801** | **0.023** |
| H9 (temporal) | 0.00 | 0.00 | 0.00 | 0.00 | 0.342 | 0.005 |
| H10 (network) | 0.00 | 0.04 | 0.02 | 0.02 | 0.500 | 0.009 |
| H11 (LLM) | 0.00 | 0.04 | 0.02 | 0.02 | 0.500 | 0.009 |
| H11-tfidf | 0.00 | 0.04 | 0.06 | 0.04 | 0.571 | 0.015 |
| H13 k-NN | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 | 1.000 |
| H13 IF | 0.00 | 0.00 | 0.00 | 0.00 | 0.542 | 0.010 |
| **Combined** | **0.70** | **0.88** | **0.80** | **0.63** | **0.993** | **0.484** |

### Results: All Authors Combined (323 suspended / 31,296 authors)

| Hypothesis | P@10 | P@25 | P@50 | P@100 | AUC-ROC | AUC-PR |
|------------|------|------|------|-------|---------|--------|
| H8 (merge rate) | 0.00 | 0.00 | 0.02 | 0.01 | 0.760 | 0.023 |
| H9 (temporal) | 0.00 | 0.00 | 0.00 | 0.00 | 0.427 | 0.008 |
| H10 (network) | 0.10 | 0.08 | 0.04 | 0.03 | 0.544 | 0.013 |
| H11 (LLM) | 0.50 | 0.40 | 0.22 | 0.15 | 0.528 | 0.033 |
| H11-tfidf | 0.00 | 0.04 | 0.06 | 0.07 | 0.586 | 0.019 |
| H13 k-NN | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 | 1.000 |
| H13 IF | 0.00 | 0.00 | 0.00 | 0.01 | 0.564 | 0.012 |
| Combined | 0.50 | 0.44 | 0.26 | 0.17 | 0.974 | 0.180 |

### Analysis: Two Populations, Two Detection Profiles

The single-repo and multi-repo populations require fundamentally different detection approaches. Treating them as one population masks the strengths of each.

#### Multi-repo detection profile

- **Best unsupervised signal**: H11 (LLM content), AUC 0.619, P@10 = 0.70
- **Network features**: Informative but weaker than initially thought (AUC 0.523 on full population vs 0.952 on biased seed set)
- **Merge rate**: Weak (AUC 0.565) and contaminated (see below)
- **Combined (without k-NN)**: H11 carries most of the unsupervised signal

#### Single-repo detection profile

- **Best unsupervised signal**: H8 (merge rate), AUC 0.801. But this is the most contaminated number in the study -- see Lookahead Contamination section.
- **Network features**: Degenerate (AUC 0.500 exactly). Every single-repo author has the same hub score by definition.
- **LLM content**: Degenerate (AUC 0.500). Single-repo authors were excluded from LLM pre-filtering (requires repos >= 2), so all get default score 0.0.
- **H11-tfidf**: Not degenerate (AUC 0.571), the only title-based signal available for single-repo authors. Weak but non-trivial.
- **Combined**: Dominated by k-NN (AUC 0.993, P@100 = 0.63). The combined score here is largely the semi-supervised component, though H11-tfidf now contributes some signal where LLM cannot.

#### Why merge rate works for single-repo but not multi-repo

Among single-repo authors, merge rate is binary or near-binary: either all your PRs got merged or none did. A single-repo author with 0% merge rate is a much stronger signal than a multi-repo author with 0% merge rate, because the multi-repo case includes legitimate contributors who got some PRs merged in some repos.

The AUC tells the story: H8 goes from 0.565 (multi-repo, weak) to 0.801 (single-repo, strong). The same feature, measured on different populations, behaves differently. However, both numbers are inflated by lookahead contamination.

#### The k-NN dominance problem

The combined score for single-repo authors is heavily influenced by k-NN: AUC 0.993, AUC-PR = 0.484. k-NN is the strongest discriminator in this population -- merge rate has AUC 0.801 but terrible precision (P@100 = 0.05), while k-NN has both. But k-NN is semi-supervised (it uses the suspended labels as seeds), making the combined score partially circular. The seeds were also selected via merge-rate-biased sampling, compounding the contamination.

Without k-NN, single-repo detection reduces to merge rate and TF-IDF title features -- better than chance but not actionable for ranking.

#### Feature availability gap

The core problem with single-repo authors is feature poverty:

| Feature | Multi-repo | Single-repo |
|---------|-----------|-------------|
| H8 merge rate | Available (weak) | Available (strong but contaminated) |
| H9 temporal | Available (inverted) | Available where 2+ PRs (inverted) |
| H10 network | Available (weak) | Degenerate (constant) |
| H11 LLM | Available (strong) | Not computed (excluded by pre-filter) |
| H11-tfidf | Available | Available (covers all authors) |
| H13 k-NN | Available (circular) | Available (circular) |
| H13 IF | Available (inverted) | Available (weak) |

Two of the six original features are structurally unavailable for single-repo authors: network topology is undefined for a single connection, and LLM content scoring was scoped to multi-repo authors. The TF-IDF title analysis (H11-tfidf) partially addresses this gap by providing local, deterministic title features for all 31,296 authors including single-repo.

### Population Comparison Summary

| Metric | Multi-repo (3,208) | Single-repo (28,088) | All (31,296) |
|--------|-------------------|---------------------|--------------|
| Suspended | 61 (1.9%) | 262 (0.9%*) | 323 (1.0%*) |
| Best unsupervised | H11 AUC 0.619 | H8 AUC 0.801 | H8 AUC 0.760 |
| Combined AUC | 0.928 | 0.993 | 0.974 |
| Combined P@10 | 0.70 | 0.70 | 0.50 |
| Combined P@100 | 0.25 | 0.63 | 0.17 |

*Rate among checked authors. 18,398 single-repo authors remain unchecked (assumed active).

The single-repo combined score looks better, but it's almost entirely k-NN. The multi-repo combined score has more genuine unsupervised signal from H11.

### Pipeline Details (Single-Repo)

- Ground truth: 9,682 single-repo authors checked via GitHub API (`--min-repos 1 --limit 10000`), ordered by merge rate ascending. 262 suspended, 9,420 active. ~5.9 hours.
- Features: Reused existing author_features.parquet (all 31,296 authors already had H8-H13 features from multi-repo round). Re-ran stages 6c (semi-supervised with 323 seeds) and stage 7 (evaluation with population segmentation).
- New stage 7 variant: `run_stage7_by_population()` produces separate results for multi-repo, single-repo, and all populations.

---

## Lookahead Contamination & Future Work

### The Problem

`merge_rate` is computed from ALL of an author's PRs with no temporal windowing (`cache.get_author_aggregate_stats()` uses a simple `GROUP BY author`). This means merge_rate contains the outcome data being predicted. The contamination propagates through multiple stages:

1. **Ground truth sampling bias**: `check_account_status.py` orders authors by merge rate ascending. Authors checked first (and thus most likely to be in the ground truth) are the ones with the lowest merge rates. This creates a correlation between "being checked" and "having low merge rate" that biases the suspended/active distribution.

2. **H8 evaluation is circular**: `merge_rate` is both the feature and (via the auxiliary target `merge_rate < 0.30`) the label. The primary target (account suspension) avoids this specific circularity, but merge_rate still contains future data.

3. **H11 LLM pre-filter**: `stage6_llm_content.py` pre-filters by `merge_rate < 0.5`. This means the LLM evaluation population was selected using a contaminated feature. The LLM itself scores based on title content (not merge rate), so P@10 = 0.70 among multi-repo authors is probably genuine signal -- 7 of the 10 most spammy-titled authors are actually suspended. AUC 0.619 is attenuated by the pre-filter assigning 0.0 to all non-filtered authors.

4. **H13 k-NN uses merge_rate as input**: `stage6_semi_supervised.py` includes `merge_rate` in its feature columns (FEATURE_COLS line 22). k-NN distances incorporate this contaminated feature. Seeds were selected from the biased ground truth sample.

5. **Combined score inherits contamination**: Through H8 and H13 components directly, and through H11's biased population indirectly.

### What Conclusions Survive

1. **The unit-of-analysis pivot is genuine.** Author-level > PR-level is a real finding, independent of merge rate contamination.

2. **H9 (temporal) is not contaminated.** AUC 0.420 (inverted) is a genuine finding -- suspended accounts don't have distinctive timing patterns. This is the cleanest result in the study.

3. **H10 (network) is not directly contaminated.** Hub score (degree centrality from the bipartite graph) does not use merge rate. AUC 0.523 on the full multi-repo population is the honest number. The initial 0.952 was inflated by seed selection bias (lowest merge rate authors checked first = extreme network topology). The auxiliary target AUC of 0.990 is circular (the target is partially defined by repo count, which correlates with hub score).

4. **H11 (LLM) provides real signal on title quality.** The LLM scores based on title content, independent of merge rate. P@10 = 0.70 is genuine. But the population was biased by the merge_rate < 0.5 pre-filter, and single-repo authors got score 0.0.

5. **H11-tfidf is not contaminated.** TF-IDF title features (shortness, lexical poverty, homogeneity, template matching, cross-author commonality) use only title text. AUC 0.595 (multi-repo) and 0.571 (single-repo) are clean numbers. Coverage is universal (all 31,296 authors).

6. **H8 merge rate signal may be real but its magnitude is unknowable.** AUC 0.565 (multi-repo) and 0.801 (single-repo) are upper bounds on the true signal. They could be much lower after removing lookahead.

7. **Campaign detection (H12) is structurally different.** It uses within-month rejection rates, not global merge rates. Moderate contamination (monthly windows are bounded but still include all PRs within the window). The 101 anomalous months and 609 campaign authors are structural findings about Hacktoberfest-era PR patterns.

### What a Clean Experiment Would Look Like

A temporal holdout design:

1. **Time-based train/test split**: Pick a cutoff date (e.g., 2022-01-01). Compute all features using only PRs before the cutoff. Evaluate predictions against PRs after the cutoff.
2. **Compute merge_rate only from pre-cutoff PRs**: `merge_rate = merged_before_cutoff / total_before_cutoff`. This removes the lookahead.
3. **Check ground truth without merge-rate ordering**: Sample authors uniformly (or stratified by repo count) rather than ordering by merge rate ascending.
4. **Remove merge_rate from k-NN features**: Or at least use the temporal version.
5. **Remove the auxiliary target entirely**: It's defined in terms of merge_rate, making it circular for any feature correlated with merge rate.

This experiment would produce lower AUC numbers for H8 and the combined score, but they would be trustworthy.

---

## Iteration 6: Temporal Holdout Experiment

### Motivation

Iterations 5/5b showed author-level bot detection works (AUC 0.619-1.000), but the headline numbers are contaminated by merge rate lookahead. `merge_rate` is computed from all of an author's PRs (no temporal windowing), and it propagates through ground truth sampling order, LLM pre-filtering, k-NN features, and the auxiliary target definition. The "What a Clean Experiment Would Look Like" section above described the fix. This is that fix.

### Design

Six semi-annual global cutoff dates: 2020-01-01, 2021-01-01, 2022-01-01, 2022-07-01, 2023-01-01, 2024-01-01. For each cutoff T, all features are computed using only PRs with `created_at < T`. This answers "if we deployed this system at time T, how well would it work?"

Key decontamination steps:
- **merge_rate**: computed from pre-cutoff PRs only
- **account_age_days**: `cutoff - account_created_at` (not `now - created_at`)
- **Network graph**: built from pre-cutoff author-repo pairs only
- **LLM pre-filter**: uses pre-cutoff merge rate
- **LLM titles**: pre-cutoff titles only
- **TF-IDF titles**: pre-cutoff titles only
- **k-NN H13-clean**: excludes `merge_rate`, `rejection_rate`, `pocket_veto_rate` from features
- **k-NN H13-temporal**: includes pre-cutoff merge_rate in features
- **Auxiliary target**: dropped entirely (circular by definition)
- **Ground truth labels**: reused as-is (suspension is a property of the account, not the time window)

Authors with zero pre-cutoff PRs are excluded from that cutoff's evaluation.

### Per-Cutoff Results (Primary Target: Suspended Accounts)

| Cutoff | Authors | Suspended | PRs | H8 merge | H9 temporal | H10 network | H11 LLM | H11 TF-IDF | H13 IF | Combined |
|--------|---------|-----------|-----|----------|-------------|-------------|---------|-------------|--------|----------|
| 2020-01-01 | 253 | 7 | 405 | 0.537 | 0.304 | 0.497 | 0.497 | 0.444 | 0.707 | 0.989 |
| 2021-01-01 | 402 | 16 | 849 | 0.561 | 0.288 | 0.493 | 0.495 | 0.487 | 0.550 | 0.970 |
| 2022-01-01 | 749 | 26 | 2,471 | 0.677 | 0.414 | 0.508 | 0.515 | 0.532 | 0.536 | 0.976 |
| 2022-07-01 | 1,514 | 43 | 16,619 | 0.785 | 0.388 | 0.500 | 0.519 | 0.471 | 0.452 | 0.972 |
| 2023-01-01 | 2,227 | 68 | 31,634 | 0.809 | 0.342 | 0.500 | 0.524 | 0.509 | 0.447 | 0.967 |
| 2024-01-01 | 4,804 | 147 | 71,046 | 0.787 | 0.434 | 0.513 | 0.517 | 0.533 | 0.474 | 0.950 |
| **Mean±SD** | | | | **0.693±0.110** | **0.362±0.055** | **0.502±0.007** | **0.511±0.011** | **0.496±0.032** | **0.528±0.090** | **0.971±0.011** |

H13 k-NN (both clean and temporal variants) is omitted: AUC = 1.000 at every cutoff because seeds get distance 0 and the evaluation target is those same seeds. This is structural circularity, not a viable signal.

### Comparison to Contaminated Numbers

| Hypothesis | All-time AUC (Iter 5/5b) | Temporal mean AUC (Iter 6) | Verdict |
|---|---|---|---|
| H8 merge_rate | 0.760 | 0.693±0.110 | **Survives.** Drops ~9%, still well above chance. Improves with more data (0.54→0.81). |
| H9 temporal | 0.427 | 0.362±0.055 | **Survives (inverted).** Slightly worse but consistent: suspended accounts are less temporally variable. |
| H10 network | 0.544 | 0.502±0.007 | **Dead.** Drops to chance. The earlier 0.544 was seed selection bias, not signal. |
| H11 LLM | 0.528 | 0.511±0.011 | **Dead.** At chance. The pre-cutoff title set and decontaminated pre-filter eliminate the signal. |
| H11 TF-IDF | 0.586 | 0.496±0.032 | **Dead.** Drops to chance. The all-time TF-IDF had access to post-cutoff titles. |
| H13 k-NN | 1.000 | 1.000 | **Circular** (same issue pre- and post-decontamination). |
| H13 IF | 0.564 | 0.528±0.090 | **Dead.** Noisy, no consistent signal direction. |
| Combined | 0.974 | 0.971±0.011 | **Inflated** by k-NN circularity. Without k-NN, the surviving components would produce ~0.69. |

### Analysis

#### Merge rate is the only real signal

H8 (merge_rate) is the sole hypothesis that survives temporal decontamination with meaningful discrimination. AUC 0.693±0.110 across six cutoffs, with a clear trend: more pre-cutoff data produces better discrimination (0.54 at T1 with 405 PRs → 0.81 at T5 with 31K PRs). The signal is genuine -- authors who will eventually be suspended have measurably lower merge rates from their earliest PRs.

The contaminated all-time number (0.760) was ~9% higher, consistent with lookahead inflation rather than fabrication.

#### Everything else is at chance

H9, H10, H11 (both LLM and TF-IDF), and H13 IF all produce AUC within ~0.01-0.05 of 0.50 across cutoffs. The contaminated numbers (0.42-0.59) were inflated by various forms of information leakage:
- H10's seed selection bias (low-merge-rate authors checked first had extreme network topology)
- H11's all-time title corpus (more titles = better TF-IDF features, even for spam detection)
- H13 IF's contaminated feature inputs

#### The combined score is misleading

Combined AUC 0.971 looks strong but is dominated by k-NN circularity (seeds = labels). The honest combined score from non-circular components would be approximately the H8 AUC (~0.69), since no other component contributes above chance.

#### The k-NN circularity is fundamental, not fixable

H13 k-NN gets AUC 1.000 at every cutoff because the algorithm assigns distance 0 to seeds, and the evaluation labels ARE the seeds. This isn't a temporal issue -- it's a design issue. Evaluating k-NN against the same labels used for seeding requires held-out positive labels, which requires more suspended accounts than the 7-147 available per cutoff. Leave-one-out evaluation of k-NN with so few positives would be extremely noisy.

#### Earlier cutoffs are noisier but not uninformative

T1 (2020-01-01) has only 253 evaluable authors and 7 suspended, making results noisy. By T4 (2022-07-01), with 1,514 authors and 43 suspended, the estimates stabilize. The consistent trend across cutoffs (H8 improving, everything else flat near chance) increases confidence in the findings.

### Implications for Next-Gen Scoring

1. **Merge rate works and is cheap.** Pre-cutoff merge rate is a viable feature for a production scoring model. It just needs temporal windowing, which the current good-egg scorer doesn't do.

2. **Title analysis needs rethinking.** The all-time TF-IDF and LLM scores were artifacts of having more data than available at prediction time. Title-based features might still work with enough pre-cutoff titles, but the effect is too weak to detect with the current ground truth size.

3. **Network features are not useful for this task.** Degree centrality in the bipartite author-repo graph doesn't predict suspension. Suspended authors don't have distinctive network topology.

4. **The ground truth bottleneck is real.** With only 7-147 suspended accounts per cutoff, statistical power is limited. Any future experiment needs either more labeled data or a different evaluation strategy (e.g., precision@k only, no AUC).

### Pipeline Details

- Total runtime: ~4 minutes for all 6 cutoffs
- LLM calls: 2-158 per cutoff (only authors passing pre-cutoff merge_rate < 0.5 filter)
- Output: `data/temporal_holdout/T_{date}/author_features.parquet` and `author_evaluation.json` per cutoff, plus `data/temporal_holdout/aggregated_results.json`
- Original `data/features/author_features.parquet` and `data/results/author_evaluation.json` untouched

## Iteration 7: Merge Rate Non-Monotonicity (stage 10)

Tested whether the merge_rate → suspension relationship is non-monotonic (very low = bot/spam, very high = survivorship artifact, middle = normal). Compared 4 model parameterizations across 6 temporal cutoffs using the same holdout data from Iteration 6.

**Features**: `hub_score` and `merge_rate` only. `account_age_days` was excluded because it's 100% NaN for suspended accounts (their GitHub profiles are unavailable), making it a leaked indicator of the target variable.

### Models Tested

1. **Linear** — logit(hub_score, merge_rate)
2. **Quadratic** — logit(hub_score, merge_rate, merge_rate²)
3. **Two-feature** — logit(hub_score, merge_rate, low_merge_flag) with threshold tuned via inner CV
4. **GBT** — GradientBoostingClassifier(hub_score, merge_rate)

### Results (3 stable cutoffs with ≥30 suspended)

| Cutoff | n_susp | Linear | Quadratic | Two-feature | GBT |
|---|---|---|---|---|---|
| 2022-07-01 | 43 | 0.580 | 0.574 | 0.577 | 0.569 |
| 2023-01-01 | 68 | 0.617 | 0.617 | 0.613 | 0.634 |
| 2024-01-01 | 147 | 0.593 | 0.596 | 0.588 | 0.578 |

The 3 early cutoffs (7-26 suspended) produce AUC < 0.5, indicating insufficient statistical power for any model -- not enough positives for LOO-CV to produce stable predictions.

### Answers

1. **Is the relationship non-monotonic?** No strong evidence. The quadratic merge_rate² coefficient is positive at most cutoffs (penalizes extremes), but the improvement over linear is negligible (< 0.01 AUC difference). Spearman correlation between binned merge_rate and suspension rate is negative (rho ≈ -0.7 to -0.9), confirming a monotonic trend: lower merge_rate → higher suspension probability.

2. **Which parameterization wins?** No consistent winner. Linear wins at 2/6 cutoffs, quadratic at 1, two-feature at 2, GBT at 1. Mean AUC differences are within noise (0.33±0.27 across all cutoffs, 0.59±0.02 for stable cutoffs only). DeLong tests show no significant improvement for any variant over linear (all adjusted p > 0.05).

3. **Fitted coefficients?** merge_rate has a consistently negative coefficient (-0.5 to -0.9), confirming low merge_rate predicts suspension. The quadratic term is positive but small. The low_merge_flag coefficient is positive (0.1-1.4) but inconsistent across cutoffs.

4. **Stable across cutoffs?** No. Rankings shuffle freely across cutoffs. The signal is too weak for any parameterization to dominate.

5. **GBT partial dependence?** merge_rate is the dominant feature (70-96% importance). hub_score contributes 4-30%. The partial dependence curve for merge_rate is monotonically decreasing -- no U-shape detected.

### Implications

The merge_rate → suspension relationship is **monotonic**, not U-shaped. Low merge_rate predicts suspension; high merge_rate does not additionally predict it. The negative weight in v2's logistic regression (-0.7783) captures the correct direction. Non-linear parameterizations don't help because there's no non-linearity to capture.

The modest AUC (0.58-0.63) at larger cutoffs is consistent with Iteration 6's finding (0.69±0.11). The lower values here likely reflect using only 2 features (hub_score + merge_rate) vs Iteration 6's merge_rate-only univariate AUC.

### Pipeline Details

- Runtime: ~65 seconds for all 6 cutoffs
- No DB queries, no LLM calls -- pure computation on existing parquets
- Output: `data/temporal_holdout/merge_rate_experiment.json`
- Code: `stages/stage10_merge_rate_models.py`, CLI: `run-merge-rate-experiment`

---

## Iteration 8: Two-Model Pipeline (stage 11)

### Motivation

Good Egg v2 computes a trust score via logistic regression on graph_score, merge_rate, and account_age. It was trained to predict contributor quality, not suspension. Iterations 6-7 showed merge_rate monotonically predicts suspension (AUC ~0.69 univariate). The question: does adding a dedicated suspension classifier alongside a GE v2-style trust score improve detection? This frames Good Egg as a two-model pipeline -- trust scoring + suspension risk -- and evaluates whether the combination beats either model alone.

### Feature Audit

After Iteration 6's decontamination analysis, 16 features survive as safe (0% NaN for both classes, no leakage, no differential missingness). Leaked features excluded: `account_age_days` (100% NaN for suspended), `followers`/`public_repos` (always 0 for suspended), `knn_distance_to_seed*` (seeds are the positives), `isolation_forest_score` (trained with labels). Differentially missing: `inter_pr_cv`, `burst_episode_count` (80% NaN for suspended vs 66% for active).

Skewed features (`median_additions`, `median_files_changed`, `career_span_days`, `total_prs`) are log-transformed before scaling. All logistic regressions use `class_weight="balanced"`.

### Models

**Baselines:**
1. **merge_rate_only** -- `1 - merge_rate` as suspension score (Iteration 6 reference)
2. **ge_v2_proxy** -- LR on hub_score + merge_rate, negated to produce suspension score (simulates GE v2 without leaked account_age)

**Suspension classifiers:**
3. **susp_lr_small** -- LR on top-5 univariate AUC features: mean_title_length, rejection_rate, merge_rate, career_span_days, hour_entropy
4. **susp_lr_full** -- LR on all 16 safe features
5. **susp_gbt** -- GradientBoostingClassifier on all 16 safe features (n_estimators=100, max_depth=3)

**Combined pipelines** (using best single suspension classifier per cutoff):
6. **linear_combo** -- `alpha * ge_proxy_susp + (1-alpha) * susp_score`, alpha tuned via inner 3-fold CV
7. **stacked** -- second-stage LR on both models' OOF probabilities
8. **product** -- `ge_proxy_susp * susp_score`
9. **max_score** -- `max(ge_proxy_susp, susp_score)`

### Evaluation Protocol

Same CV strategy as Iteration 7: LOO-CV when n_suspended < 30 (cutoffs 2020-2022), 5-fold stratified CV otherwise. DeLong paired tests vs ge_v2_proxy with Holm-Bonferroni correction. Combined models use OOF probabilities from the same CV folds.

### Results

| Cutoff | n_susp | ge_v2_proxy | susp_lr_full | susp_gbt | Best combo | Best combo AUC |
|---|---|---|---|---|---|---|
| 2020-01-01 | 7 | 0.980 | 0.546 | 0.551 | max_score | 0.972 |
| 2021-01-01 | 16 | 0.981 | 0.548 | 0.492 | max_score | 0.820 |
| 2022-01-01 | 26 | 0.839 | 0.610 | 0.629 | max_score | 0.826 |
| 2022-07-01 | 43 | 0.420 | 0.647 | 0.665 | susp_gbt* | 0.665 |
| 2023-01-01 | 68 | 0.383 | 0.637 | 0.683 | susp_gbt* | 0.683 |
| 2024-01-01 | 147 | 0.407 | 0.695 | 0.686 | stacked | 0.695 |

*At these cutoffs, no combined model beats the best single suspension classifier.

**Aggregated across cutoffs (mean ± std AUC-ROC):**

| Model | Mean AUC | Std AUC |
|---|---|---|
| merge_rate_only | 0.571 | 0.051 |
| ge_v2_proxy | 0.668 | 0.269 |
| susp_lr_small | 0.530 | 0.191 |
| susp_lr_full | 0.614 | 0.053 |
| susp_gbt | 0.618 | 0.072 |
| linear_combo | 0.617 | 0.070 |
| stacked | 0.419 | 0.263 |
| product | 0.619 | 0.049 |
| max_score | 0.663 | 0.219 |

### Key Finding: GE v2 Proxy Flips Direction at Later Cutoffs

The most striking result is that ge_v2_proxy's performance is bimodal. At early cutoffs (2020-2021), it produces AUC 0.98 -- near perfect. At later cutoffs (2022-07 onward), it drops below 0.5, meaning the model's trust predictions are *inverted* relative to suspension. The model predicts that suspended accounts are *more trustworthy* than active ones.

This happens because the ge_v2_proxy is `1 - LR(hub_score, merge_rate)`, where the LR was trained with balanced class weights on the cutoff's data. At early cutoffs with very few suspended accounts (7-16), the balanced weighting amplifies a clean separation. At later cutoffs with more suspended accounts (43-147), the model learns a relationship that doesn't generalize across the OOF folds, producing inverted predictions.

The high variance (std = 0.269) confirms this instability. No other model comes close to this level of variance.

### Analysis

**The dedicated suspension classifiers beat ge_v2_proxy at larger cutoffs.** At cutoffs with ≥43 suspended accounts (the 3 stable cutoffs from Iteration 7), susp_gbt achieves 0.57-0.68 AUC while ge_v2_proxy achieves 0.38-0.42. DeLong tests confirm this difference is significant (adjusted p < 0.05 at all 3 stable cutoffs). The suspension classifiers are more stable: susp_gbt has std = 0.072 vs ge_v2_proxy's 0.269.

**No combined model consistently beats the best single model.** At early cutoffs where ge_v2_proxy dominates, max_score preserves most of the signal (0.97 vs 0.98). At later cutoffs where suspension classifiers dominate, the combined models can't improve on the single best. The product combiner (mean AUC 0.619) marginally edges susp_gbt (0.618) but the difference is noise. The stacked combiner (0.419) is the worst overall, unstable at small sample sizes.

**linear_combo alpha values are informative.** At early cutoffs (2020-2021), the tuned alpha is 0.9, heavily weighting ge_v2_proxy. At later cutoffs (2022-07 onward), alpha drops to 0.1, heavily weighting the suspension classifier. The optimizer correctly identifies which sub-model is useful per cutoff, but this doesn't help at test time when you don't know which regime you're in.

**susp_lr_full vs susp_gbt is a wash.** GBT wins at 4 of 6 cutoffs (mean rank 3.8 vs 4.2), lr_full wins at 2 (including the largest cutoff, 2024-01-01 with 147 suspended). Neither dominates.

**Best suspension classifier per cutoff**: susp_gbt wins at 4 cutoffs, susp_lr_full at 2. This is consistent with GBT handling nonlinearities in the 16-feature space slightly better, but with insufficient ground truth to make it reliable.

### Implications for Good Egg

1. **A dedicated suspension classifier with ~16 behavioral features outperforms the GE v2-style trust model at predicting suspension.** The trust model's features (hub_score + merge_rate) were chosen for trust, not suspension -- different targets, different optimal features.

2. **Combining the two models doesn't help.** The trust model's instability (flipping direction across cutoffs) makes it an unreliable component in any combination. The suspension classifier alone is the better choice.

3. **The practical ceiling is AUC ~0.69.** Both susp_lr_full and susp_gbt plateau around 0.69 at the largest cutoff (147 suspended). This is consistent with Iteration 6's merge_rate-only AUC of 0.69±0.11. The additional 15 features beyond merge_rate provide modest lift (0.60→0.69 at the largest cutoff).

4. **Account_age would help if it weren't leaked.** The feature audit excluded account_age_days because it's 100% NaN for suspended accounts (their profiles are unavailable). In a production system where you're scoring active accounts before they get suspended, account_age would be available and likely useful.

### Pipeline Details

- Runtime: ~90 seconds for all 6 cutoffs (dominated by LOO-CV at early cutoffs)
- Output: `data/temporal_holdout/two_model_pipeline.json`
- Code: `stages/stage11_two_model_pipeline.py`, CLI: `run-two-model-pipeline`

---

## Iteration 9: k-NN Holdout Experiment (stage 12)

### Motivation

Iteration 6 dismissed k-NN (H13) as "structurally circular, not fixable" because the suspended accounts used as seeds ARE the evaluation labels, giving seeds distance 0 and AUC 1.000 at every cutoff. But this is fixable: split the suspended accounts into seed and evaluation sets via cross-validation folds, then test whether k-NN distance from the remaining seeds can identify the held-out suspended accounts among the active population.

Secondary question: does k-NN distance to suspended seeds (computed WITHOUT merge_rate) predict PR merge rates among active accounts? This tests whether "behavioral proximity to bots" generalizes as a quality signal beyond the circular suspension prediction.

### Design

**Experiment A: Suspension prediction with held-out seeds.** Per cutoff, fold the suspended accounts only into CV splits. For each fold, the non-held-out suspended accounts are seeds, and the held-out suspended + all active accounts are the eval set. Scaler fit on seeds + active (excluding held-out). LOO-CV when n_suspended < 30, 5-fold otherwise. Active accounts appear in every fold's eval set; their score is averaged across folds.

Three k-NN variants tested:
- **knn_safe16_euclidean**: 16 safe features, euclidean distance, k=5
- **knn_safe15_no_mr**: 15 features (no merge_rate), euclidean, k=5
- **knn_safe16_cosine**: 16 safe features, cosine distance, k=5

Plus a k sweep on the best variant: k ∈ {3, 5, 10, 15}.

Baselines: `merge_rate_only` (1 - merge_rate, no CV) and `susp_lr_full` (LR on 16 features, same CV protocol as Iteration 8). Note: all models are evaluated on the same population (active + suspended accounts only), which differs from Iteration 6's evaluation that included unchecked authors as negatives. This is why merge_rate_only AUCs here (0.571 mean) are lower than Iteration 6's H8 AUCs (0.693 mean).

**Experiment B: Quality prediction among active accounts.** Using ALL suspended accounts as seeds (no splitting -- different target population, different target variable, no circularity), test whether k-NN distance predicts merge_rate among active accounts. Uses 15 features (no merge_rate) to avoid predicting merge_rate from itself.

### Results: Experiment A (Suspension Prediction)

| Cutoff | n_susp | knn_16_euc | knn_15_no_mr | knn_16_cos | merge_rate | susp_lr_full |
|---|---|---|---|---|---|---|
| 2020-01-01 | 7 | 0.189 | 0.187 | 0.563 | 0.508 | 0.546 |
| 2021-01-01 | 16 | 0.392 | 0.401 | 0.472 | 0.500 | 0.548 |
| 2022-01-01 | 26 | 0.432 | 0.417 | 0.489 | 0.566 | 0.610 |
| 2022-07-01 | 43 | 0.529 | 0.513 | 0.582 | 0.620 | 0.647 |
| 2023-01-01 | 68 | 0.543 | 0.515 | 0.589 | 0.631 | 0.637 |
| 2024-01-01 | 147 | 0.585 | 0.567 | 0.613 | 0.599 | 0.695 |
| **Mean±SD** | | **0.445±0.132** | **0.433±0.124** | **0.551±0.053** | **0.571±0.051** | **0.614±0.053** |

**Stable cutoffs only (n_suspended ≥ 30):**

| Model | Mean AUC | Std AUC |
|---|---|---|
| knn_safe16_euclidean | 0.552 | 0.024 |
| knn_safe15_no_mr | 0.532 | 0.025 |
| knn_safe16_cosine | 0.595 | 0.013 |
| merge_rate_only | 0.617 | 0.013 |
| susp_lr_full | 0.660 | 0.026 |

**k sweep (best variant: knn_safe16_cosine):**

Higher k generally helps slightly. At the largest cutoff (147 suspended), k=15 achieves AUC 0.630 vs k=5's 0.613. Early cutoffs can't test large k (seed set too small).

**Ranking consistency:** susp_lr_full wins 5/6 cutoffs (mean rank 1.2). merge_rate_only ranks 2.3 on average. knn_safe16_cosine ranks 2.5. Euclidean variants rank 4-5.

### Results: Experiment B (Quality Prediction)

| Metric | Mean | Std |
|---|---|---|
| Spearman rho (knn_score vs merge_rate) | -0.272 | 0.051 |
| Binary AUC (merge_rate < 0.3) | 0.756 | 0.066 |

Quartile analysis (by k-NN suspiciousness score, averaged across cutoffs):

| Quartile | Mean merge_rate | Interpretation |
|---|---|---|
| Q1 (most suspicious) | 0.291 | Closest to bots → lowest merge rate |
| Q2 | 0.183 | |
| Q3 | 0.077 | |
| Q4 (least suspicious) | 0.102 | Farthest from bots → higher merge rate |

The Q3/Q4 non-monotonicity is likely noise from the score distribution's tail. The overall trend is clear: proximity to suspended accounts in feature space predicts lower merge rates among active accounts.

Note: `rejection_rate` remains in the 15-feature set and is mechanically correlated with merge_rate (~complementary). This is acceptable because rejection patterns are substantively meaningful beyond merge rate, but the Spearman correlation includes this indirect path.

### Analysis

**k-NN circularity is now broken.** All AUCs are well below 1.0, confirming the held-out CV design works. The previous AUC = 1.000 was entirely an artifact of seeds getting distance 0.

**k-NN is a weak suspension predictor.** The best variant (cosine, 16 features) achieves 0.595 mean AUC on stable cutoffs -- above chance but worse than susp_lr_full (0.660). knn_cosine is competitive with merge_rate_only (0.617 vs 0.595 on stable cutoffs, and cosine beats merge_rate at the largest cutoff 0.613 vs 0.599). Euclidean distance performs particularly poorly (0.552), likely because feature magnitudes dominate over direction in high-dimensional space.

**Cosine >> Euclidean for k-NN.** Cosine distance is consistently better (0.595 vs 0.552 on stable cutoffs). This makes sense: suspended accounts differ from active ones in the *direction* of their feature profiles (behavioral shape), not necessarily in magnitude.

**Removing merge_rate hurts.** knn_safe15_no_mr (0.532) is worse than knn_safe16_euclidean (0.552) and knn_safe16_cosine (0.595). merge_rate carries real signal even in a k-NN context, consistent with Iteration 6's finding.

**k-NN does predict merge quality (Experiment B).** Among active accounts, proximity to suspended seeds (using 15 features, no merge_rate) predicts merge rate with Spearman rho = -0.272 and binary AUC = 0.756 for low-quality authors (merge_rate < 0.3). This suggests behavioral similarity to bots generalizes as a quality indicator. Caveat: `rejection_rate` is in the 15-feature set and is mechanically complementary to merge_rate, so some of this correlation may flow through that proxy rather than representing independent signal.

**The practical ceiling hasn't moved.** susp_lr_full achieves 0.660 mean AUC on stable cutoffs (range 0.637-0.695), consistent with Iteration 8's results. The 16 behavioral features contain limited information about suspension. More labeled data or richer features (e.g., account_age for active accounts) might help.

### Implications

1. **k-NN is not competitive for suspension prediction.** LR on the same features consistently outperforms it. The non-parametric distance approach doesn't capture the suspension boundary as well as a linear decision surface with balanced class weights.

2. **k-NN distance shows promise as a quality signal, with caveats.** Experiment B's Spearman rho of -0.272 and binary AUC of 0.756 suggest "distance from known bad actors" predicts merge quality among active accounts. However, `rejection_rate` in the feature set acts as a merge_rate proxy, inflating the correlation. A cleaner test would exclude both merge_rate and rejection_rate from the feature set.

3. **The Iteration 6 dismissal was partially wrong.** k-NN circularity was fixable. The honest AUC is ~0.55-0.60 (not 1.0 and not useless). But the corrected result confirms that k-NN adds little beyond what LR already captures.

### Pipeline Details

- Runtime: ~5 seconds for all 6 cutoffs
- Output: `data/temporal_holdout/knn_holdout_experiment.json`
- Code: `stages/stage12_knn_holdout.py`, CLI: `run-knn-holdout`

---

## Iteration 10: Four Experiments for Good Egg Model Development (stages 13-14)

### Motivation

Iterations 5-9 studied bot detection via suspension prediction. The key findings: merge_rate is the only feature surviving temporal decontamination (AUC 0.66-0.70 via LR on 16 features), k-NN is weak for suspension but shows quality-prediction signal, and no individual feature beyond merge_rate predicts suspension above chance. But these experiments targeted suspension -- not the trust/merge prediction that Good Egg actually computes. This iteration runs four experiments connecting research findings to product decisions.

### Experiment A: Merge Prediction

**Question:** Which author-level features predict whether a user's *future* PRs will be merged?

**Design:** For each temporal cutoff T, use pre-cutoff features to predict post-cutoff merge rate >= 0.5 (binary target). Post-cutoff merge rates computed from DuckDB `prs` table. Authors need >=1 pre-cutoff PR (from the parquet) and >=1 post-cutoff PR.

**Models:**
1. `merge_rate_only` -- pre-cutoff merge_rate as univariate score (v2 reference)
2. `ge_v2_proxy` -- LR(hub_score, merge_rate) simulating v2
3. `lr_full` -- LR on all 16 safe features
4. `lr_full_no_mr` -- LR on 15 features (no merge_rate) to measure marginal signal
5. `knn_cosine` -- k-NN cosine (k=5) with high-merge authors as seeds

**Results:**

| Model | Mean AUC | Std AUC | Mean Spearman rho |
|---|---|---|---|
| merge_rate_only | 0.576 | 0.023 | 0.121 |
| ge_v2_proxy | 0.542 | 0.070 | 0.069 |
| lr_full | 0.598 | 0.075 | 0.010 |
| lr_full_no_mr | 0.571 | 0.069 | -0.019 |
| knn_cosine | 0.467 | 0.023 | 0.006 |

Population sizes: 62 (T_2020) to 2,154 (T_2024) authors with post-cutoff PRs.

**DeLong tests (T_2024, largest cutoff):** lr_full significantly beats both merge_rate_only (adj_p < 0.001) and ge_v2_proxy (adj_p < 0.001). ge_v2_proxy does not significantly differ from merge_rate_only (adj_p = 0.14). knn_cosine is significantly *worse* than both baselines.

**Key finding:** Pre-cutoff merge_rate is the strongest single predictor of future merge outcomes (AUC 0.576, rho 0.121). The 16-feature LR (0.598) provides modest improvement. Removing merge_rate drops AUC from 0.598 to 0.571 -- the other 15 features have some marginal signal. The ge_v2_proxy (hub_score + merge_rate) underperforms merge_rate alone, suggesting hub_score adds noise for merge prediction. k-NN is poor for merge prediction -- the label is not as clustered in feature space as suspension is.

### Experiment B: Advisory Suspension Score

**Question:** Can the 16-feature LR suspension classifier produce a useful advisory score for CI?

**Design:** Three model variants with decreasing feature requirements:
- `susp_lr_full` -- 16 features (requires extra API calls)
- `susp_lr_available` -- 10 features (available from current graph build without extra API calls)
- `susp_lr_cheap` -- 6 features (derivable from current v2 data)

Evaluation: precision at fixed FPR thresholds, calibration, advisory tier precision.

**Results (mean across 6 cutoffs):**

| Model | Mean AUC | Std AUC | Brier | P@1%FPR | P@5%FPR | P@10%FPR |
|---|---|---|---|---|---|---|
| susp_lr_full | 0.614 | 0.053 | 0.198 | 0.124 | 0.086 | 0.074 |
| susp_lr_available | 0.608 | 0.053 | 0.207 | 0.170 | 0.118 | 0.076 |
| susp_lr_cheap | 0.538 | 0.193 | 0.234 | 0.057 | 0.038 | 0.053 |

**Advisory tier precision:**

| Model | HIGH (top 1%) | ELEVATED (next 4%) | NORMAL |
|---|---|---|---|
| susp_lr_full | 0.141 | 0.076 | 0.029 |
| susp_lr_available | 0.206 | 0.094 | 0.027 |
| susp_lr_cheap | 0.061 | 0.032 | 0.032 |

**DeLong tests:** No significant differences between susp_lr_available and susp_lr_full at most cutoffs (p > 0.05 at 4/6 cutoffs). The 10-feature available model is statistically indistinguishable from the full 16-feature model for practical purposes.

**Key finding:** The available-features model (10 features, zero extra API calls) performs comparably to the full model (AUC 0.608 vs 0.614). The HIGH tier achieves 14-21% precision (vs base rate ~3%), a 5-7x lift. The cheap model (6 features) is unstable (std 0.193) and performs poorly. An advisory tier using 10 available features is viable, but the absolute precision is modest -- roughly 1 in 5-7 "HIGH RISK" authors is actually suspended.

### Experiment C: k-NN Bot Proximity as Feature

**Question:** Does k-NN distance to suspended seeds improve merge prediction when added to ge_v2_proxy?

**Design:** Compute `bot_proximity_score` (cosine distance to k=5 nearest suspended accounts using 15 features, no merge_rate) for each author. Compare ge_v2_proxy alone vs ge_v2_proxy + bot_proximity in LR for merge prediction.

**Results (per cutoff):**

| Cutoff | n_seeds | Base AUC | +BotProx AUC | Delta | DeLong p |
|---|---|---|---|---|---|
| 2020-01-01 | 7 | 0.463 | 0.582 | +0.119 | 0.243 |
| 2021-01-01 | 16 | 0.431 | 0.567 | +0.136 | 0.056 |
| 2022-01-01 | 26 | 0.578 | 0.689 | +0.110 | 0.001 |
| 2022-07-01 | 43 | 0.620 | 0.699 | +0.079 | <0.001 |
| 2023-01-01 | 68 | 0.567 | 0.638 | +0.071 | <0.001 |
| 2024-01-01 | 147 | 0.592 | 0.625 | +0.033 | 0.001 |
| **Mean** | | **0.542** | **0.633** | **+0.092** | |

**Key finding:** Bot proximity consistently improves merge prediction (mean delta +0.092 AUC). The effect is significant at 4/6 cutoffs (DeLong p < 0.05). The feature encodes "behavioral distance from known-bad accounts" and carries signal independent of hub_score and merge_rate. At the largest cutoff (147 seeds), the delta narrows to +0.033 -- more seeds may paradoxically dilute the signal if the suspended population is heterogeneous.

This is the strongest evidence yet that a shipped seed file of suspended account feature vectors could improve Good Egg predictions. Runtime overhead would be negligible (<10ms per author for 147-seed brute-force k-NN in 15 dimensions).

### Experiment D: Temporal Windowing of Merge Rate

**Question:** Does a short-term lookback merge rate predict future merges better than all-time?

**Design:** Five merge_rate variants: alltime, 1yr, 6mo, 3mo, exponentially weighted (half-life 180d). Each tested as univariate predictor and in v2-style LR(hub_score, mr_variant). Population filter: >=2 PRs in the lookback window.

**Results (aggregated, cutoffs with sufficient data):**

| Variant | Uni AUC | V2 LR AUC | Spearman rho | n_cutoffs |
|---|---|---|---|---|
| mr_alltime | 0.542 | 0.415 | 0.087 | 6 |
| mr_1yr | 0.611 | 0.507 | 0.199 | 6 |
| mr_6mo | 0.618 | 0.503 | 0.234 | 6 |
| mr_3mo | 0.675 | 0.647 | 0.211 | 4 |
| mr_weighted | 0.577 | 0.545 | 0.122 | 6 |

Note: mr_3mo only has 4 cutoffs because early cutoffs have <10 eligible authors in the 3-month window. The v2 LR AUC is lower than univariate for some variants because hub_score adds noise for merge prediction (consistent with Experiment A).

**Best window per-cutoff (stable cutoffs only, n>=100):**

| Cutoff | mr_alltime | mr_1yr | mr_6mo | mr_3mo | mr_weighted |
|---|---|---|---|---|---|
| 2022-01-01 | 0.512 | 0.600 | 0.529 | **0.657** | 0.561 |
| 2022-07-01 | **0.711** | 0.663 | 0.624 | 0.737 | 0.660 |
| 2023-01-01 | 0.615 | **0.659** | 0.658 | 0.645 | 0.645 |
| 2024-01-01 | 0.630 | 0.590 | **0.697** | 0.659 | 0.606 |

**Key finding:** Short-term merge rates (3mo, 6mo) outperform all-time merge rate for predicting future merges. The 3-month window achieves the highest univariate AUC (0.675) when data is available. The exponentially weighted rate (half-life 180d) performs only marginally better than all-time (0.577 vs 0.542), suggesting the half-life is too long to capture recency signal effectively. Good Egg's current all-time merge rate is suboptimal; a 3-6 month lookback window would likely improve predictions.

### Implications for Good Egg

1. **Merge rate is the strongest single feature for both suspension and merge prediction.** The 16-feature LR provides modest improvement (AUC 0.598 vs 0.576), but the marginal features aren't worth the API cost for merge prediction.

2. **Bot proximity is worth shipping.** A static seed file (~50KB JSON) of suspended account feature vectors, plus ~30 lines in scorer.py to compute k-NN distance, adds meaningful signal to merge prediction (+0.09 AUC). The runtime cost is negligible.

3. **Short-term merge rate beats all-time.** Switching from all-time to 3-6 month lookback could improve merge prediction by 0.05-0.10 AUC. This is a config change, not a code change -- v2's recency_config already supports weighting.

4. **Advisory suspension tiers are viable but limited.** The 10-feature model (zero extra API calls) achieves 14-21% precision in the HIGH tier (5-7x lift over base rate). Useful as an informational signal, not reliable enough for blocking.

5. **hub_score adds noise for merge prediction.** ge_v2_proxy (hub_score + merge_rate) underperforms merge_rate alone. This suggests the current v2 scoring formula may be overweighting graph structure for merge prediction.

### Pipeline Details

- Stage 13 runtime: ~3 seconds for all 6 cutoffs (A/C/D experiments)
- Stage 14 runtime: ~7 seconds for all 6 cutoffs (Experiment B)
- Outputs: `data/temporal_holdout/merge_prediction_experiment.json`, `data/temporal_holdout/advisory_score_experiment.json`
- Code: `stages/stage13_merge_prediction.py`, `stages/stage14_advisory_score.py`
- CLI: `run-merge-prediction`, `run-advisory-score`

---

## Decisions and Open Questions (Post-Iteration 10)

### Decisions made

**1. k-NN bot proximity will not ship in the open-source Good Egg.**
The feature requires maintaining a non-stale seed file of suspended account feature vectors. Seeds go stale as GitHub suspends new accounts and the population shifts. Keeping seeds current requires periodic re-scraping of account status -- a hosted service concern, not something an open-source GitHub Action can do autonomously. This is a feature for a commercial/hosted version of Good Egg, not the OSS release.

**2. Merge rate lookback window will be shortened in Good Egg v3.**
Experiment D showed 3-month merge rate (AUC 0.675) substantially outperforms all-time (0.542) for predicting future merges. v2's `recency_config` already supports exponential decay weighting, but the current half-life (180 days) is too long -- the weighted rate (AUC 0.577) barely beat all-time. v3 will shorten the effective lookback. The exact window (3mo vs 6mo) is a tuning question; 3mo has higher AUC but drops more authors below the minimum PR threshold.

**3. "Bad Egg" suspension advisory score will be added to the Good Egg repo as a separate score type.**
Based on the 10-feature `susp_lr_available` model (AUC 0.608, zero extra API calls). Ships as an advisory field in TrustScore output, not affecting trust level classification. The HIGH tier (top 1%) achieves 14-21% precision (5-7x lift over base rate) -- informational, not blocking.

### Open question: feature selection for Bad Egg

We have not done a proper ablation on the 10-feature set. The coefficient stability analysis from Iteration 8 raises concerns:

| Feature | In 10-feat model | LR coeff sign stable | GBT importance (T_2024) |
|---|---|---|---|
| merge_rate | yes | NO | 0.024 |
| total_prs | yes | NO | 0.014 |
| career_span_days | yes | NO | 0.071 |
| mean_title_length | yes | NO | 0.237 |
| hub_score | yes | NO | 0.000 |
| bipartite_clustering | yes | NO | 0.087 |
| isolation_score | yes | NO | 0.000 |
| total_repos | yes | NO | 0.000 |
| median_additions | yes | yes | 0.162 |
| median_files_changed | yes | yes | 0.103 |

Only 2 of 10 features (median_additions, median_files_changed) have sign-stable LR coefficients across cutoffs. Three features (hub_score, isolation_score, total_repos) have zero GBT importance at the largest cutoff. The 6-feature "cheap" model was unstable (AUC std 0.193), but a *different* subset of 6-8 features -- chosen by ablation rather than API-cost convenience -- might perform better.

**Needed experiment:** Forward feature selection or leave-one-out ablation on the 10-feature set, evaluated on the 3 stable cutoffs (n_suspended >= 43). Target: find the minimal feature subset that doesn't significantly degrade AUC vs the full 10. This would tell us whether hub_score, isolation_score, and total_repos can be dropped (simplifying the scoring pipeline) or whether they contribute through interactions despite zero univariate importance.

### Open question: hub_score in Good Egg v3

Experiment A showed hub_score adds noise for merge prediction -- ge_v2_proxy (hub_score + merge_rate) scored AUC 0.542 vs merge_rate_only at 0.576. But hub_score was designed for a different question: "is this person an established contributor to the open-source ecosystem?" rather than "will their future PRs get merged?" These may not be the same thing.

**Proposed experiment (Iteration 11):** Test hub_score's value in a *repo-specific* context rather than the cross-repo merge prediction tested here.

Design: For each author with post-cutoff PRs in a specific repo R, predict whether their PRs to R will be merged. Features:
- `hub_score` (global graph centrality)
- `repo_specific_score` -- hub_score contribution from repo R's neighborhood only
- `has_prior_merged_in_R` -- binary: did the author have a merged PR in R pre-cutoff?
- `merge_rate` (global)
- `merge_rate_in_R` (repo-specific, if available)

The hypothesis is that hub_score helps when the question is "does this person belong in this project's ecosystem?" (the original Good Egg question) rather than "will PRs get merged somewhere?" (what Iteration 10 tested). If hub_score helps for repo-specific prediction but not cross-repo, then v3 should keep it but reweight it. If it doesn't help even repo-specifically, it should be downweighted or removed.

This experiment requires restructuring the target from per-author to per-author-per-repo, which is a different evaluation framework than the temporal holdout parquets currently support. Estimated scope: new stage15, querying per-repo merge outcomes from DuckDB.

## Iteration 11: Feature Ablation + Hub Score Repo-Specific

### Experiment 1: Bad Egg Feature Ablation (stage15)

**Question:** What is the minimal subset of the 10 available features that doesn't significantly degrade suspension prediction AUC?

**Method:** Two complementary analyses across 3 stable cutoffs (T_2022-07, T_2023-01, T_2024-01), each with 5-fold stratified CV and balanced LR:
1. **Leave-one-out ablation** -- drop each feature, DeLong test vs full model, Holm-Bonferroni correction across 10 tests
2. **Forward selection** -- greedy add best feature, DeLong test k-feature vs (k-1)-feature, stop when p > 0.05

**LOO ablation results (mean across 3 cutoffs):**

| Feature | Mean AUC Delta | Dispensable? |
|---|---|---|
| merge_rate | -0.040 | No (2/3) |
| isolation_score | -0.008 | No (1/3) |
| career_span_days | -0.003 | Yes (3/3) |
| median_additions | -0.002 | Yes (3/3) |
| mean_title_length | -0.001 | Yes (3/3) |
| hub_score | +0.000 | Yes (3/3) |
| total_repos | +0.000 | Yes (3/3) |
| total_prs | +0.001 | Yes (3/3) |
| bipartite_clustering | +0.001 | Yes (3/3) |
| median_files_changed | +0.006 | Yes (3/3) |

Only `merge_rate` and `isolation_score` are non-dispensable. 8 of 10 features can be dropped without significant AUC loss. `median_files_changed` actually *improves* AUC when removed (+0.006), suggesting it adds noise.

**Forward selection results (mean rank across 3 cutoffs):**

| Feature | Mean Rank | Selected in >0 cutoffs? |
|---|---|---|
| merge_rate | 1.3 | 3/3 |
| isolation_score | 4.0 | 1/3 |
| median_additions | 4.7 | 0/3 |
| hub_score | 4.7 | 0/3 |
| career_span_days | 5.7 | 0/3 |
| total_repos | 5.7 | 0/3 |
| total_prs | 6.0 | 0/3 |
| mean_title_length | 6.7 | 1/3 |
| bipartite_clustering | 7.3 | 0/3 |
| median_files_changed | 9.0 | 0/3 |

Forward selection consistently picks `merge_rate` first (rank 1.3), then `median_additions` or `isolation_score` as the second and third features. The formal stopping rule (p > 0.05 at each step) halts at k=1 in 2/3 cutoffs because the k=1→k=2 step has p=0.078 and p=0.113. But see below for why this is misleadingly conservative.

**The top-3 forward-selected model beats the full 10-feature model in every cutoff:**

| Cutoff | k=1 (mr) | k=2 (+med_adds) | k=3 (+isolation) | Full 10 | top3 - full |
|---|---|---|---|---|---|
| T_2022-07 | 0.598 | 0.672 | 0.682 | 0.641 | +0.042 |
| T_2023-01 | 0.617 | 0.672 | 0.671 | 0.646 | +0.026 |
| T_2024-01 | 0.613 | 0.668 | 0.680 | 0.678 | +0.002 |

The 7 noise features actively degrade performance -- the 3-feature model is both simpler and higher-AUC than the 10-feature model. The AUC jump from k=1 to k=2 is +0.055 to +0.074 across cutoffs, a large effect that fails the per-step significance threshold only because of low power (43-147 suspended accounts with Holm-Bonferroni across 10 tests). At the largest cutoff (T_2024, 147 suspended), the 3-feature model {mean_title_length, merge_rate, isolation_score} was formally selected with both steps significant.

**Recommended Bad Egg feature set: `merge_rate`, `median_additions`, `isolation_score`.**

These three features have clear mechanistic interpretations for suspension detection:
- **`merge_rate`** -- fraction of PRs that get merged. Suspended accounts have lower merge rates. The strongest single predictor, consistently ranked first in forward selection.
- **`isolation_score`** -- fraction of an author's repos where no other multi-repo author contributes. Suspended accounts tend to work in repos that no established contributor touches. Non-dispensable in 2/3 cutoffs in LOO ablation.
- **`median_additions`** -- median lines added per PR (log-transformed). Captures whether PR sizes are typical or anomalous. Consistently ranked 2nd in forward selection despite not being formally non-dispensable (a power issue -- the LOO test with 10 simultaneous corrections requires very large effects).

All three features are already computed from the same GraphQL data GE fetches, so there is no additional API cost.

### Experiment 2: Hub Score Repo-Specific (stage16)

**Question:** Does hub_score improve merge prediction when the target is repo-specific ("will this author's PRs to repo R be merged?")?

**Method:** For each cutoff and qualifying repo R (>=20 authors with post-cutoff PRs), predict binary target (post-cutoff merge rate in R >= 0.5). 7 model variants compare global vs repo-specific features, with and without hub_score. Pooled across repos, 5-fold stratified CV. DeLong tests with Holm-Bonferroni correction.

**Per-cutoff results (key models):**

| Cutoff | n pairs | mr_only | mr_repo | mr+hub | full_repo | full_combined |
|---|---|---|---|---|---|---|
| T_2020 | 22 | 0.494 | 0.588 | 0.118 | 0.282 | 0.306 |
| T_2021 | 51 | 0.528 | 0.576 | 0.253 | 0.372 | 0.324 |
| T_2022 | 271 | 0.623 | 0.630 | 0.597 | 0.653 | 0.662 |
| T_2022-07 | 775 | 0.630 | 0.640 | 0.615 | 0.705 | 0.698 |
| T_2023 | 968 | 0.604 | 0.600 | 0.599 | 0.670 | 0.661 |
| T_2024 | 1304 | 0.622 | 0.622 | 0.653 | 0.679 | 0.692 |

**Aggregated means:** mr_only 0.583, mr_repo 0.609, mr_plus_hub 0.472, full_repo 0.560, full_combined 0.557.

**DeLong significance:**
- `mr_plus_hub vs mr_only`: hub_score *hurts* in 3/6 cutoffs (significantly), helps in 1/6 (T_2024 only)
- `mr_repo_plus_hub vs mr_repo`: same pattern -- hub_score hurts in early cutoffs, neutral or helps late
- `full_combined vs full_repo`: never significant (0/6) -- global features add nothing on top of repo-specific
- `full_repo vs mr_only`: significant in 4/6 cutoffs -- repo-specific features consistently beat global MR

**Key findings:**
1. **Repo-specific merge rate beats global.** `mr_repo` (0.609) > `mr_only` (0.583), and `full_repo` significantly beats `mr_only` in 4/6 cutoffs. Knowing someone's history *in this specific repo* is more informative than their global track record.
2. **Hub score is unreliable for repo-specific prediction.** It catastrophically hurts with small samples (AUC < 0.3 for n < 100), is neutral for medium samples, and only helps at the largest cutoff (T_2024, n=1304, delta +0.031). The mean across cutoffs is negative.
3. **Global features don't add to repo-specific ones.** `full_combined` ≈ `full_repo` across all cutoffs; DeLong test never significant.
4. **The `full_repo` model (merge_rate_in_R, has_prior_merged_in_R, n_prior_prs_in_R) is the best stable model**, beating global MR significantly in most cutoffs.

**Implication for GE v3:** hub_score should not be upweighted for repo-specific trust scoring. The strongest predictors are repo-specific: does this author have prior merged PRs in this repo, and what's their merge rate here? This aligns with the existing GE `skip_known_contributors` feature -- authors with prior merged PRs in the target repo are already trusted. For unknown authors, global merge_rate is the best single predictor; hub_score adds noise.

However, stage16 pooled all author-repo pairs including known contributors. Since GE's `skip_known_contributors` (default true) fast-tracks authors with prior merged PRs, the scoring model only runs on authors *unknown to the target repo*. Stage17 tests hub_score specifically on that population.

### Experiment 3: Hub Score for Unknown Contributors (stage17)

**Question:** For authors with zero merged PRs in repo R (the population GE actually scores), does hub_score improve merge prediction?

**Method:** Same temporal holdout framework as stage16, but filtered to author-repo pairs where `has_prior_merged_in_R = 0`. Only repos with >=100 pre-cutoff PRs (medium+). Results stratified by repo size: medium (100-499 PRs), large (500-1999), XL (2000+). 4 models compared: `mr_only`, `mr_hub` (+ hub_score), `mr_repos` (+ total_repos), `mr_hub_repos` (+ both).

**Per-cutoff results (all medium+ repos pooled):**

| Cutoff | n pairs | mr_only | mr+hub | mr+repos | hub sig? |
|---|---|---|---|---|---|
| T_2020 | 23 | 0.451 | 0.000 | 0.000 | Yes (hurts) |
| T_2021 | 49 | 0.467 | 0.283 | 0.283 | No |
| T_2022 | 179 | 0.525 | 0.517 | 0.517 | No |
| T_2022-07 | 476 | 0.572 | 0.537 | 0.537 | Yes (hurts) |
| T_2023 | 535 | 0.518 | 0.530 | 0.530 | No |
| T_2024 | 1096 | 0.565 | 0.578 | 0.579 | No |

**Aggregated by repo size tier:**

| Tier | mr_only | mr+hub | mr+repos | Hub delta |
|---|---|---|---|---|
| All (medium+) | **0.516** | 0.408 | 0.408 | -0.108 |
| Medium (100-499) | **0.538** | 0.394 | 0.392 | -0.144 |
| Large (500-1999) | **0.553** | 0.484 | 0.486 | -0.069 |
| XL (2000+) | **0.533** | 0.405 | 0.405 | -0.128 |

`mr_only` wins in every tier. Adding hub_score or total_repos degrades AUC by 0.07-0.14 on average. The pattern holds for large and XL repos -- this is not a small-sample artifact.

**DeLong significance:** `mr_hub vs mr_only` is significant in 2/6 cutoffs (both times hub_score *hurts*). It never significantly helps. `mr_hub vs mr_repos` is never significant (0/6) -- hub_score and total_repos carry identical information and are interchangeable.

**Why hub_score hurts for unknown contributors:** Hub_score (degree centrality in the author-repo bipartite graph) measures how many repos an author has contributed to and how connected those repos are. For unknown-to-repo authors, this measures ecosystem breadth -- "has this person worked on many projects?" But ecosystem breadth is only weakly correlated with whether someone's PRs to a *new* repo will be merged. The LR overfits to noise in the hub_score dimension, hurting generalization.

The T_2024 cutoff (n=1096) shows the closest result: mr+hub 0.578 vs mr_only 0.565. Even here the improvement is +0.013 and non-significant (p=0.94). With the largest available sample at the most favorable cutoff, hub_score can't clear the bar.

**Conclusion for GE v3: drop hub_score from the scoring model.** For the population GE actually scores (unknown contributors to the target repo), `merge_rate` alone outperforms every model that includes hub_score. This holds across all repo size tiers and all temporal cutoffs. hub_score should be removed from the v3 scoring formula, not just downweighted.

## Iteration 12: Recency Window for Unknown Contributors (stage18)

**Question:** Does a shorter merge_rate lookback window (3mo, 6mo, 1yr, 2yr) improve prediction over alltime for the GE scoring population?

Stage13 Experiment D showed 3mo MR (AUC 0.675) dramatically outperformed alltime MR (0.542) for cross-repo prediction on all authors. But that included known contributors with rich activity histories. For unknown contributors, the picture is different.

**Method:** Same unknown-to-repo population as stage17. Compare 5 raw MR variants (alltime, 2yr, 1yr, 6mo, 3mo) plus exponentially-weighted (half-life 180 days) as univariate predictors. Also test fallback variants (use window MR if >=2 PRs in window, else alltime). All 6 cutoffs, DeLong tests.

**Results (mean AUC across cutoffs):**

| Window | All (medium+) | Large | XL |
|---|---|---|---|
| alltime | 0.516 | 0.553 | 0.533 |
| 2yr | 0.519 | 0.568 | 0.541 |
| 1yr | 0.521 | 0.568 | 0.547 |
| 6mo | 0.529 | 0.569 | 0.565 |
| 3mo | 0.525 | 0.564 | 0.552 |
| weighted | 0.516 | 0.550 | 0.537 |

**Zero DeLong tests are significant** across any window, any tier, any cutoff.

6mo is the best raw performer overall (0.529) and in XL repos (0.565), but the differences are tiny (0.01-0.03). The fallback variants (window MR if >=2 PRs, else alltime) don't help either -- all within 0.01 of alltime.

**Why recency doesn't matter here:** At the T_2024 cutoff, only 29% of unknown contributors have >=2 PRs in the 6mo window, and only 20% have >=2 in the 3mo window. Most unknown contributors are too sparse for short windows to differentiate them. The signal in merge_rate for this population is "have you ever gotten PRs merged anywhere?" not "have you been active lately?" -- because someone new to a repo is, almost by definition, not a frequent recent contributor.

This contrasts with stage13's cross-repo result (3mo AUC 0.675 >> alltime 0.542) because that tested all authors including known, active contributors where recent activity is highly informative.

**Conclusion for GE v3: keep alltime merge_rate.** There is no evidence that a recency window improves prediction for unknown contributors. The implementation complexity of windowed MR (minimum-PR-count fallback, window selection) is not justified. GE v3 should use alltime merge_rate as-is.

## Iteration 12b: Account Age for Unknown Contributors (stage19)

**Question:** Does `log_account_age` improve merge prediction for unknown contributors on top of merge_rate?

PR #27 (validation study) found account_age was LRT-significant (LR = 19.16, p = 1.2e-5) against graph_score, but did not improve AUC when combined. The current scorer.py v2 formula uses `graph_score + merge_rate + log_account_age`. If we drop graph_score (per iteration 11b), does account_age still add value alongside merge_rate for the unknown-contributor population?

**Method:** Same unknown-to-repo population as stage17/18. Three models: `mr_only` (merge_rate as univariate), `mr_age` (LR on merge_rate + log_account_age), `age_only` (log_account_age as univariate). 5-fold CV (LOO when n_pos < 30). DeLong tests with Holm-Bonferroni. Note: account_age_days is available for ~41% of authors in the parquet; analysis restricted to rows with non-null values.

**Results (4 stable cutoffs, 5-fold CV):**

| Cutoff | N | mr_only | age_only | mr+age | DeLong p (mr+age vs mr) |
|---|---|---|---|---|---|
| T_2022 | 130 | 0.584 | 0.505 | 0.576 | 0.807 |
| T_2022-07 | 431 | 0.606 | 0.522 | 0.606 | 0.992 |
| T_2023 | 474 | 0.552 | 0.516 | 0.534 | 0.076 |
| T_2024 | 1014 | 0.580 | 0.516 | 0.569 | 0.111 |

The two early cutoffs (T_2020 n=18, T_2021 n=34) used LOO-CV and showed wild instability (mr+age AUC 0.11-0.17), typical of overfitting on tiny samples.

**By repo size tier (mean across 6 cutoffs):**

| Tier | mr_only | mr+age | age_only |
|---|---|---|---|
| All (medium+) | **0.540** | 0.427 | 0.554 |
| Large | **0.593** | 0.497 | 0.525 |
| XL | **0.564** | 0.515 | 0.538 |

**Key findings:**

1. **mr+age never beats mr_only** at any stable cutoff. It ties (T_2022-07) or hurts (all others). DeLong p > 0.07 everywhere.
2. **age_only** is barely above chance (0.505-0.522) for stable cutoffs, far worse than merge_rate.
3. The mean across all 6 cutoffs is distorted by LOO instability at T_2020/T_2021, where mr+age collapses. Even excluding those, the LR adds nothing.

**Why account_age doesn't help unknown contributors:** Account age weakly proxies "experience level" but doesn't distinguish between experienced contributors trying a new repo (high age, high merge likelihood) and inactive old accounts returning (high age, uncertain merge likelihood). For unknown contributors, the meaningful signal is "do your PRs get accepted elsewhere?" (merge_rate), not "how long has your GitHub account existed?"

PR #27 found account_age useful as a cold-start tiebreaker when graph_score = 0 (14.2% of PRs from newcomers). But our population is already filtered to unknown contributors, where 100% are "cold" to the target repo. In this population, account_age adds noise.

**Conclusion for GE v3: drop log_account_age from the scoring model.** The v2 formula (graph_score + merge_rate + log_account_age) simplifies to merge_rate alone for the unknown-contributor population. All three v2 features beyond merge_rate (graph_score, hub_score proxy, and account_age) have now been tested and eliminated.

### Remaining work: Bad Egg

1. **Threshold calibration** -- the tier cutoffs (top 1%, top 5%) are arbitrary percentiles. Calibrate against a decision-theoretic cost model: what's the cost of a false positive (annoying a legitimate contributor) vs false negative (missing a bad actor)?
2. **Integration** -- add `SuspicionScore` model to `models.py`, advisory tier computation to `scorer.py`, and output formatting to `formatter.py`. Wire into the Action, CLI, and MCP interfaces.
3. **Seed-free design** -- the OSS version uses only pre-trained LR coefficients (shipped as config), no seed file. The coefficients from the T_2024 fit are candidates, but cross-temporal stability should be verified.

### Remaining work: Good Egg v3

1. **Remove hub_score and log_account_age from scoring formula** -- stage17 shows hub_score hurts for unknown contributors across all repo sizes. Stage19 shows account_age adds nothing on top of merge_rate for this population. The graph is still built (needed for repo discovery and contributor mapping), but neither hub_score nor account_age should be scoring inputs.
2. **Scoring formula refit** -- replace the v2 3-feature LR (graph_score + merge_rate + log_account_age) with alltime merge_rate as the sole scoring input for unknown contributors. Stage18 confirms no recency window helps this population. The v2 LR is unnecessary when only one feature survives.

---

## Data Limitations

- Author metadata (account age, followers) only available for the 12,898 checked authors
- No labeled spam ground truth beyond account suspension status
- Suspension may be for reasons unrelated to PR spam (ToS violations, other abuse)
- LLM classification depends on PR title quality (bodies often empty)
- LLM scores only computed for 820 multi-repo authors with merge_rate < 0.5 -- all 28,088 single-repo authors get default score 0.0
- 18,398 single-repo authors remain unchecked for suspension status (higher merge rates)
- Network features are degenerate for single-repo authors (89.7% of population)
- Merge rate lookahead contamination affects H8, H13, the combined score, and the ground truth sample ordering (see Lookahead Contamination section)
