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

## Data Limitations

- Author metadata (account age, followers) only available for the 12,898 checked authors
- No labeled spam ground truth beyond account suspension status
- Suspension may be for reasons unrelated to PR spam (ToS violations, other abuse)
- LLM classification depends on PR title quality (bodies often empty)
- LLM scores only computed for 820 multi-repo authors with merge_rate < 0.5 -- all 28,088 single-repo authors get default score 0.0
- 18,398 single-repo authors remain unchecked for suspension status (higher merge rates)
- Network features are degenerate for single-repo authors (89.7% of population)
- Merge rate lookahead contamination affects H8, H13, the combined score, and the ground truth sample ordering (see Lookahead Contamination section)
