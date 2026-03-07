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
| H13 k-NN | 1.00 | 1.00 | 1.00 | 0.61 | 1.000 | 1.000 |
| H13 IF | 0.00 | 0.00 | 0.00 | 0.01 | 0.391 | 0.016 |
| **Combined** | **0.60** | **0.52** | **0.38** | **0.28** | **0.948** | **0.323** |

### Results: Auxiliary Target (98 suspicious / 31,296 authors)

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
- 0 of 609 campaign authors overlap with the 61 confirmed suspended accounts (different populations: campaigns are time-clustered, suspensions may be for other reasons)

### Analysis (Multi-Repo Population)

#### What worked

1. **The unit-of-analysis pivot was decisive.** Author-level features on the same 200K PR corpus produce AUC 0.619-1.000 against real ground truth. PR-level features on the same data produced AUC 0.479-0.503. Same data, different framing, completely different results.

2. **H11 (LLM content) is the strongest unsupervised signal for multi-repo authors.** P@10 = 0.70 (7 of the top 10 most suspicious authors by LLM score are actually suspended). AUC-ROC of 0.619 is moderate -- many suspended authors have titles that look legitimate in isolation. Cost: ~$0.15 for 820 authors.

3. **The combined score is practical.** P@25 = 0.52, P@100 = 0.28, AUC 0.948. Rank all 3,208 multi-repo authors by this score, check the top 50, and 19 are confirmed suspended.

4. **H13 k-NN is circular but useful.** Perfect AUC against primary target is expected -- seeds are the positives. Its value is in the combined score where it boosts features that correlate with known-bad accounts.

#### What didn't work

1. **H8 (merge rate alone) is weak.** AUC 0.565, P@k near zero. With 61 suspended accounts among 3,208 multi-repo authors, merge rate cannot separate them from legitimate low-merge-rate contributors.

2. **H9 (temporal) is inverted.** AUC 0.420, below chance. Suspended accounts have *less* variable timing than legitimate authors, opposite of the hypothesis.

3. **H10 (network) collapsed from 0.952 to 0.523.** The earlier result (Iteration 5, 27 seeds) was inflated by the seed selection bias -- the first 27 suspended accounts were cherry-picked as the lowest-merge-rate multi-repo authors, who also happen to have extreme network topology. Checking all 3,216 multi-repo authors added 34 suspended accounts with more typical network profiles.

4. **H13 Isolation Forest detects anomalies but not the right ones.** AUC 0.391 (below chance). It finds statistical outliers, most of which are prolific legitimate contributors.

5. **Campaign detection (H12) finds a different population.** The 609 time-clustered campaign authors don't overlap with the 61 suspended accounts.

### Pipeline Details (Multi-Repo)

- Ground truth: All 3,216 multi-repo authors checked via GitHub API (`--min-repos 2`). 61 suspended, 3,155 active. ~107 min.
- Stage 5: Author aggregate features (H8) + bipartite network graph (H10) for 31,296 authors. ~1m45s.
- Stage 6a: Time-series features (H9) for 31,296 authors. ~0.5s.
- Stage 6b: LLM content analysis (H11) for 820 pre-filtered authors (merge_rate < 0.5, repos >= 2) via Gemini 2.0 Flash. 820 cached. 4 parse failures (defaulted to 0.5). ~2s (cache hits).
- Stage 6c: Semi-supervised (H13) k-NN + Isolation Forest. 61 suspended seeds. ~instant.
- Stage 7: Author-level evaluation against primary (suspended) and auxiliary (suspicious) targets. Precision@k, AUC-ROC, AUC-PR, Mann-Whitney U. ~4s.
- Stage 8: Campaign detection. 101 anomalous repo-months, 609 campaign authors. ~1s.

---

## Iteration 5b: Single-Repo Population

### Motivation

The multi-repo analysis (Iteration 5) established that author-level features work for the 3,208 authors who contribute to 2+ repos. But 28,088 authors (89.7%) only appear in a single repo. These are a structurally different population: network features are degenerate (every single-repo author has identical hub score), LLM pre-filtering excludes them, and cross-repo behavior doesn't exist. Do the same hypotheses work, or does this population need different approaches?

### Ground Truth Expansion

Checked 9,682 single-repo authors via GitHub API (`--min-repos 1 --limit 10000`), ordered by merge rate ascending. Found **262 suspended accounts** (2.7% of checked). Combined with the 61 multi-repo suspended, total ground truth is now **323 suspended / 12,898 checked**.

The single-repo suspension rate (2.7%) is higher than multi-repo (1.9%). This makes sense: accounts that get suspended often don't survive long enough to contribute to multiple repos.

18,398 single-repo authors remain unchecked (higher merge rates, outside top 10K by suspicion).

### Results: Single-Repo (262 suspended / 28,088 authors)

| Hypothesis | P@10 | P@25 | P@50 | P@100 | AUC-ROC | AUC-PR |
|------------|------|------|------|-------|---------|--------|
| **H8 (merge rate)** | **0.10** | **0.08** | **0.08** | **0.05** | **0.801** | **0.023** |
| H9 (temporal) | 0.00 | 0.00 | 0.00 | 0.00 | 0.342 | 0.005 |
| H10 (network) | 0.00 | 0.04 | 0.02 | 0.02 | 0.500 | 0.009 |
| H11 (LLM) | 0.00 | 0.04 | 0.02 | 0.02 | 0.500 | 0.009 |
| H13 k-NN | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 | 1.000 |
| H13 IF | 0.00 | 0.00 | 0.00 | 0.00 | 0.542 | 0.010 |
| **Combined** | **0.70** | **0.72** | **0.82** | **0.86** | **0.999** | **0.860** |

### Results: All Authors Combined (323 suspended / 31,296 authors)

| Hypothesis | P@10 | P@25 | P@50 | P@100 | AUC-ROC | AUC-PR |
|------------|------|------|------|-------|---------|--------|
| H8 (merge rate) | 0.00 | 0.00 | 0.02 | 0.01 | 0.760 | 0.023 |
| H9 (temporal) | 0.00 | 0.00 | 0.00 | 0.00 | 0.427 | 0.008 |
| H10 (network) | 0.10 | 0.08 | 0.04 | 0.03 | 0.544 | 0.013 |
| H11 (LLM) | 0.50 | 0.40 | 0.22 | 0.15 | 0.528 | 0.033 |
| H13 k-NN | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 | 1.000 |
| H13 IF | 0.00 | 0.00 | 0.00 | 0.01 | 0.564 | 0.012 |
| Combined | 0.50 | 0.48 | 0.28 | 0.14 | 0.981 | 0.214 |

### Analysis: Two Populations, Two Detection Profiles

The single-repo and multi-repo populations require fundamentally different detection approaches. Treating them as one population masks the strengths of each.

#### Multi-repo detection profile

- **Best unsupervised signal**: H11 (LLM content), AUC 0.619, P@10 = 0.70
- **Network features**: Informative but weaker than initially thought (AUC 0.523 on full population vs 0.952 on biased seed set)
- **Merge rate**: Weak (AUC 0.565) -- multi-repo authors with low merge rates are often legitimate explorers
- **Combined (without k-NN)**: H11 carries most of the unsupervised signal

#### Single-repo detection profile

- **Best unsupervised signal**: H8 (merge rate), AUC 0.801
- **Network features**: Degenerate (AUC 0.500 exactly). Every single-repo author has the same hub score by definition.
- **LLM content**: Degenerate (AUC 0.500). Single-repo authors were excluded from LLM pre-filtering (requires repos >= 2), so all get default score 0.0.
- **Combined**: Dominated by k-NN (AUC 0.999, P@100 = 0.86). The combined score here is almost entirely the semi-supervised component.

#### Why merge rate works for single-repo but not multi-repo

Among single-repo authors, merge rate is binary or near-binary: either all your PRs got merged or none did. A single-repo author with 0% merge rate is a much stronger signal than a multi-repo author with 0% merge rate, because the multi-repo case includes legitimate contributors who got some PRs merged in some repos.

The AUC tells the story: H8 goes from 0.565 (multi-repo, weak) to 0.801 (single-repo, strong). The same feature, measured on different populations, behaves differently.

#### The k-NN dominance problem

The combined score for single-repo authors is almost entirely k-NN: P@250 = 1.00, AUC-PR = 0.860. This is because k-NN is the only feature with real discriminative power in this population -- merge rate has AUC 0.801 but terrible precision (P@100 = 0.05), while k-NN has both. But k-NN is semi-supervised (it uses the suspended labels as seeds), making the combined score partially circular.

Without k-NN, single-repo detection reduces to merge rate alone -- better than chance but not actionable for ranking.

#### Feature availability gap

The core problem with single-repo authors is feature poverty:

| Feature | Multi-repo | Single-repo |
|---------|-----------|-------------|
| H8 merge rate | Available (weak) | Available (strong) |
| H9 temporal | Available (inverted) | Available where 2+ PRs (inverted) |
| H10 network | Available (weak) | Degenerate (constant) |
| H11 LLM | Available (strong) | Not computed (excluded by pre-filter) |
| H13 k-NN | Available (circular) | Available (circular) |
| H13 IF | Available (inverted) | Available (weak) |

Two of the six features are structurally unavailable for single-repo authors: network topology is undefined for a single connection, and LLM content scoring was scoped to multi-repo authors. Extending LLM scoring to single-repo authors could help, but at 28K authors the cost and API time would be significant.

### Population Comparison Summary

| Metric | Multi-repo (3,208) | Single-repo (28,088) | All (31,296) |
|--------|-------------------|---------------------|--------------|
| Suspended | 61 (1.9%) | 262 (0.9%*) | 323 (1.0%*) |
| Best unsupervised | H11 AUC 0.619 | H8 AUC 0.801 | H8 AUC 0.760 |
| Combined AUC | 0.948 | 0.999 | 0.981 |
| Combined P@10 | 0.60 | 0.70 | 0.50 |
| Combined P@100 | 0.28 | 0.86 | 0.14 |

*Rate among checked authors. 18,398 single-repo authors remain unchecked (assumed active).

The single-repo combined score looks better, but it's almost entirely k-NN. The multi-repo combined score has more genuine unsupervised signal from H11.

### Pipeline Details (Single-Repo)

- Ground truth: 9,682 single-repo authors checked via GitHub API (`--min-repos 1 --limit 10000`), ordered by merge rate ascending. 262 suspended, 9,420 active. ~5.9 hours.
- Features: Reused existing author_features.parquet (all 31,296 authors already had H8-H13 features from multi-repo round). Re-ran stages 6c (semi-supervised with 323 seeds) and stage 7 (evaluation with population segmentation).
- New stage 7 variant: `run_stage7_by_population()` produces separate results for multi-repo, single-repo, and all populations.

## Data Limitations

- Author metadata (account age, followers) only available for the 12,898 checked authors
- No labeled spam ground truth beyond account suspension status
- Suspension may be for reasons unrelated to PR spam (ToS violations, other abuse)
- LLM classification depends on PR title quality (bodies often empty)
- LLM scores only computed for 820 multi-repo authors with merge_rate < 0.5 -- all 28,088 single-repo authors get default score 0.0
- 18,398 single-repo authors remain unchecked for suspension status (higher merge rates)
- Network features are degenerate for single-repo authors (89.7% of population)
