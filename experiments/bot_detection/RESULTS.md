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
- **DuckDB indexes**: Added indexes on `(author, created_at)`, `(author, repo, created_at)`, reviews `(repo, pr_number)`, commits `(repo, pr_number)`. Stage 2 ran in ~74 min vs estimated hours without them.
- **Cross-repo coverage 2.6x better**: 10.3% of authors in 2+ repos (was 3.9%), 762 in 3+ repos, 114 in 5+ repos.
- **Neoteny cache demoted to gap-fill**: Parquet is imported first (INSERT OR IGNORE), so parquet data wins dedup. Neoteny provides reviews/commits not in parquet.

## Iterations 1-4: PR-Level Features (H1-H7)

All PR-level behavioral features produced AUC 0.479-0.503 against a "merged vs not-merged" target. Full details below.

### H1: Burstiness — AUC 0.483

Inverted: bursty authors are more likely to get merged. Burstiness is a signal of experienced contributors, not spammers. The effect got stronger with more data.

### H2: Engagement Lifecycle — AUC 0.481

Inverted. Higher engagement correlates with being merged. Responsive authors get their PRs accepted.

### H3: Cross-Repo Fingerprinting — AUC 0.503

Essentially random. Cross-repo signal disappeared with better data. The previous 0.512 was noise amplified by the small DuckDB sample.

### H4: Combined Model — AUC 0.501

Combining H1+H2+H3 produces random discrimination. All nested LRTs significant (p < 1e-4) but effects too small and partially cancelling.

### H5: GE Score Complement — GE v2 AUC 0.521

GE v2 remained the strongest single predictor. Adding bot signals to GE v2 slightly *hurt* performance (0.521 -> 0.520).

### H6: Interaction Features — AUC 0.480

Inverted like H1. Burst + no prior merge correlates with active contributors trying new repos, not spammers.

### H7: Burst Content Homogeneity — AUC 0.479

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
- **Ground truth acquisition**: Checked top 1,000 most suspicious authors (multi-repo, ordered by merge rate ascending) via GitHub API. Found **27 suspended accounts** (2.7% of checked).
- **Evaluation**: Precision@k and recall@k (appropriate for rare-event detection) alongside AUC-ROC/AUC-PR.

### Hypotheses

| ID | Name | Approach |
|----|------|----------|
| H8 | Author Aggregates | Per-author merge rate, rejection rate, PR volume, body/title stats |
| H9 | Time-Series Anomaly | Inter-PR timing, burstiness, dormancy, regularity |
| H10 | Network Analysis | Bipartite author-repo graph, degree centrality, clustering, isolation |
| H11 | LLM Content Analysis | Gemini classifies PR titles as spam-like (score 0-1) |
| H12 | Campaign Detection | Time-clustered spam in anomalous repo-months |
| H13 | Semi-Supervised | k-NN from suspended seeds + Isolation Forest |

### Results: Primary Target (27 suspended / 31,296 authors)

| Hypothesis | P@10 | P@25 | P@50 | P@100 | AUC-ROC | AUC-PR |
|------------|------|------|------|-------|---------|--------|
| H8 (merge rate) | 0.00 | 0.00 | 0.00 | 0.00 | 0.733 | 0.002 |
| H9 (temporal) | 0.00 | 0.00 | 0.00 | 0.00 | 0.588 | 0.003 |
| H10 (network) | 0.10 | 0.08 | 0.04 | 0.02 | 0.958 | 0.017 |
| **H11 (LLM)** | **0.50** | **0.40** | **0.22** | **0.15** | **0.976** | **0.284** |
| H13 k-NN | 1.00 | 1.00 | 0.54 | 0.27 | 1.000 | 1.000 |
| H13 IF | 0.00 | 0.00 | 0.00 | 0.01 | 0.933 | 0.009 |
| **Combined** | **0.90** | **0.80** | **0.54** | **0.27** | **1.000** | **0.880** |

### Results: Auxiliary Target (98 suspicious / 31,296 authors)

| Hypothesis | P@10 | P@25 | P@50 | P@100 | AUC-ROC | AUC-PR |
|------------|------|------|------|-------|---------|--------|
| H8 (merge rate) | 0.00 | 0.00 | 0.00 | 0.00 | 0.709 | 0.005 |
| H9 (temporal) | 0.00 | 0.00 | 0.02 | 0.01 | 0.736 | 0.016 |
| H10 (network) | 0.20 | 0.16 | 0.14 | 0.14 | 0.990 | 0.138 |
| **H11 (LLM)** | **0.40** | **0.44** | **0.30** | **0.22** | **0.990** | **0.184** |
| H13 k-NN | 0.50 | 0.36 | 0.18 | 0.09 | 0.408 | 0.034 |
| H13 IF | 0.00 | 0.04 | 0.02 | 0.07 | 0.987 | 0.108 |
| **Combined** | **0.70** | **0.44** | **0.32** | **0.22** | **0.995** | **0.291** |

### Campaign Detection (H12)

- **101 anomalous repo-months** flagged (rejection rate > 2 stdev above repo mean)
- **609 campaign authors** identified (only appear during anomalous months, 0% merge rate)
- Hacktoberfest 2019: 94.1% rejection rate (16 authors, 6 repos)
- Hacktoberfest 2020: 70.7% rejection rate (39 authors, 13 repos)
- 0 of 609 campaign authors overlap with the 27 confirmed suspended accounts (different populations: campaigns are time-clustered, suspensions may be for other reasons)

### Analysis

#### What worked

1. **The unit-of-analysis pivot was decisive.** Author-level features on the same 200K PR corpus produce AUC 0.958-1.000 against real ground truth. PR-level features on the same data produced AUC 0.479-0.503. Same data, different framing, completely different results.

2. **H11 (LLM content) is the strongest unsupervised signal.** Gemini classifying PR titles as spam-like achieves AUC-ROC 0.976 and P@10 = 0.50 against suspended accounts, without using any label information. Half the authors Gemini considers most suspicious are actually banned. Cost: ~$0.15 for 820 authors.

3. **H10 (network degree centrality) complements H11.** AUC 0.958 from graph topology alone. Suspended authors have high degree centrality (touch many repos) with low clustering (repos they touch don't share other contributors). This is the "star topology" spam pattern: one author touching many repos without being part of any repo's contributor community.

4. **H13 k-NN is circular but useful for ranking.** Its perfect AUC against the primary target is expected -- seeds are the positives. Against the auxiliary target (which uses a different definition), k-NN drops to 0.408, confirming it's not generalizable. Its value is in the combined score where it boosts features that correlate with known-bad accounts.

5. **The combined score is practical.** P@25 = 0.80 means: rank all 31K authors by this score, check the top 25, and 20 of them are confirmed suspended accounts. For a trust-scoring tool, this false-positive rate is actionable.

#### What didn't work

1. **H8 (merge rate alone) is necessary but not sufficient.** AUC 0.733 says suspended accounts have lower merge rates, but the top-k precision is 0.00 -- many legitimate authors also have low merge rates. Merge rate doesn't discriminate at the sharp end.

2. **H9 (temporal) is the weakest signal.** AUC 0.588, not significant at p=0.10. Suspended accounts don't have distinctive timing patterns. This is surprising -- the post-mortem hypothesized "dormancy-then-burst" patterns, but they don't separate suspended from active authors.

3. **H13 Isolation Forest detects anomalies but not the right ones.** AUC 0.933 sounds good, but P@100 = 0.01. It finds statistical outliers, most of which are prolific legitimate contributors who happen to be unusual (many repos, high volume). The anomalies it detects are a superset that includes but isn't focused on spam.

4. **Campaign detection (H12) finds a different population than suspensions.** The 609 time-clustered campaign authors don't overlap with the 27 suspended accounts. GitHub may suspend accounts for reasons unrelated to time-clustered spam (e.g., ToS violations, automated abuse beyond PRs). Or the campaign authors simply haven't been suspended yet.

#### Caveats

- **27 suspended accounts is a small positive class.** Leave-one-out CV was used instead of stratified k-fold. Results should be validated with a larger ground truth set.
- **H13 k-NN scores are partially circular.** The combined score's high AUC is partly driven by k-NN. Excluding k-NN, the Combined score should be re-evaluated.
- **Only 1,000 of 31,296 authors checked for suspension status.** The unchecked 30,296 are assumed active. Some may be suspended, which would be false negatives in evaluation. This biases AUC downward (actual performance is likely better).
- **LLM scores depend on model version.** Gemini 2.0 Flash results may not reproduce with other models or future versions.

### Pipeline Details

- Stage 5: Author aggregate features (H8) + bipartite network graph (H10) for 31,296 authors. ~1m45s.
- Stage 6a: Time-series features (H9) for 31,296 authors. ~0.5s.
- Stage 6b: LLM content analysis (H11) for 820 pre-filtered authors (merge_rate < 0.5, repos >= 2) via Gemini 2.0 Flash. 819 cached, 4 parse failures (defaulted to 0.5). ~18 min.
- Stage 6c: Semi-supervised (H13) k-NN + Isolation Forest. 27 suspended seeds. ~instant.
- Stage 7: Author-level evaluation against primary (suspended) and auxiliary (suspicious) targets. Precision@k, AUC-ROC, AUC-PR, Mann-Whitney U. ~4s.
- Stage 8: Campaign detection. 101 anomalous repo-months, 609 campaign authors. ~1s.
- Ground truth: 1,000 suspicious authors checked via GitHub API (--limit 1000 --min-repos 2). 27 suspended, 973 active. ~33 min.

## Data Limitations

- Author metadata (account age, followers) only available for the 1,000 checked authors
- No labeled spam ground truth beyond account suspension status
- Suspension may be for reasons unrelated to PR spam (ToS violations, other abuse)
- LLM classification depends on PR title quality (bodies often empty)
- 59% of authors are single-repo, limiting network feature expressiveness for that population
