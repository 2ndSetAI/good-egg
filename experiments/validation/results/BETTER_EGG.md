# Better Egg: What the Validation Study Tells Us

**Date:** 2026-02-18
**Context:** Post-V4-audit analysis of the GE validation study results

This document synthesizes findings from the GE validation study to answer:
*What works, what doesn't, and how should the scoring approach evolve?*

---

## 1. What Works

### Recency-weighted contribution history

The ablation study (H2) shows recency is the dominant signal. Removing recency
decay drops AUC from 0.650 to 0.551 (Delta = -0.099, p < 10^-60). No other
graph dimension comes close. The core insight of GE---that *recent* merged
contributions predict future merge success---is sound.

Why it works: recent contributions demonstrate that an author is currently
active and familiar with the norms, tooling, and codebase patterns of the
open-source ecosystem. Stale contributions from years ago are less predictive
because projects, maintainers, and coding practices change.

### Author merge rate (external to graph)

With exact temporal scoping (using backfilled closed PR timestamps), author
merge rate alone achieves AUC = 0.534 and adds LR = 51.5 (p < 10^-12) when
combined with the GE score. The LRT confirms merge rate carries information
the graph misses. However, this statistical significance does not translate
to ranking improvement: LR(GE + merge_rate + age) achieves AUC = 0.656 on
the 4,467-PR merge-rate subset, not significantly different from the graph
alone (0.658 on the same subset, DeLong p = 0.65). Merge rate and age add
information but not discriminative power beyond what the graph already
captures through recency weighting.

The initially dramatic-looking result (merge rate alone matching GE) was
entirely an artifact of temporal leakage: using lifetime `merged_count /
(merged_count + closed_count)` included future information. This is a
cautionary tale about feature leakage in retrospective studies.

### Account age (modest signal)

Account age provides a small but statistically significant incremental signal
(LR = 19.16, p = 1.2e-5). However, like merge rate, it does not improve AUC
when combined with the graph (see Section 6.7). It is useful primarily as a
cold-start tiebreaker when the graph score is zero.

### Embedding similarity (informative, but inverted)

PR body vs. repo README embedding similarity adds significant signal (LR =
35.20, p = 3.0e-9), but the direction is inverted: higher similarity is
associated with *lower* merge probability (standalone AUC = 0.411). This
inversion is consistent across every method tested in the robustness
sub-study (Gemini, TF-IDF, MiniLM at three token lengths, Jaccard).

Possible explanations: (a) boilerplate/template PRs closely match README text
but are lower quality, (b) merged PRs tend to target specific subsystems whose
vocabulary diverges from the high-level README. The feature captures something
real---possibly PR *novelty* or *specificity* rather than "content relevance"
in the positive sense originally hypothesized.

A [robustness sub-study](similarity_comparison/comparison_report.md) confirmed
the signal is not a Gemini artifact: on the Gemini subset (n=1,569), Jaccard
(a simple bag-of-words method) also survives Holm-Bonferroni correction (adj.
p = 0.001). On the full dataset (n=5,417, using title as fallback text), all
five non-Gemini methods are highly significant (all adj. p < 10^-8).

**Limitation:** Gemini embeddings were available for only 1,569 of 5,417 PRs
(29%). However, simpler methods (TF-IDF, Jaccard) work on the full dataset
since they only require raw text, not a separate embedding API call.

---

## 2. What Doesn't Work (for Merge Prediction)

### Graph structure beyond recency

The H2 ablation shows recency dominates, but repo quality and language match
now show small, statistically significant contributions that survive
Holm-Bonferroni correction. Three of six dimensions remain non-significant:

| Dimension | Delta AUC | Assessment |
|-----------|-----------|------------|
| Language match | +0.002 | Small but significant (raw p = 0.038, adj. p = 0.038) |
| Diversity/volume | +0.000 | No signal (adj. p = 0.969) |
| Language normalization | +0.000 | No signal (adj. p = 0.969) |
| Self-contribution penalty | -0.000 | No signal (adj. p = 0.969) |
| Repo quality | -0.003 | Significant (raw p = 0.003, adj. p = 0.015) |

Repo quality and language match survive correction, but their effect sizes
are tiny compared to recency (|Delta| < 0.003 vs. -0.099). The remaining
three dimensions may serve other purposes (interpretability, fairness, gaming
resistance) but do not improve merge prediction.

### Self-contribution penalty

Designed to penalize contributions to self-owned repos, this has zero measured
impact. A [dedicated sub-study](self_penalty_evaluation/report.md) tested three
variants: 0.3x (current), 1.0x (no penalty), and 0.0x (full exclusion from the
graph). All three produce effectively identical AUCs (0.650, 0.650, 0.651; all
pairwise DeLong p > 0.32). Even full exclusion changes only 46% of scores (mean
shift -0.019) and moves AUC by +0.001. The penalty is neither helpful nor
harmful --- it can be safely removed to simplify the model.

### The full graph vs. simple features

After correcting temporal leakage (backfilling exact closed PR timestamps),
the GE graph significantly outperforms all simple feature baselines:

| Baseline | AUC | vs. GE (p) |
|----------|-----|-----------|
| Author merge rate alone | 0.534 | < 10^-21 |
| Model A (rate + age) | 0.556 | < 10^-19 |
| Prior merge count | 0.604 | < 10^-12 |
| Followers (log) | 0.615 | < 10^-3 |

The graph's complexity is justified by merge prediction performance. The
initial finding that "merge rate matches GE" was entirely driven by temporal
leakage in the merge rate denominator. With proper scoping, the graph adds
4--12 AUC points over any simple feature.

---

## 3. The Recency Bias Question

Is the GE score just a "recency bias engine"? Recency is the dominant
dimension, but the graph adds more than just recency:

1. **Recency accounts for most of the signal above chance**
   (AUC 0.650 vs. 0.551 without recency; 0.551 is only 5pp above 0.500).
2. **The graph without recency (AUC = 0.551) is barely above chance.** All
   other dimensions combined add ~5pp over random.
3. **However, prior merge count alone (AUC = 0.604) falls well short of the
   full graph (0.650).** The graph's recency weighting and normalization
   across the contribution network adds ~5 AUC points over a simple count.
4. **No simple feature combination matches the graph.** Even followers +
   prior count combined would not reach 0.650 (DeLong tests confirm
   significant gaps for all baselines tested).

The graph is more than just a count of recent merges. The exponential decay,
repo quality weighting (marginal), and graph scoring normalization contribute
real signal that simple arithmetic misses.

---

## 4. Spammer and Gaming Vulnerability

The GE score has a structural vulnerability to certain adversarial patterns:

### Survivorship bias

GE computes scores exclusively from *merged* PRs. An author who submits 100
PRs with 10 merged and 90 rejected appears identical to one who submits 10 PRs
with 10 merged. The rejected PRs are invisible to the graph. This means:

- A spammer who opens many low-quality PRs to many repos, getting a few merged
  by chance, builds trust indistinguishable from a careful contributor.
- The H5 result (author merge rate adds LR = 51.5 beyond GE) confirms the
  graph misses rejection signal.

A [rejection awareness sub-study](rejection_awareness/report.md) tested whether
graph-integrated merge-rate scaling could address this. Two approaches were
evaluated: per-repo scaling (scale each edge by the author's merge rate at
that repo) and author-level scaling (scale all edges by overall merge rate).
Neither produced a statistically significant improvement (all DeLong
p > 0.07). Notably, even among high-rejection authors (merge rate < 0.5,
n=927), graph-integrated scaling barely moved AUC (0.551 vs. full model's
0.551), while the LR(GE + merge_rate) feature-engineering approach achieved
0.584 in that subgroup (cross-validated). This suggests rejection signal is
better captured as a *separate feature* than through edge weight scaling in
the graph.

### Newcomer cold-start

14.2% of PRs in the study come from authors with score = 0 (AUC = 0.500, pure
chance). These are not necessarily bad contributors---they have a 67.8% merge
rate---but the GE score cannot help with them at all. Any contributor's first
PR to a new ecosystem is unscored.

---

## 5. Evidence Disposition: What Stays, What Goes

Every component of the v1 scoring model has now been evaluated. This table
summarizes the evidence and the disposition for v2.

| Component | v1 Status | Evidence | v2 Decision |
|-----------|-----------|----------|-------------|
| Recency decay | Active | H2: Delta AUC = -0.099, p < 10^-60 | **Keep** |
| Repo quality (stars, archived, fork) | Active | H2: Delta AUC = -0.003, raw p = 0.003, adj. p = 0.015 | **Keep** (significant, cheap) |
| Self-contribution penalty (0.3x) | Active | [Self-penalty study](self_penalty_evaluation/report.md): all 3 variants identical, p > 0.32 | **Drop** |
| Language match (personalization) | Active | H2: Delta AUC = +0.002, raw p = 0.038 | **Keep** (significant, cheap) |
| Diversity/volume scaling | Active | H2: Delta AUC = +0.000, p = 0.969 | **Drop** |
| Language normalization (star multipliers) | Active | H2: Delta AUC = +0.000, p = 0.969 | **Drop** |
| Author merge rate | Not in graph | H5: LR = 51.5, p < 10^-12 (n=5,129, proportional merge rate); [rejection study](rejection_awareness/report.md): LR = 65.73, p < 10^-16 (n=4,467, cross-validated, exact timestamps only) | **Add as feature** |
| Account age | Not in graph | H3: LR = 19.16, p = 1.2e-5 | **Add as feature** |
| Text dissimilarity | Not in graph | H4: LR = 35.20, p = 3.0x10^-9 (inverted) | **Add as feature** |
| Graph-integrated rejection scaling | Tested | [Rejection study](rejection_awareness/report.md): both approaches adj. p > 0.07 | **Do not add** |

---

## 6. Better Egg v2: Full Specification

### 6.1 Architecture Overview

v2 uses a two-layer architecture: a simplified trust graph (Layer 1) combined
with external features (Layer 2) via a trained logistic regression (Layer 3).

```
final_score = CombinedModel(
    graph_score,          # Layer 1: simplified trust graph
    temporal_merge_rate,  # Layer 2: external features
    log_account_age,
    text_dissimilarity,   # required for AUC lift; simple methods suffice
)
```

---

### 6.2 Layer 1: Simplified Trust Graph

A bipartite directed graph (user->repo, repo->user) scored with graph scoring.
Structurally identical to v1, but with three dimensions removed.

#### 6.2.1 Graph Construction

**Nodes:**
- `user:{login}` --- one per author
- `repo:{owner/name}` --- one per contributed repository

**Edges (user->repo, forward):**

```
forward_weight = sum_pr [ recency_decay(pr.days_ago)
                          * repo_quality(repo)
                          * edge_multiplier ]
```

Summed over the author's merged PRs at that repo, capped at
`MAX_PRS_PER_REPO` most recent.

**Edges (repo->user, reverse):**

```
reverse_weight = forward_weight * reverse_edge_ratio
```

**What changed from v1:** The `* 0.3` self-contribution penalty is removed.
All repos are treated equally regardless of ownership.

#### 6.2.2 Recency Decay

```
recency_decay(days_ago) =
    0.0                                    if days_ago > max_age_days
    exp(-0.693 * days_ago / half_life)     otherwise
```

| Parameter | Default | Type | Evidence |
|-----------|---------|------|----------|
| `half_life_days` | 180 | int | H2: removing recency -> AUC 0.551. Half-life not separately optimized; 180 is the v1 default. |
| `max_age_days` | 730 | int | Contributions >2 years old decay to ~0 anyway with 180-day half-life. Hard cutoff for efficiency. |

#### 6.2.3 Repo Quality

```
repo_quality(meta) =
    1.0                                   if meta is None
    quality = log(1 + stars)              otherwise
    if is_archived: quality *= 0.5
    if is_fork:     quality *= 0.3
```

**What changed from v1:** The `stars * language_multiplier` term is simplified
to just `stars`. The 28-entry language normalization table is removed
(H2: Delta AUC = 0.000, p = 0.969). Stars alone carry the signal.

| Parameter | Default | Type | Evidence |
|-----------|---------|------|----------|
| `archived_penalty` | 0.5 | float | Retained from v1; not independently tested but cheap and logically sound |
| `fork_penalty` | 0.3 | float | Same |

#### 6.2.4 Personalization Vector

The restart distribution for graph scoring, determining how much weight each
repo node receives.

```
personalization[context_repo] = context_repo_weight
personalization[other_repos]  = other_weight        # uniform for all non-context repos
personalization[user_nodes]   = 0.0
```

Normalized to sum to 1.0.

**What changed from v1:** Three things removed:

1. `same_language_weight` --- all non-context repos get the same weight
   (H2: Delta AUC = +0.002 for language match, now significant at p = 0.038)
2. `diversity_scale` --- no longer adjusts `other_weight` based on unique
   repo count (H2: Delta AUC = 0.000)
3. `volume_scale` --- no longer adjusts `other_weight` based on PR count
   (H2: Delta AUC = 0.000)

| Parameter | Default | Type | Evidence |
|-----------|---------|------|----------|
| `context_repo_weight` | 0.5 | float | Retained from v1. Not independently ablated but structurally necessary --- the context repo should dominate the restart vector. |
| `other_weight` | 0.03 | float | Now a fixed value, no longer dynamically adjusted. |

#### 6.2.5 Graph Scoring Algorithm

```
scores = nx.pagerank(graph, alpha=alpha, personalization=pvec, weight="weight")
raw_score = scores["user:{login}"]
```

Internally uses `nx.pagerank()`.

| Parameter | Default | Type | Evidence |
|-----------|---------|------|----------|
| `alpha` (damping) | 0.85 | float | Standard damping factor; not independently tuned |
| `max_iterations` | 100 | int | Convergence parameter |
| `tolerance` | 1e-6 | float | Convergence parameter |

#### 6.2.6 Normalization

```
baseline = 1.0 / n_nodes
ratio = raw_score / baseline
normalized_score = ratio / (ratio + 1.0)     # sigmoid mapping: uniform -> 0.5
```

Unchanged from v1. Maps raw graph scoring output to [0, 1].

#### 6.2.7 Anti-Gaming Mechanisms

| Mechanism | Value | Configurable? | Purpose |
|-----------|-------|:---:|---------|
| `MAX_PRS_PER_REPO` | 20 | Hardcoded | Prevents score inflation via high-volume contributions to a single repo |
| `reverse_edge_ratio` | 0.3 | Hardcoded | Reverse edges (repo->user) are 0.3x forward weight; prevents repo node from over-amplifying author score |
| `edge_multiplier` | 1.0 | Hardcoded | Base weight for merged PR edges (extensible to reviews, stars in future) |

---

### 6.3 Layer 2: External Features

Three features not captured by the graph, combined with the graph score via
a trained model.

#### 6.3.1 Temporal Merge Rate

```
merged_before_T = count(author's merged PRs where merged_at < T)
closed_before_T = count(author's closed PRs where closed_at < T)
temporal_merge_rate = merged_before_T / (merged_before_T + closed_before_T)
```

Where `T` = the creation time of the PR being scored.

**Data requirement:** Closed PR timestamps per author. Available via GitHub
GraphQL `pullRequests(states: CLOSED)`. The validation study backfilled data
for 1,958 of 2,538 authors (77.1%), capped at 500 most recent per author.
Production should fetch all available or document the cap.

**Evidence:** H5 corrected LR = 51.5 (p < 10^-12). Rejection awareness study
LRT = 65.73 (p < 10^-16, cross-validated). Among high-rejection authors (rate < 0.5, n=927),
this feature lifts subgroup AUC from 0.551 to 0.584 (cross-validated).

**Edge case:** If `merged_before_T + closed_before_T = 0`, merge rate is
undefined. Use a prior or treat as missing (impute population mean, or exclude
from the feature vector and let the combined model handle it).

**No tunable parameters.** This is a computed statistic.

#### 6.3.2 Account Age

```
log_account_age = log(account_age_days + 1)
```

**Evidence:** H3 LR = 19.16, p = 1.2e-5. Modest signal. Most useful as a
cold-start discriminator when graph score is 0 (14.2% of PRs are from
newcomers with zero graph history; these newcomers have a 67.8% merge rate
but AUC = 0.500 from the graph alone).

**No tunable parameters.** Computed from `UserProfile.created_at`.

#### 6.3.3 Text Dissimilarity

```
text_dissimilarity = 1.0 - similarity(pr_text, repo_readme)
```

Where `similarity` can be any of:

- **TF-IDF cosine** --- no external API, works on all PRs (recommended for
  production)
- **Jaccard** --- simplest; word-set overlap
- Gemini/MiniLM embeddings --- more expensive, validated but not required

The *inverted* direction is intentional: higher PR-README similarity predicts
*lower* merge probability (standalone AUC = 0.411). PRs that diverge from the
README (targeting specific subsystems) merge more often than generic/template
PRs that echo the README text.

**Evidence:** H4 LR = 35.20, p = 3.0x10^-9. Robustness sub-study: Jaccard
survives Holm-Bonferroni on Gemini subset (adj. p = 0.001); all 5 non-Gemini
methods significant on full dataset (all adj. p < 10^-8).

**Preprocessing parameters (if using TF-IDF):**

| Parameter | Default | Type | Notes |
|-----------|---------|------|-------|
| `max_features` | 10000 | int | Vocabulary size |
| `max_pr_text_chars` | 2000 | int | Truncation for PR body |
| `max_readme_chars` | 4000 | int | Truncation for README |

These are preprocessing constants, not model hyperparameters. The validation
study used these values.

---

### 6.4 Layer 3: Combined Model

#### 6.4.1 Architecture

Logistic regression combining the graph score with external features:

```
P(merge) = sigmoid(
    w0
  + w1 * graph_normalized_score
  + w2 * temporal_merge_rate
  + w3 * log_account_age
  + w4 * text_dissimilarity          # required for AUC improvement
)
```

**Evidence:** The combined model (GE + merge_rate + age + embedding) achieved
AUC = 0.654 (CV mean = 0.622) vs. GE alone at 0.650. DeLong p = 0.026 for the
improvement. However, the intermediate model *without* text dissimilarity
(GE + merge_rate + age) achieves AUC = 0.656, which is not significantly
different from GE alone (DeLong p = 0.65). The AUC improvement comes from
text dissimilarity, not from merge rate or account age. Those features carry
statistically significant information (per LRT) but do not improve ranking
performance beyond what the graph already captures.

#### 6.4.2 Training

| Parameter | Value | Notes |
|-----------|-------|-------|
| `penalty` | None | Unregularized. Required for valid LRT chi-squared assumption per DOE. With 3--4 features and thousands of observations, regularization is unnecessary. |
| `max_iter` | 1000 | Convergence for logistic regression solver |
| `random_state` | 42 | Reproducibility |

**Training data:** The weights `w0..w4` must be fit on a calibration dataset.
Options:

1. Fit on the full validation study data (5,417 PRs, 49 repos) and ship fixed
   weights
2. Fit per-deployment on the target repository's historical PR data
3. Use the validation study weights as defaults with per-repo fine-tuning

The validation study used option 1 (in-sample). Cross-validation (5-fold,
grouped by repo) showed stability: mean AUC = 0.649 +/- 0.044.

#### 6.4.3 Output and Classification

The combined model outputs a calibrated probability P(merge) in [0, 1],
classified into trust levels:

| Trust Level | Threshold | Evidence |
|-------------|-----------|----------|
| HIGH | P >= `high_trust` | v1 default; validation confirms meaningful separation (OR = 3.88 HIGH vs LOW, CI: 3.39--4.45) |
| MEDIUM | `medium_trust` <= P < `high_trust` | v1 default |
| LOW | P < `medium_trust` | v1 default |
| UNKNOWN | No data | Newcomer with no merged PRs and no closed PRs |
| BOT | `is_bot` flag | Short-circuit, unchanged from v1 |

| Parameter | Default | Type |
|-----------|---------|------|
| `high_trust` | 0.7 | float |
| `medium_trust` | 0.3 | float |
| `new_account_days` | 30 | int |

---

### 6.5 Complete Parameter Inventory

#### Graph construction (6 parameters, down from 14 in v1)

| Parameter | Default | Retained from v1? |
|-----------|---------|:-:|
| `half_life_days` | 180 | Yes |
| `max_age_days` | 730 | Yes |
| `archived_penalty` | 0.5 | Yes |
| `fork_penalty` | 0.3 | Yes |
| `MAX_PRS_PER_REPO` | 20 | Yes |
| `reverse_edge_ratio` | 0.3 | Yes |

#### Graph scoring (3 parameters, unchanged)

| Parameter | Default |
|-----------|---------|
| `alpha` | 0.85 |
| `max_iterations` | 100 |
| `tolerance` | 1e-6 |

#### Personalization (2 parameters, down from 5)

| Parameter | Default | Status |
|-----------|---------|--------|
| `context_repo_weight` | 0.5 | Retained |
| `other_weight` | 0.03 | Retained (now fixed, not dynamically adjusted) |
| ~~`same_language_weight`~~ | ~~0.3~~ | **Removed** (H2: p = 0.038) |
| ~~`diversity_scale`~~ | ~~0.5~~ | **Removed** (H2: p = 0.969) |
| ~~`volume_scale`~~ | ~~0.3~~ | **Removed** (H2: p = 0.969) |

#### Classification (2 parameters, unchanged)

| Parameter | Default |
|-----------|---------|
| `high_trust` | 0.7 |
| `medium_trust` | 0.3 |

#### Combined model (4--5 weights, NEW, learned)

| Weight | Learned from data |
|--------|:-:|
| `w0` (intercept) | Yes |
| `w1` (graph score) | Yes |
| `w2` (merge rate) | Yes |
| `w3` (account age) | Yes |
| `w4` (text dissimilarity) | Yes (required for AUC lift) |

#### Removed entirely from v1

| Parameter | v1 Default | Evidence for removal |
|-----------|------------|----------------------|
| `self_contribution_penalty` | 0.3 | [Self-penalty study](self_penalty_evaluation/report.md): 0.3x = 1.0x = 0.0x (all p > 0.32) |
| `same_language_weight` | 0.3 | H2: Delta AUC = +0.002, p = 0.038 (now significant; retained in v2 per updated disposition) |
| `diversity_scale` | 0.5 | H2: Delta AUC = 0.000, p = 0.969 |
| `volume_scale` | 0.3 | H2: Delta AUC = 0.000, p = 0.969 |
| `language_normalization.multipliers` | 28-entry table | H2: Delta AUC = 0.000, p = 0.969 |
| `language_normalization.default` | 3.0 | Same |
| `edge_weights.merged_pr` | 1.0 | Collapses to constant 1.0 when it is the only edge type; remove the config indirection |

**Net change:** 14 graph parameters -> 6, plus 4--5 learned weights. The
28-entry language normalization table is deleted entirely.

---

### 6.6 Data Requirements (v1 -> v2 diff)

| Data | v1 | v2 | Source |
|------|----|----|--------|
| Author's merged PRs | Required | Required | GraphQL `pullRequests(states: MERGED)` |
| Repo metadata (stars, language, archived, fork) | Required | Required (stars, archived, fork only; language no longer used in scoring) | GraphQL `repository` |
| Author's closed PRs (timestamps) | Not used | **Required** for merge rate | GraphQL `pullRequests(states: CLOSED)` --- fetch `closedAt` for each |
| Author profile (created_at) | Fetched but not scored | **Scored** (account age feature) | GraphQL `user` |
| PR body text | Not used | **Optional** (text dissimilarity) | REST API or already available at PR creation |
| Repo README text | Not used | **Optional** (text dissimilarity) | REST API `repos/{owner}/{repo}/readme` |

The main new data cost is closed PR timestamps. The validation study found 77.1%
of authors had this data available (1,958/2,538), with a backfill cap of 500
most recent per author. In production, fetching closed PRs adds one GraphQL
query per author.

**Note on text dissimilarity:** PR body and README text are listed as optional
data inputs, but text dissimilarity is the only external feature that produces
a statistically significant AUC improvement over the graph alone (see Section
6.7). Deployments that omit it will have merge rate and account age available
but should not expect AUC gains from those features alone.

---

### 6.7 Expected Performance

| Model | AUC | CV | n |
|-------|-----|----|----|
| v1 (full graph, 14 params) | 0.650 | 0.649 +/- 0.044 | 5,417 |
| v2 graph only (6 params, no externals) | ~0.650 | ~same | 5,417 |
| v2 combined (graph + merge_rate + age) | 0.656 ^a | cross-validated | 4,467 ^b |
| v2 combined + text dissimilarity | 0.654 | 0.622 +/- 0.037 | 1,495 ^c |

^a Cross-validated AUC from the [rejection awareness
study](rejection_awareness/report.md). The intermediate model does not
significantly differ from the graph alone (DeLong p = 0.65 on the same
4,467-PR subset where the graph achieves 0.658). Adding age significantly
improves over GE + merge_rate alone (LRT = 12.5, p = 0.0004), but the
combined model with external features roughly matches rather than exceeds
the graph.
^b Subset with temporally-scoped merge rate data (77.1% of authors).
^c Subset with valid Gemini embeddings and merge rate data.

The graph simplification (removing 3 inert dimensions: diversity/volume,
language normalization, and self-penalty) is expected to preserve AUC exactly
--- the ablation data confirms removing them individually or in combination
has no measurable effect. Repo quality and language match are retained as
they now show small but significant contributions. The improvement comes
from the external features, primarily merge rate.

---

### 6.8 Future Work (Speculative, Not in v2 Scope)

These ideas have some theoretical motivation but no direct evidence from the
validation study:

**a. Repository-specific calibration.**
The calibration plot shows systematic miscalibration. Fitting per-repository
or per-language Platt scaling could improve probability estimates, though
AUC (rank-based) would be unchanged.

**b. Temporal dynamics.**
Track how an author's merge rate *changes* over time, not just the static
rate. An author whose merge rate is improving may be more trustworthy than
one whose rate is declining.

**c. Review latency signals.**
Time-to-merge and time-to-close carry information about maintainer
confidence. Fast merges may indicate high trust; slow merges may indicate
caution. These are available in the existing data but not currently used.

---

## 7. What the Graph *Could* Be Good For (Beyond Merge Prediction)

The validation study only measures merge prediction. The GE graph may provide
value in ways not tested here:

1. **Cross-project reputation transfer.** The graph aggregates contributions
   across repos. A contributor who is active in the broader ecosystem may
   be more trustworthy than one who only contributes to a single repo, even
   if their merge rates are similar.

2. **Gaming resistance.** The graph's structural properties (graph scoring
   normalization, repo quality weighting) may make it harder to game than
   simple merge rate. A spammer who gets PRs merged in low-quality repos
   scores lower than one who contributes to high-quality repos. This wasn't
   tested in the validation study.

3. **Interpretability.** The graph provides a *story* ("this author has recent
   merged PRs in these related repos") that a single merge rate number does
   not. This may matter for trust decisions even if the AUC is similar.

4. **Network effects.** As more projects adopt GE, the graph becomes richer
   and potentially more discriminative. The current study uses a fixed set of
   49 repos; in production, the graph would span the entire contribution
   network visible to each author.

These hypotheses are worth testing in future work, particularly gaming
resistance and cross-project reputation transfer.
