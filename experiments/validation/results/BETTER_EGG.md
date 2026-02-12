# Better Egg: What the Validation Study Tells Us

**Date:** 2026-02-12
**Context:** Post-V2-audit analysis of the GE validation study results

This document synthesizes findings from the GE validation study to answer:
*What works, what doesn't, and how should the scoring approach evolve?*

---

## 1. What Works

### Recency-weighted contribution history

The ablation study (H2) shows recency is the dominant signal. Removing recency
decay drops AUC from 0.671 to 0.550 (Delta = -0.121, p < 10^-68). No other
graph dimension comes close. The core insight of GE---that *recent* merged
contributions predict future merge success---is sound.

Why it works: recent contributions demonstrate that an author is currently
active and familiar with the norms, tooling, and codebase patterns of the
open-source ecosystem. Stale contributions from years ago are less predictive
because projects, maintainers, and coding practices change.

### Author merge rate (external to graph)

With exact temporal scoping (using backfilled closed PR timestamps), author
merge rate alone achieves AUC = 0.546 and adds LR = 49.8 (p < 10^-12) when
combined with the GE score. It is a meaningful incremental signal but far
weaker than it appeared before fixing temporal leakage (original LR = 462.4
with lifetime counts, corrected to 49.8 with exact scoping).

The initially dramatic-looking result (merge rate alone matching GE) was
entirely an artifact of temporal leakage: using lifetime `merged_count /
(merged_count + closed_count)` included future information. This is a
cautionary tale about feature leakage in retrospective studies.

### Account age (modest signal)

Account age provides a small but statistically significant incremental signal
(LR = 8.64, p = 0.003). Older accounts have more history and tend to be more
reliable contributors. This is useful primarily as a cold-start tiebreaker
when the graph score is zero.

### Embedding similarity (informative, but inverted)

PR body vs. repo README embedding similarity adds significant signal (LR =
20.82, p = 5.1e-6), but the direction is inverted: higher similarity is
associated with *lower* merge probability (standalone AUC = 0.416). This
inversion is consistent across every method tested in the robustness
sub-study (Gemini, TF-IDF, MiniLM at three token lengths, Jaccard).

Possible explanations: (a) boilerplate/template PRs closely match README text
but are lower quality, (b) merged PRs tend to target specific subsystems whose
vocabulary diverges from the high-level README. The feature captures something
real---possibly PR *novelty* or *specificity* rather than "content relevance"
in the positive sense originally hypothesized.

A [robustness sub-study](similarity_comparison/comparison_report.md) confirmed
the signal is not a Gemini artifact: on the Gemini subset (n=1,293), Jaccard
(a simple bag-of-words method) also survives Holm-Bonferroni correction (adj.
p = 0.001). On the full dataset (n=4,977, using title as fallback text), all
five non-Gemini methods are highly significant (all adj. p < 10^-8).

**Limitation:** Gemini embeddings were available for only 1,293 of 4,977 PRs
(26%). However, simpler methods (TF-IDF, Jaccard) work on the full dataset
since they only require raw text, not a separate embedding API call.

---

## 2. What Doesn't Work (for Merge Prediction)

### Graph structure beyond recency

The H2 ablation is definitive: five of six graph dimensions have negligible
impact on merge prediction (all |Delta AUC| < 0.002, none significant after
correction). Specifically:

| Dimension | Delta AUC | Assessment |
|-----------|-----------|------------|
| Language match | +0.001 | No signal |
| Diversity/volume | +0.000 | No signal |
| Language normalization | +0.000 | No signal |
| Self-contribution penalty | -0.000 | No signal |
| Repo quality | -0.002 | Suggestive (raw p = 0.023) but not significant |

These dimensions may serve other purposes (interpretability, fairness, gaming
resistance) but do not improve merge prediction.

### Self-contribution penalty

Designed to penalize contributions to self-owned repos, this has zero measured
impact. Likely because: (a) self-owned repo PRs are already excluded from the
study, and (b) the penalty is a flat 0.3x multiplier that doesn't distinguish
between meaningful self-hosted projects and trivial ones.

### The full graph vs. simple features

After correcting temporal leakage (backfilling exact closed PR timestamps),
the GE graph significantly outperforms all simple feature baselines:

| Baseline | AUC | vs. GE (p) |
|----------|-----|-----------|
| Author merge rate alone | 0.546 | < 10^-22 |
| Model A (rate + age) | 0.561 | < 10^-24 |
| Prior merge count | 0.608 | < 10^-18 |
| Followers (log) | 0.609 | < 10^-9 |

The graph's complexity is justified by merge prediction performance. The
initial finding that "merge rate matches GE" was entirely driven by temporal
leakage in the merge rate denominator. With proper scoping, the graph adds
6--12 AUC points over any simple feature.

---

## 3. The Recency Bias Question

Is the GE score just a "recency bias engine"? Recency is the dominant
dimension, but the graph adds more than just recency:

1. **Recency accounts for most of the signal above chance**
   (AUC 0.671 vs. 0.550 without recency; 0.550 is only 5pp above 0.500).
2. **The graph without recency (AUC = 0.550) is barely above chance.** All
   other dimensions combined add ~5pp over random.
3. **However, prior merge count alone (AUC = 0.608) falls well short of the
   full graph (0.671).** The graph's recency weighting and normalization
   across the contribution network adds ~6 AUC points over a simple count.
4. **No simple feature combination matches the graph.** Even followers +
   prior count combined would not reach 0.671 (DeLong tests confirm
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
- The H5 result (author merge rate adds LR = 49.8 beyond GE) confirms the
  graph misses rejection signal.

### Newcomer cold-start

14.4% of PRs in the study come from authors with score = 0 (AUC = 0.500, pure
chance). These are not necessarily bad contributors---they have a 69% merge
rate---but the GE score cannot help with them at all. Any contributor's first
PR to a new ecosystem is unscored.

---

## 5. Recommendations for Better Egg v2

Based on the validation results, here are concrete recommendations ranked by
expected impact:

### High priority (clear evidence)

**a. Incorporate author merge rate as a first-class signal.**
A meaningful incremental predictor (LR = 49.8, p < 10^-12). Computation:
count merged and closed PRs using GitHub API, apply temporal scoping
(`merged_at < T`, `closed_at < T`). Closed PR timestamps are available
via GraphQL `pullRequests(states: CLOSED)` query. The combined model
(GE + merge rate + age + embedding) achieves AUC = 0.680.

**b. Simplify the graph to recency + repo quality.**
Repo quality is the only non-recency dimension with a suggestive signal
(raw p = 0.023). Drop language match, diversity/volume, self-contribution
penalty, and language normalization from the scoring model. This reduces
API calls, computation, and configuration surface area while preserving
(or improving) prediction.

**c. Add account age as a cold-start tiebreaker.**
When the graph score is 0 (newcomer), use account age as the primary
discriminator. A 5-year account with many public repos is a better bet than
a 2-day-old account, even without contribution history.

### Medium priority (promising evidence)

**d. Add text similarity as a novelty/specificity signal.**
PR body vs. repo README similarity adds real predictive signal (LR = 20.8),
but the relationship is *inverted*: higher similarity predicts lower merge
probability. This means the useful signal is likely PR *specificity* or
*novelty* — PRs that diverge from the README tend to target concrete
subsystems, while generic/template PRs that echo the README text tend to fail.
Simple methods (TF-IDF, Jaccard) capture this signal without an embedding API
call and work on all PRs including title-only ones. Consider using
`1 - similarity` as a specificity feature, or using high similarity as a flag
for template/boilerplate PRs.

**e. Incorporate rejection/closed PR data.**
Currently invisible due to survivorship bias. The graph should at minimum
track the *existence* of closed PRs (not just merged ones) to penalize
high-rejection-rate authors. This requires expanding the GitHub data
collection to include closed PRs per author.

**f. Redesign self-contribution penalty.**
The current flat 0.3x multiplier is ineffective. Consider: instead of
penalizing self-contributions, use author merge rate on *external* repos
only. This naturally handles the gaming vector of self-merged PRs without
a blunt multiplier.

### Lower priority (speculative)

**g. Repository-specific calibration.**
The calibration plot shows systematic miscalibration. Fitting per-repository
or per-language Platt scaling could improve probability estimates, though
AUC (rank-based) would be unchanged.

**h. Temporal dynamics.**
Track how an author's merge rate *changes* over time, not just the static
rate. An author whose merge rate is improving may be more trustworthy than
one whose rate is declining.

**i. Review latency signals.**
Time-to-merge and time-to-close carry information about maintainer
confidence. Fast merges may indicate high trust; slow merges may indicate
caution. These are available in the existing data but not currently used.

---

## 6. The Simplest Better Egg

The validation results support keeping the graph as the foundation and
augmenting it with external features. The minimal effective improvement:

```
score = w1 * ge_graph_score(author, cutoff=T)   [keep as-is]
      + w2 * temporal_merge_rate(author, cutoff=T)
      + w3 * log(account_age_days)
      - w4 * text_similarity(pr_body, repo_readme)       [optional]
```

Where:
- `ge_graph_score` is the existing recency-weighted graph scoring (the core
  that works and outperforms all simple baselines)
- `temporal_merge_rate` adds rejection signal the graph misses
  (merged/(merged+closed) using only PRs before cutoff)
- `account_age` handles cold-start
- `text_similarity` enters with a *negative* weight (higher similarity →
  lower score) — this captures PR specificity/novelty, as generic/template PRs
  that echo the README text are less likely to merge. Simple methods like
  TF-IDF or Jaccard suffice; no embedding API call required.

The combined model (GE + all three features) achieves AUC = 0.680 with CV
mean = 0.671, a modest but real improvement over the graph alone (0.671).

A simpler graph (recency only, dropping the five ineffective dimensions)
should also be evaluated. This would reduce configuration surface area and
API requirements while preserving the core signal.

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
