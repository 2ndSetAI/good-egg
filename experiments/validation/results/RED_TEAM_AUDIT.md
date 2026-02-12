# Red Team Audit: GE Validation Study

**Date:** 2026-02-12
**Auditor:** Automated code review (full source read of all files in experiments/validation/)

---

## CRITICAL Issues

### C1. Likelihood ratio tests use L2-regularized logistic regression

**Files:** `stages/stage6_analyze.py` lines 244-266 (H3), 278-306 (H4), 318-346 (H5)

All three LRTs (H3 account age, H4 embedding similarity, H5 merge rate) use
`LogisticRegression(max_iter=1000, random_state=seed)`, which defaults to
`penalty='l2', C=1.0`. L2 regularization penalizes additional parameters and
can shrink them toward zero, which:

1. Invalidates the chi-squared distributional assumption of the LRT
2. Attenuates the test statistic for the extended model
3. Can produce LR = 0.0 when regularization fully absorbs the additional feature

**Impact:** H3 (LR=0.0, p=1.0) and H4 (LR=0.0, p=1.0) are almost certainly
artifacts of regularization, not genuine null results. H5 is still significant
(LR=315.2) because the signal is strong enough to overcome regularization,
but the test statistic is attenuated.

**Fix:** Add `penalty=None` to all LogisticRegression calls used in LRTs.

### C2. H4 embedding similarity is fundamentally broken

**Files:** `stages/stage5_features.py` lines 150-199, `embedding.py`

Multiple issues make the H4 analysis meaningless:

1. **Wrong text inputs**: The DOE specifies "cosine similarity between the PR
   description and the target repository's README/description." The
   implementation embeds `repo.replace("/", " ")` — e.g., "fastapi fastapi" —
   not actual PR descriptions or repo READMEs (stage5 lines 153-156).

2. **Author repos not embedded**: Author repo embeddings are looked up in
   `repo_emb_map`, which only contains the ~49 study target repos. Most
   authors' contributed repos are outside this set, so `author_embs` is
   typically empty, resulting in `sim = None`.

3. **Zero-vector fallback**: If `google-generativeai` is not installed,
   all embeddings are zero vectors (embedding.py lines 76-84), producing
   cosine similarity of 0.0 everywhere.

**Impact:** The H4 result ("not supported") is an artifact of broken feature
computation, not evidence that semantic similarity lacks predictive value.

**Fix:** Mark H4 as "inconclusive (implementation limitation)" in the report.
A proper implementation would require fetching repo READMEs and PR descriptions
via the GitHub API, which is out of scope for a code-only fix. Fixing the LRT
regularization (C1) is still warranted so the test is at least statistically
valid given whatever data is present.

### C3. Still-open PRs from study period never collected

**Files:** `stages/stage1_collect_prs.py` lines 154-167

The DOE (Section 2.3) specifies: "PR is still open and the elapsed time since
opening exceeds the stale threshold plus a 30-day buffer → POCKET_VETO." But
Stage 1 only collects merged and closed PRs within each temporal bin. The
"open" search (lines 155-159) looks for PRs created *before* the first bin
(`<2024-01-01`), which is outside the study period entirely. PRs opened during
2024-2025 that are still open are never collected.

**Impact:** The pocket veto class is incomplete — it only contains closed PRs
whose time-to-close exceeded the stale threshold, missing all still-open
PRs. This underestimates the pocket veto rate and could bias results toward
the explicitly-rejected class.

**Fix:** Add a search for open PRs created within each temporal bin. This
requires API calls (backfill).

---

## MAJOR Issues

### M1. Brier score and log loss computed on uncalibrated scores

**Files:** `stats.py` lines 313-336, `stages/stage6_analyze.py` line 86

`compute_binary_metrics` passes the raw GE normalized score to
`brier_score_loss` and `log_loss`. These metrics require calibrated
probabilities as input. The GE score is not calibrated (as the calibration
plot confirms — scores in the 0.2-0.5 range correspond to ~90%+ actual merge
rates). The resulting Brier score (0.252) and log loss (4.639) are not
interpretable.

**Impact:** These metrics are misleading as reported. AUC-ROC and AUC-PR
are rank-based and remain valid.

**Fix:** Either remove Brier/log loss from the primary results, or add a note
that these metrics are computed on uncalibrated scores and should be
interpreted with caution. Alternatively, fit a Platt scaling (logistic
regression) calibration and report calibrated Brier/log loss.

### M2. Holm-Bonferroni correction applied to 10 tests instead of 6

**Files:** `stages/stage6_analyze.py` lines 221-228

The DOE Section 7.4 specifies: "Apply Holm-Bonferroni correction across the
6 single-dimension comparisons (H2). The two-way interaction variants serve
as exploratory checks." But the code applies correction to all ablation
variants including the 3 two-way interactions and the recursive quality
variant (10 total), making the correction more conservative than pre-registered.

**Impact:** The only significant results (no_recency, no_recency_no_quality)
survive even the over-correction, so the conclusion doesn't change. But the
correction should match the DOE specification.

**Fix:** Separate the 6 primary single-dimension ablations from the exploratory
variants and only apply Holm-Bonferroni to the 6 primaries.

### M3. Self-owned repository PRs not excluded from dataset

**Files:** `stages/stage2_discover_authors.py`

DOE Section 5 lists "Self-owned repositories: PRs to repos owned by the PR
author" as an exclusion criterion. Stage 2 does not implement this filter.
PRs where the author is an owner of the target repository remain in the
dataset.

**Impact:** Self-merges do not represent external trust signals, as the DOE
notes. Including them could inflate merge rates for prolific contributors.

**Fix:** Add a check in Stage 2 comparing `pr.author_login` against the
repository owner (the part before `/` in the repo name). This is an
approximation (org membership ≠ ownership), but matches the DOE's intent.

### M4. Spam filter excludes legitimately fast-merged PRs

**Files:** `stages/stage2_discover_authors.py` lines 264-271

The exclusion filter removes PRs where `(end_time - created_at) < 1 day`,
and `end_time` is `pr.closed_at or pr.merged_at`. This means merged PRs
that were merged within 24 hours are excluded as "spam." Many legitimate PRs
(trivial fixes, dependency updates, pre-approved changes) merge within hours.

**Impact:** Fast-merge bias — the dataset systematically excludes fast merges,
which are likely to be from trusted contributors. This could attenuate the
effect size for H1.

**Fix:** Only apply the < 1 day filter to non-merged PRs (closed without merge).

---

## MINOR Issues

### m1. Cochran-Armitage trend test not run (DOE Section 6.3)

The function `cochran_armitage_trend` exists in `stats.py` but is never called
in `stage6_analyze.py`. This is a pre-registered test.

### m2. Odds ratios not computed (DOE Section 6.4)

The function `odds_ratio` exists in `stats.py` but is never called. The DOE
specifies computing odds ratios for each trust level pair with 95% CIs.

### m3. One-vs-rest AUC not computed (DOE Section 6.2)

The DOE specifies three one-vs-rest AUC values (one per outcome class). Not
implemented.

### m4. 3x3 confusion matrix not generated (DOE Section 6.2)

The DOE specifies a confusion matrix at Youden's J optimal threshold. Not
implemented.

### m5. `_MERGE_BOT_CLOSERS` defined but never used

In `stage2_discover_authors.py` line 44, `_MERGE_BOT_CLOSERS` is defined but
`_is_merge_bot_close` only checks labels, not the closer's identity.

### m6. Feature importance plot uses regularized coefficients

The feature importance visualization uses L2-regularized logistic regression.
Coefficients are biased toward zero, which understates feature contributions.
Should use `penalty=None` for interpretability.

### m7. NoSelfPenaltyGraphBuilder override confirmed working

`ablations.py` overrides `_is_self_contribution()` as a static method. Verified
that the base class `TrustGraphBuilder` defines this as a separate `@staticmethod`
(not inline in `build_graph()`), and the signatures match exactly. The override
works correctly. The near-zero AUC difference genuinely means the self-contribution
penalty has negligible impact on merge prediction. **Not a bug.**

### m8. pilot_report.md is redundant with SUMMARY.md

Both exist in the results directory, causing confusion about which is
authoritative.

---

## Summary Table

| ID | Severity | Issue | Fixable in code? |
|----|----------|-------|:----------------:|
| C1 | CRITICAL | LRTs use regularized LR | Yes |
| C2 | CRITICAL | H4 embeddings are broken | Partial (report fix) |
| C3 | CRITICAL | Open PRs not collected | Backfill needed |
| M1 | MAJOR | Brier/log loss on uncalibrated scores | Yes |
| M2 | MAJOR | Holm-Bonferroni over-corrects | Yes |
| M3 | MAJOR | Self-owned repos not excluded | Yes |
| M4 | MAJOR | Spam filter removes fast merges | Yes |
| m1 | MINOR | Cochran-Armitage not run | Yes |
| m2 | MINOR | Odds ratios not computed | Yes |
| m3 | MINOR | One-vs-rest AUC missing | Yes |
| m4 | MINOR | Confusion matrix missing | Yes |
| m5 | MINOR | _MERGE_BOT_CLOSERS unused | Yes |
| m6 | MINOR | Feature importance regularized | Yes |
| m7 | MINOR | Self-penalty ablation may be no-op | Needs investigation |
| m8 | MINOR | Redundant pilot_report.md | Yes |
