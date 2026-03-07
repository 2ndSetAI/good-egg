# Red Team Audit Log

This document records findings from red team reviews at each stage checkpoint.
Issues are classified as CRITICAL / MAJOR / MINOR.

## Checkpoint 1: Data Integrity

*Triggered after Stage 1 writes `data/bot_detection.duckdb` and `data/stage1_complete.json`.*

Status: PASS (no issues found)

## Checkpoint 2: Anti-Lookahead

*Triggered after Stage 2 writes feature Parquet files to `data/features/`.*

Status: CRITICAL -- Merge rate lookahead contamination

`cache.get_author_aggregate_stats()` computes `merge_rate` from ALL PRs with no
temporal windowing. This means every downstream component that uses merge_rate
sees the outcome data it's supposed to predict. Specific contamination paths:

1. `check_account_status.py` orders authors by merge_rate ascending -- ground
   truth labels are biased toward low-merge-rate authors.
2. `stage6_llm_content.py` pre-filters by `merge_rate < 0.5` -- the LLM
   evaluation population was selected using a contaminated feature.
3. `stage6_semi_supervised.py` includes `merge_rate` in FEATURE_COLS (line 22)
   -- k-NN distances incorporate the contaminated feature.
4. `stage7_author_evaluate.py` defines the auxiliary target as
   `merge_rate < 0.30` -- evaluating H8 against this is circular.
5. The combined score inherits contamination from all components above.

Resolution: requires temporal holdout validation (future experiment). See
RESULTS.md "Lookahead Contamination" section.

## Checkpoint 3: Statistical Methods

*Triggered after Stage 3 writes `results/statistical_tests.json`.*

Status: PASS (PR-level iterations 1-4 correctly identified as non-discriminative)

## Checkpoint 4: Scale-Up

*Triggered after successful small-scale run (10 repos).*

Status: PASS

## Final Audit

*Triggered after full-scale run completes.*

Status: MAJOR -- Results require contamination caveats

All AUC numbers for H8, H13 (k-NN), and the combined score are inflated by
merge rate lookahead. H9 (temporal) and H10 (network) are not directly
contaminated. H11 (LLM) content signal is genuine but its population was
biased by the pre-filter. See RESULTS.md for detailed analysis.
