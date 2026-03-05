# Red Team Audit Log

This document records findings from red team reviews at each stage checkpoint.
Issues are classified as CRITICAL / MAJOR / MINOR.

## Checkpoint 1: Data Integrity

*Triggered after Stage 1 writes `data/bot_detection.duckdb` and `data/stage1_complete.json`.*

Status: PENDING

## Checkpoint 2: Anti-Lookahead

*Triggered after Stage 2 writes feature Parquet files to `data/features/`.*

Status: PENDING

## Checkpoint 3: Statistical Methods

*Triggered after Stage 3 writes `results/statistical_tests.json`.*

Status: PENDING

## Checkpoint 4: Scale-Up

*Triggered after successful small-scale run (10 repos).*

Status: PENDING

## Final Audit

*Triggered after full-scale run completes.*

Status: PENDING
