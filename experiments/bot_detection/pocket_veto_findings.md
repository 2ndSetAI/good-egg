# Pocket Veto Investigation — Findings

Investigation for issue #51. Does counting stale open PRs as implicit
rejections meaningfully change merge-rate distributions and improve the
signal's ability to separate suspended from active accounts?

## Dataset

- 200172 PRs across 96 repos
- State totals: {'CLOSED': 34637, 'MERGED': 146033, 'OPEN': 19502}
- Outcome totals: {'merged': 146033, 'pocket_veto': 30894, 'rejected': 23245}
- Labeled authors: 31293 (739 suspended, 30554 active)

## Staleness definitions compared

- **v3 (baseline)**: `merged / (merged + closed)` — current scorer.py.
- **age_universal**: open PR is stale if age > 90d since `created_at`.
- **age_per_repo**: open PR is stale if age > that repo's
  `stale_threshold_days` (populated in the DuckDB; default 30d).
- **idle_universal**: open PR is stale if it is still open AND idle > 90d (`fetch_now - updated_at`).
- **idle_per_repo**: same, with the per-repo threshold substituted.

The `idle_*` variants use a live re-fetch of every DB-OPEN PR's
`updatedAt` (see `fetch_open_pr_activity.py`). PRs that were OPEN at
the snapshot but have since been closed or merged are treated as
non-stale — the close/merge event itself is activity.

## Calibration sanity check

- Repos using the default 30d threshold: 78 / 96
- Repos with a calibrated threshold: 18
- Per-repo calibrated thresholds vs 2x median time-to-close:
  mean delta = 31.3110, median delta = 28.6972 (days).

## Distribution shift

Mean merge rate across all authors:

| Definition | mean | median | p10 | p90 |
|---|---|---|---|---|
| v3 baseline | 0.5567 | 0.7500 | 0.0000 | 1.0000 |
| age_universal (90d) | 0.5369 | 0.6667 | 0.0000 | 1.0000 |
| age_per_repo | 0.5333 | 0.6182 | 0.0000 | 1.0000 |
| idle_universal (90d) | 0.5514 | 0.7000 | 0.0000 | 1.0000 |
| idle_per_repo | 0.5506 | 0.6729 | 0.0000 | 1.0000 |

Per-author drop from the v3 baseline (n authors, >0.10 / >0.25):

- **age_universal**: 1836 / 895
- **age_per_repo**: 2111 / 1094
- **idle_universal**: 459 / 241
- **idle_per_repo**: 529 / 279

## Signal quality vs ground truth

2-feature logistic regression (merge_rate + log1p(median_additions)),
5-fold CV on 31293 labeled authors:

| Definition | CV AUC | Active mean | Suspended mean | Cohen's d |
|---|---|---|---|---|
| v3 baseline | 0.5494 | 0.5577 | 0.5136 | 0.0962 |
| age_universal | 0.5488 | 0.5377 | 0.5011 | 0.0811 |
| age_per_repo | 0.5486 | 0.5341 | 0.5010 | 0.0734 |
| idle_universal | 0.5489 | 0.5523 | 0.5128 | 0.0866 |
| idle_per_repo | 0.5488 | 0.5515 | 0.5128 | 0.0847 |

## Recommendation

See the `recommendation` field in `data/results/pocket_veto_analysis.json` for the machine-readable
decision logic. Text summary and follow-up branch sketch below.

**Keep v3 as-is** — No variant beats v3 CV AUC 0.5494 by >0.005 (aucs={'merge_rate_v3': 0.5494, 'merge_rate_universal': 0.5488, 'merge_rate_per_repo': 0.5486, 'merge_rate_idle_universal': 0.5489, 'merge_rate_idle_per_repo': 0.5488}). Cohen's d also fails to improve (base=0.096, best_alt=0.087).

### Follow-up branch sketch (if adopted)

- `src/good_egg/github_client.py`: extend `_COMBINED_QUERY` with an
  `openPullRequests` selection that pulls `createdAt`/`updatedAt` for
  each OPEN PR on the scored user (or `totalCount` if we can push the
  staleness filter into the query).
- `src/good_egg/models.py`: add `open_stale_pr_count: int` (or similar)
  to `UserContributionData`.
- `src/good_egg/scorer.py:256-261`: change the `_score_v3` merge-rate
  formula to `merged / (merged + closed + open_stale)`.
- `src/good_egg/config.py`: add the staleness threshold as a tunable
  config value.
- Tests: parallel coverage in `tests/test_scorer.py`.
