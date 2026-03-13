# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-03-13

### Added

- **Diet Egg (v3) scoring model** -- uses alltime merge rate as the sole
  scoring signal, dropping graph score and log account age from v2.
- **Fresh Egg advisory** -- accounts under 365 days old receive an
  informational "Fresh Egg" note in all output formats. Does not affect
  the score. Bots are excluded since their synthetic profiles have
  unreliable age data.
- `FreshAccountAdvisory` model and `fresh_account` field on `TrustScore`.
- "Diet Egg" branding on PR comments when using v3.
- Example workflow for v3: `examples/diet-egg-workflow.yml`.

### Changed

- Default `scoring_model` changed from `v1` to `v3`.

## [1.0.0] - 2026-02-23

### Added

- **Better Egg (v2) scoring model** -- opt-in combined model that extends the
  graph score with merge rate and account age features via logistic regression.
  Trained on 5,129 PRs (of 5,417 total, filtered to those with merge rate
  data) from 49 repositories.
- `scoring_model` config option (`v1` default, `v2` for Better Egg).
- `v2:` config block with `graph:`, `features:`, and `combined_model:`
  sub-sections.
- `--scoring-model` CLI option.
- `scoring-model` GitHub Action input and output.
- `scoring_model` MCP server parameter on scoring tools.
- `GOOD_EGG_SCORING_MODEL` environment variable override.
- Component score breakdown in v2 output (`graph_score`, `merge_rate`,
  `log_account_age`).
- `scoring_model` and `component_scores` fields on `TrustScore` model.
- "Better Egg" branding on PR comments when using v2.
- Example workflow for v2: `examples/better-egg-workflow.yml`.

## [0.1.0] - 2026-02-10

### Added

- GitHub Action for automated PR author trust scoring.
- CLI tool (`good-egg score`) for scoring contributors from the command line.
- Python library API for programmatic trust scoring.
- MCP server for AI assistant integration (optional `mcp` extra).
- Graph-based trust scoring engine built on weighted contribution graphs.
- Bipartite graph construction from merged PRs, reviews, and repository metadata.
- Configurable trust level thresholds (HIGH, MEDIUM, LOW).
- Recency decay so recent contributions count more than old ones.
- Language ecosystem normalization for fair cross-language comparison.
- SQLite-backed response cache with configurable TTLs.
- YAML configuration with environment variable overrides.
- Multiple output formatters: Markdown, CLI table, JSON, and GitHub check-run.

[2.0.0]: https://github.com/2ndSetAI/good-egg/releases/tag/v2.0.0
[1.0.0]: https://github.com/2ndSetAI/good-egg/releases/tag/v1.0.0
[0.1.0]: https://github.com/2ndSetAI/good-egg/releases/tag/v0.1.0
