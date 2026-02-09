# Good Egg -- Project Instructions

## Overview

Good Egg is a trust scoring tool for GitHub PR authors. It builds a weighted contribution graph from a user's merged PRs and computes a personalised trust score to assess how established a contributor is relative to a given project. It runs as a GitHub Action and as a CLI.

## Tech Stack

- **Language**: Python 3.12+
- **Package manager**: uv
- **Framework**: Pydantic for config/models, Click for CLI, httpx for async HTTP, NetworkX for graph scoring
- **Testing**: pytest (with pytest-asyncio, respx for HTTP mocking, pytest-cov)
- **Linting**: ruff
- **Type checking**: mypy (strict)

## Project Structure

```
src/good_egg/       # Main package
  config.py         # Pydantic config models, YAML + env var loading
  models.py         # Data models (UserProfile, MergedPR, TrustScore, etc.)
  github_client.py  # Async GitHub GraphQL/REST client with retry
  graph_builder.py  # Bipartite trust graph construction
  scorer.py         # Graph-based trust scoring engine
  formatter.py      # Output formatters (Markdown, CLI, JSON, check-run)
  action.py         # GitHub Action entry point
  cli.py            # Click CLI
  cache.py          # SQLite-backed response cache
  exceptions.py     # Custom exception hierarchy
tests/              # pytest test suite (mirrors src/ structure)
scripts/            # Utility scripts (validation, language multiplier generation)
examples/           # Example GitHub Actions workflows
```

## Commands

### Development

```bash
uv run pytest --cov=good_egg -v     # Run tests with coverage
uv run ruff check src/ tests/ scripts/  # Lint
uv run mypy src/good_egg/           # Type check
uv run ruff check --fix src/ tests/ scripts/  # Auto-fix lint issues
```

### Running

```bash
good-egg score <username> --repo <owner/repo>   # Score a user
good-egg cache-stats                             # Cache statistics
good-egg cache-clear                             # Clear cache
```

## Code Conventions

- Use `from __future__ import annotations` in every module.
- Type annotations required on all function signatures.
- Line length: 99 characters (enforced by ruff).
- Ruff rule set: E, F, I, N, W, UP, B, A, SIM.
- Tests go in `tests/` with `test_` prefix matching the source module.
- Prefer editing existing files over creating new ones.
- Keep changes minimal and focused -- don't refactor surrounding code unless asked.

## Configuration

- Config class is `GoodEggConfig` in `config.py`, composed of sub-configs: `GraphScoringConfig`, `EdgeWeightConfig`, `RecencyConfig`, `ThresholdConfig`, `CacheTTLConfig`, `LanguageNormalization`, `FetchConfig`.
- YAML config key for scoring parameters is `graph_scoring` (not "pagerank").
- Environment variable overrides use `GOOD_EGG_` prefix.

## Important Rules

- **No AI attribution**: Do not add `Co-Authored-By`, `Signed-off-by`, or any other trailer attributing AI/Claude to commits. Do not sign PR descriptions, comments, or code comments as Claude or any AI assistant. Commits should be attributed solely to the human author.
- **No "PageRank" branding**: The project uses NetworkX's `nx.pagerank()` internally, but all user-facing names, docs, config keys, and class names use "graph scoring" instead. Do not introduce "PageRank" into any user-facing surface.
- **Test before committing**: Always run `uv run pytest --cov=good_egg -v` and `uv run ruff check src/ tests/ scripts/` before considering work complete.
- **Coverage threshold**: Maintain >= 90% test coverage.
