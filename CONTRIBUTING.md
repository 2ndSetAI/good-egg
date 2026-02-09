# Contributing to Good Egg

## Setup

```bash
# Clone
git clone https://github.com/2ndSetAI/good-egg.git
cd good-egg

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

## Project Structure

```
good-egg/
├── src/good_egg/
│   ├── action.py         # GitHub Action entry point
│   ├── cache.py          # SQLite-backed response cache
│   ├── cli.py            # Click CLI (good-egg score, cache-stats, cache-clear)
│   ├── config.py         # YAML + env var configuration
│   ├── exceptions.py     # Custom exception hierarchy
│   ├── formatter.py      # Markdown, CLI, JSON, check-run formatters
│   ├── github_client.py  # Async GitHub GraphQL/REST client with retry
│   ├── graph_builder.py  # Bipartite trust graph construction
│   ├── models.py         # Pydantic data models
│   └── scorer.py         # Graph-based trust scoring engine
├── tests/                # pytest test suite
├── scripts/
│   └── validate_scoring.py  # Validation against real repos
└── examples/             # Example GitHub Actions workflows
```

## Development Workflow

```bash
# Run tests
uv run pytest --cov=good_egg -v

# Lint
uv run ruff check src/ tests/ scripts/

# Type check
uv run mypy src/good_egg/

# Format (auto-fix)
uv run ruff check --fix src/ tests/ scripts/
```

## Code Style

- Ruff enforces style (E, F, I, N, W, UP, B, A, SIM rules)
- Line length: 99 characters
- Type annotations required on all function signatures (mypy strict)
- Use `from __future__ import annotations` in every module

## Running the Validation Suite

The validation script scores real PR authors from popular repos to check scoring methodology:

```bash
GITHUB_TOKEN=$(gh auth token) uv run python scripts/validate_scoring.py --sample-size 5
```

This requires `gh` CLI authenticated. Results are written to the `validation/` directory.

## Pull Requests

1. Create a feature branch from `main`
2. Make your changes with tests
3. Ensure `uv run pytest --cov=good_egg -v` passes with >= 90% coverage
4. Ensure `uv run ruff check src/ tests/ scripts/` is clean
5. Open a PR with a clear description
