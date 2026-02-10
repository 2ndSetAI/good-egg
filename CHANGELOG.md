# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/2ndSetAI/good-egg/compare/main...HEAD
