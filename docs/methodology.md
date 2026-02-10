# Methodology

## The Problem

AI has eliminated the natural barrier to entry for open-source contributions. Generating a plausible pull request now takes seconds, not hours. The result: contribution volume is up, but signal-to-noise is down. Projects can no longer assume that a pull request represents genuine investment in the codebase.

## Approaches to Trust

### Explicit Vouching

Mitchell Hashimoto's [Vouch](https://github.com/mitchellh/vouch) system takes the direct approach: maintainers manually vouch for contributors they trust. This creates a web-of-trust where established participants validate newcomers.

**Strengths**: High-signal, rooted in human judgment, works well for tight-knit communities.

**Weaknesses**: Manual effort that doesn't scale, cold-start problem for new projects with no vouch network, and requires maintainers to actively participate in a separate system.

### Behavioral Mining

Good Egg takes a different approach: instead of asking maintainers to do extra work, mine the contribution data that already exists.

The core insight is that good open-source contributors are *already exhibiting good behavior* across the ecosystem. They have merged PRs in established projects, sustained contributions over time, and worked across multiple repositories. This existing track record is a strong signal -- and it's freely available through the GitHub API.

Good Egg is automated, data-driven, and complements rather than replaces human review. It answers one specific question: *"Is this person an established contributor?"*

## How Scoring Works

### Data Collection

Good Egg fetches a user's merged pull requests via the GitHub GraphQL API, along with metadata for each repository they've contributed to (stars, language, fork status, archived status).

### Graph Construction

The scoring engine builds a **bipartite directed graph** with two node types:

- **User nodes** (`user:{login}`) -- the contributor being scored
- **Repository nodes** (`repo:{owner/name}`) -- repositories they've contributed to

Each merged PR creates a weighted edge from the user node to the repository node. Edge weights combine recency and quality:

```
edge_weight = recency_decay x repo_quality x edge_type_weight
```

The `edge_type_weight` for merged PRs is 1.0 (configurable via `edge_weights.merged_pr`).

**Anti-gaming measures**:
- Self-contributions (PRs to your own repos) are penalized at 0.3x weight
- PRs per repository are capped at 20 to prevent inflation from a single project
- Reverse edges (repo -> user) are added at 0.3x the forward weight

### Recency Decay

Recent contributions matter more than old ones. Good Egg applies exponential decay:

```
decay = exp(-0.693 x days_ago / half_life_days)
```

The default half-life is 180 days. Contributions older than `max_age_days` (default: 730 days) are excluded entirely.

### Repository Quality

Not all repositories are equal. Quality is computed as:

```
quality = log1p(stars x language_multiplier)
```

Penalties are applied for:
- **Archived repositories**: 0.5x (project is no longer active)
- **Forks**: 0.3x (lower signal of independent project quality)

### Language Normalization

Star counts vary enormously across ecosystems -- a 1,000-star Rust library represents a very different level of adoption than a 1,000-star JavaScript package. Good Egg applies static ecosystem-size multipliers to normalize:

| Language | Multiplier | Rationale |
|----------|-----------|-----------|
| JavaScript | 1.00 | Baseline (largest ecosystem) |
| Python | 1.13 | |
| Go | 2.30 | |
| Rust | 2.63 | |
| Zig | 5.44 | Niche; fewer repos, lower star counts |

These are selected examples from the full 26-language multiplier table (see `LanguageNormalization` in `config.py`). Multipliers are derived from relative ecosystem sizes on GitHub. Contributions to niche ecosystems are weighted higher because they represent rarer, harder work.

### Personalization Vector

Graph scoring uses a personalization (restart) vector to bias the random walk toward the context repository. The raw weights before normalization:

| Node type | Raw weight | Purpose |
|-----------|------------|---------|
| Context repository | 0.50 | Strongest signal: contributions to *this* project |
| Same-language repos | 0.30 | Ecosystem relevance |
| Other repos | 0.03 | Baseline (adjusted for contributor diversity and volume) |
| User nodes | 0.00 | Users don't seed the walk |

These raw weights are normalized to sum to 1.0, so actual values in the random walk depend on graph composition. The "other repos" weight is dynamically adjusted based on contributor diversity (number of unique repos) and volume (total PRs), so prolific cross-ecosystem contributors aren't unfairly penalized.

### Scoring and Normalization

The directed graph is scored using personalized graph-based ranking with a damping factor (alpha) of 0.85. This produces a raw score for the user node.

Normalization converts the raw score to a 0-1 range:

```
baseline = 1 / num_nodes
ratio = raw_score / baseline
normalized = ratio / (ratio + 1)
```

This sigmoid-like mapping means a score equal to the uniform baseline maps to 0.5, with diminishing returns above.

### Classification

| Level | Threshold | Meaning |
|-------|-----------|---------|
| **HIGH** | >= 0.70 | Established contributor with strong cross-project history |
| **MEDIUM** | >= 0.30 | Some history, limited breadth or recency |
| **LOW** | < 0.30 | Little to no prior contribution history |
| **UNKNOWN** | -- | Insufficient data (no merged PRs found) |
| **BOT** | -- | Detected bot account |

## Known Limitations

### New Contributor Cold Start

Good Egg measures *established* contribution history by design. A genuinely talented developer making their first open-source contribution will score LOW -- and that's correct behavior, not a bug. The tool answers "is this person an established contributor?", not "is this person trustworthy?".

Different tools are needed for discovering promising new contributors. Good Egg is a point solution for one specific question.

### Language Normalization is Approximate

Static multipliers don't capture project relatedness beyond language. A contributor to `tokio` (Rust async runtime) is probably more relevant to a Rust web framework than a contributor to a Rust game engine, but Good Egg treats both equally.

A possible extension: graph-based relatedness, where shared contributors between projects create implicit edges indicating project similarity.

### API Constraints

GitHub rate limits bound how much data can be fetched per user. Good Egg is designed to be cheap to compute within these constraints -- typically 2-4 API calls per scored user. This means the scoring graph is necessarily incomplete, built from the most recent and most visible contributions rather than a complete history.

## Possible Extensions

- **Graph-based project relatedness**: Use shared contributors between projects as edges in a project similarity graph, replacing or supplementing language-only normalization.
- **Review and issue activity**: PRs aren't the only signal. Code reviews and issue participation indicate engagement patterns.
- **Organization membership**: Membership in established GitHub organizations as an additional trust signal.
- **Cross-platform data**: Contribution data from GitLab, Codeberg, and other forges to build a more complete picture.
