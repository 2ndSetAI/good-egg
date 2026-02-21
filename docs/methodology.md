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

---

## Better Egg (v2)

The v2 scoring model -- branded "Better Egg" in PR comments -- extends the
graph-based approach with external features combined via logistic regression.
It is opt-in via `scoring_model: v2` in configuration; v1 remains the default.

### Motivation

The v1 graph score relies entirely on contribution-graph structure. Two
observable signals sit outside that graph:

1. **Merge rate** -- the fraction of a user's PRs that were merged vs closed.
   The v1 data pipeline only fetches *merged* PRs, creating survivorship bias.
   Merge rate re-introduces the rejected-PR signal.
2. **Account age** -- how long the GitHub account has existed. Older accounts
   correlate with established contributors and help with cold-start cases
   where few merged PRs are available.

A third candidate, **text dissimilarity** (comparing PR descriptions to
repository README content), was investigated but not implemented. The signal
was inverted -- higher similarity correlated with lower trust -- likely
because low-effort PRs tend to parrot project language while experienced
contributors write more targeted descriptions.

### Simplified Graph

The v2 model uses a simplified graph construction compared to v1:

- **No self-contribution penalty** -- PRs to your own repos are weighted
  equally.
- **No language normalization in repo quality** -- star counts are used
  directly without ecosystem-size multipliers.
- **No `same_language_weight`** -- the personalization vector does not boost
  same-language repositories.
- **No diversity/volume adjustment** -- the "other repos" weight is static,
  not dynamically adjusted.

These simplifications reduce the number of tunable parameters and let the
logistic regression handle signal combination instead.

### External Features

| Feature | Formula | Range |
|---------|---------|-------|
| Merge rate | `merged / (merged + closed)` | 0.0 -- 1.0 |
| Account age | `log(account_age_days + 1)` | 0.0 -- ~4.3 |

Both features are computed from data already available through the GitHub API
at no additional cost.

### Combined Model

The three components -- graph score, merge rate, and log account age -- are
combined via logistic regression:

```
p = sigmoid(intercept + w1 * graph_score + w2 * merge_rate + w3 * log_account_age)
```

The sigmoid function maps the linear combination to a 0-1 probability, which
is used as the final normalized score. The model weights were trained on the
validation dataset (see below).

### Validation

The v2 model was trained and evaluated on 5,129 PRs drawn from 49
repositories in a validation study.

| Metric | v1 (graph only) | v2 (combined) |
|--------|-----------------|---------------|
| AUC | 0.647 | 0.647 |

The AUC difference is not statistically significant -- the combined model
does not improve ranking performance over the graph alone. However, the
individual features carry statistically significant information:

- **Merge rate**: Likelihood Ratio Test p < 10^-12
- **Account age**: Likelihood Ratio Test p = 1.2 x 10^-5

### Why Include Features That Don't Improve AUC?

The merge rate and account age features are retained despite the flat AUC
because they address structural limitations of the graph-only model:

- **Merge rate** corrects survivorship bias. The v1 pipeline only sees merged
  PRs, so a user with 10 merged and 90 closed looks identical to one with 10
  merged and 0 closed. Merge rate distinguishes them.
- **Account age** provides signal in cold-start scenarios where the graph has
  few edges. A 10-year-old account with 2 merged PRs is qualitatively
  different from a 2-day-old account with 2 merged PRs.

Both features carry real information (confirmed by likelihood ratio tests);
the AUC flatness reflects the fact that graph structure already captures most
of the ranking signal in the validation dataset.

### Component Score Breakdown

v2 output includes a component-level breakdown showing the individual
contribution of each feature:

| Component | Description |
|-----------|-------------|
| `graph_score` | Normalized graph-based trust score (same as v1 output) |
| `merge_rate` | Fraction of PRs that were merged |
| `log_account_age` | Log-transformed account age in days |
| `combined_score` | Final logistic regression output |

This transparency lets users understand which factors are driving the overall
score.
