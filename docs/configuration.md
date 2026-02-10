# Configuration Reference

Good Egg is configured via a YAML file, environment variables, or
programmatically through the `GoodEggConfig` class.

## Configuration File

Place a `.good-egg.yml` file in your repository root. The GitHub Action
auto-detects this file; for the CLI, pass `--config` to the `score`
subcommand:

```bash
good-egg score <username> --repo owner/name --config .good-egg.yml
```

See [examples/.good-egg.yml](../examples/.good-egg.yml) for a complete
example with all defaults.

## Config Precedence

Configuration values are resolved in this order (highest priority first):

1. **CLI arguments** (e.g. `--token`)
2. **Environment variables** (e.g. `GOOD_EGG_ALPHA`)
3. **YAML config file**
4. **Built-in defaults**

## Full YAML Schema

```yaml
# Graph-based scoring algorithm parameters
graph_scoring:
  alpha: 0.85              # Damping factor (0-1)
  max_iterations: 100      # Maximum iterations for convergence
  tolerance: 0.000001      # Convergence tolerance
  context_repo_weight: 0.5 # Weight for the PR's target repo
  same_language_weight: 0.3  # Weight for same-language repos
  other_weight: 0.03       # Base weight for other repos
  diversity_scale: 0.5     # Cross-repo diversity boost
  volume_scale: 0.3        # PR volume boost

# Trust level thresholds
thresholds:
  high_trust: 0.7          # Score >= this is HIGH trust
  medium_trust: 0.3        # Score >= this is MEDIUM (below = LOW)
  new_account_days: 30     # Accounts younger than this are flagged

# Recency decay parameters
recency:
  half_life_days: 180      # Half-life for exponential decay
  max_age_days: 730        # Ignore PRs older than this

# Edge weight multipliers for contribution types
edge_weights:
  merged_pr: 1.0           # Merged pull request
  maintainer: 2.0          # Maintainer/owner relationship
  star: 0.1                # Starring a repository
  review: 0.5              # Reviewing a pull request

# GitHub API fetch parameters
fetch:
  max_prs: 500             # Max merged PRs to fetch per user
  max_repos_to_enrich: 200 # Max repos to fetch metadata for
  rate_limit_safety_margin: 100  # Stop when this many calls remain

# Cache time-to-live settings (in hours)
cache_ttl:
  repo_metadata_hours: 168 # 7 days
  user_profile_hours: 24   # 1 day
  user_prs_hours: 336      # 14 days

# Language ecosystem size normalization
# Smaller ecosystems get higher multipliers so niche contributions
# are valued appropriately.
language_normalization:
  default: 3.0             # Multiplier for unlisted languages
  multipliers:
    JavaScript: 1.0
    Python: 1.13
    TypeScript: 1.30
    Java: 1.55
    Go: 2.30
    Rust: 2.63
    # ... see examples/.good-egg.yml for the full list
```

## Config Sections

### graph_scoring

Controls the graph-based scoring algorithm. The `alpha` parameter is the
damping factor -- higher values give more weight to the structure of the
contribution graph. The `context_repo_weight` and `same_language_weight`
parameters control how much the context repository and same-language
repositories influence the score.

### thresholds

Defines the boundaries between trust levels. A normalized score at or
above `high_trust` maps to HIGH, at or above `medium_trust` maps to MEDIUM,
and below that maps to LOW. Accounts younger than `new_account_days` are
flagged.

### recency

Controls how recent contributions are weighted relative to older ones.
`half_life_days` sets the exponential decay half-life; contributions at
that age carry half the weight of new ones. `max_age_days` sets a hard
cutoff -- older PRs are ignored entirely.

### edge_weights

Multipliers for different types of contributions. These affect the edge
weights in the contribution graph.

### fetch

Controls how much data is retrieved from the GitHub API per user. Reducing
`max_prs` lowers API usage at the cost of less data for scoring.

### cache_ttl

Time-to-live for cached GitHub API responses. The cache avoids refetching
data that has not changed.

### language_normalization

Adjusts contribution weights by language ecosystem size. Languages with
smaller ecosystems (fewer repositories on GitHub) get higher multipliers so
that contributions to niche projects are not undervalued.

## Environment Variable Overrides

The following environment variables override individual config values:

| Variable | Config Path | Type |
|----------|-------------|------|
| `GOOD_EGG_ALPHA` | `graph_scoring.alpha` | float |
| `GOOD_EGG_OTHER_WEIGHT` | `graph_scoring.other_weight` | float |
| `GOOD_EGG_DIVERSITY_SCALE` | `graph_scoring.diversity_scale` | float |
| `GOOD_EGG_VOLUME_SCALE` | `graph_scoring.volume_scale` | float |
| `GOOD_EGG_MAX_PRS` | `fetch.max_prs` | int |
| `GOOD_EGG_HIGH_TRUST` | `thresholds.high_trust` | float |
| `GOOD_EGG_MEDIUM_TRUST` | `thresholds.medium_trust` | float |
| `GOOD_EGG_HALF_LIFE_DAYS` | `recency.half_life_days` | int |

## Programmatic Configuration

In Python, create a `GoodEggConfig` directly:

```python
from good_egg import GoodEggConfig

config = GoodEggConfig(
    thresholds={"high_trust": 0.8, "medium_trust": 0.4},
    graph_scoring={"alpha": 0.9},
)
```

Or load from a YAML file:

```python
from good_egg.config import load_config

config = load_config(".good-egg.yml")
```

The `GoodEggConfig` class is composed of the following sub-configs:
`GraphScoringConfig`, `EdgeWeightConfig`, `RecencyConfig`,
`ThresholdConfig`, `CacheTTLConfig`, `LanguageNormalization`, and
`FetchConfig`.
