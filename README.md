# Good Egg

![Good Egg](assets/egg.jpg)

Trust scoring for GitHub PR authors using PageRank on contribution graphs. Good Egg analyses a contributor's merged pull requests across the GitHub ecosystem, builds a weighted contribution graph, and computes a personalised PageRank score to surface how established and trustworthy the author of an incoming pull request is relative to your project.

## Quick Start

Add Good Egg to any pull request workflow:

```yaml
name: Good Egg

on:
  pull_request:
    types: [opened, reopened, synchronize]

permissions:
  pull-requests: write
  checks: write

jobs:
  score:
    runs-on: ubuntu-latest
    steps:
      - uses: good-egg/good-egg@v1
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
```

### Action Inputs

| Input | Description | Default |
|-------|-------------|---------|
| `github-token` | GitHub token for API access | `${{ github.token }}` |
| `config-path` | Path to `.good-egg.yml` config file | _(auto-detected)_ |
| `comment` | Post a PR comment with the trust score | `true` |
| `check-run` | Create a check run with the trust score | `false` |
| `fail-on-low` | Fail the action if trust level is LOW | `false` |

### Action Outputs

| Output | Description |
|--------|-------------|
| `score` | Normalized trust score (0.0 - 1.0) |
| `trust-level` | Trust level: HIGH, MEDIUM, LOW, UNKNOWN, or BOT |
| `user` | GitHub username that was scored |

## CLI Usage

```bash
# Install
pip install good-egg

# Score a PR author
good-egg score <username> --repo <owner/repo>

# With a GitHub token for higher rate limits
GITHUB_TOKEN=ghp_... good-egg score <username> --repo <owner/repo>
```

## How It Works

1. **Fetch** -- Retrieves the user's merged pull requests and the metadata of repositories they have contributed to via the GitHub API.
2. **Build Graph** -- Constructs a directed graph where nodes represent users and repositories, and weighted edges encode contributions. Edge weights account for recency (exponential decay) and ecosystem size (language normalization).
3. **PageRank** -- Runs personalised PageRank seeded from the context repository, so contributions to related projects carry more weight.
4. **Classify** -- Normalizes the raw PageRank score to a 0-1 range and maps it to a trust level.

## Configuration

Create a `.good-egg.yml` in your repository root to customize thresholds, PageRank parameters, and more:

```yaml
thresholds:
  high_trust: 0.7
  medium_trust: 0.3
  new_account_days: 30

pagerank:
  alpha: 0.85

recency:
  half_life_days: 180
```

## Trust Levels

| Level | Description |
|-------|-------------|
| **HIGH** | Established contributor with a strong cross-project track record |
| **MEDIUM** | Some contribution history, but limited breadth or recency |
| **LOW** | Little to no prior contribution history -- review manually |
| **UNKNOWN** | Insufficient data to produce a meaningful score |
| **BOT** | Detected bot account (e.g. dependabot, renovate) |

## License

MIT

---

Egg image CC BY 2.0 (Flickr: renwest)
