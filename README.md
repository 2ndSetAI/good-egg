# Good Egg

<img src="assets/egg.jpg" alt="Good Egg" width="200">

Trust scoring for GitHub PR authors using graph-based ranking on contribution graphs. Good Egg analyses a contributor's merged pull requests across the GitHub ecosystem, builds a weighted contribution graph, and computes a personalised trust score to surface how established and trustworthy the author of an incoming pull request is relative to your project.

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
      - uses: 2ndSetAI/good-egg@main
        # Use @v1 after the first release has been tagged
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

# Or install directly from GitHub
pip install git+https://github.com/2ndSetAI/good-egg.git

# Score a PR author
good-egg score <username> --repo <owner/repo>

# With a GitHub token for higher rate limits
GITHUB_TOKEN=ghp_... good-egg score <username> --repo <owner/repo>
```

## How It Works

1. **Fetch** -- Retrieves the user's merged pull requests and the metadata of repositories they have contributed to via the GitHub API.
2. **Build Graph** -- Constructs a directed graph where nodes represent users and repositories, and weighted edges encode contributions. Edge weights account for recency (exponential decay) and ecosystem size (language normalization).
3. **Score** -- Runs personalised graph scoring seeded from the context repository, so contributions to related projects carry more weight.
4. **Classify** -- Normalizes the raw graph score to a 0-1 range and maps it to a trust level.

## Configuration

Create a `.good-egg.yml` in your repository root to customize thresholds, scoring parameters, and more:

```yaml
thresholds:
  high_trust: 0.7
  medium_trust: 0.3
  new_account_days: 30

graph_scoring:
  alpha: 0.85

recency:
  half_life_days: 180
```

### Configuration Reference

Full `.good-egg.yml` schema:

```yaml
graph_scoring:
  alpha: 0.85              # Damping factor (0-1)
  context_repo_weight: 0.5 # Weight for the PR's target repo
  same_language_weight: 0.3 # Weight for repos in the same language
  other_weight: 0.03       # Base weight for other repos
  diversity_scale: 0.5     # How much cross-repo diversity boosts other_weight
  volume_scale: 0.3        # How much PR volume boosts other_weight

thresholds:
  high_trust: 0.7
  medium_trust: 0.3
  new_account_days: 30

recency:
  half_life_days: 180      # Exponential decay half-life
  max_age_days: 730        # PRs older than this are ignored

fetch:
  max_prs: 500             # Maximum PRs to fetch per user
  max_repos_to_enrich: 200 # Maximum repos to fetch metadata for

cache_ttl:
  repo_metadata_hours: 168  # 7 days
  user_profile_hours: 24    # 1 day
  user_prs_hours: 336       # 14 days
```

#### Environment Variable Overrides

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

## Trust Levels

| Level | Description |
|-------|-------------|
| **HIGH** | Established contributor with a strong cross-project track record |
| **MEDIUM** | Some contribution history, but limited breadth or recency |
| **LOW** | Little to no prior contribution history -- review manually |
| **UNKNOWN** | Insufficient data to produce a meaningful score |
| **BOT** | Detected bot account (e.g. dependabot, renovate) |

### Troubleshooting

#### Rate Limits

Good Egg retries automatically on GitHub API rate limits with exponential backoff. If you see persistent failures:

- Use a GitHub App token instead of `GITHUB_TOKEN` for higher rate limits (5000 req/hr vs 1000).
- Reduce `fetch.max_prs` in your config to lower API usage per scored user.

#### Required Permissions

| Permission | Required For |
|-----------|-------------|
| `pull-requests: write` | Posting PR comments |
| `checks: write` | Creating check runs (when `check-run: true`) |

#### Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Rate limit exhausted` | Too many API calls | Wait for reset or use App token |
| `User not found` | Deleted/renamed account | Action continues with UNKNOWN score |
| `Could not extract PR number` | Not a PR event | Ensure workflow triggers on `pull_request` |
| `Invalid GITHUB_REPOSITORY` | Malformed env var | Check Actions environment |

### Badges

```markdown
[![Good Egg](https://img.shields.io/badge/trust-Good%20Egg-brightgreen)](https://github.com/2ndSetAI/good-egg)
```

## License

MIT

---

Egg image CC BY 2.0 (Flickr: renwest)
