# Good Egg

<img src="https://raw.githubusercontent.com/2ndSetAI/good-egg/main/assets/egg.jpg" alt="Good Egg" width="200">

Trust scoring for GitHub PR authors using graph-based ranking on contribution
graphs. Good Egg analyses a contributor's merged pull requests across the
GitHub ecosystem, builds a weighted contribution graph, and computes a
personalised trust score to surface how established a contributor is relative
to your project.

Good Egg runs as a **GitHub Action**, a **CLI tool**, a **Python library**,
and an **MCP server** for AI assistant integration.

## Installation

```bash
pip install good-egg
```

To use the MCP server for AI assistant integration:

```bash
pip install good-egg[mcp]
```

## GitHub Action

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
      - uses: 2ndSetAI/good-egg@v0
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

See [docs/github-action.md](docs/github-action.md) for advanced usage,
custom configuration, and using outputs in downstream steps.

## CLI

```bash
# Score a PR author
good-egg score <username> --repo <owner/repo>

# With a GitHub token for higher rate limits
GITHUB_TOKEN=ghp_... good-egg score octocat --repo octocat/Hello-World

# JSON output
good-egg score octocat --repo octocat/Hello-World --json

# Verbose output with contribution details
good-egg score octocat --repo octocat/Hello-World --verbose
```

### Additional Commands

```bash
good-egg cache-stats                 # Show cache statistics
good-egg cache-clear                 # Remove expired cache entries
good-egg cache-clear --category repo_metadata  # Clear specific category
good-egg --version                   # Print version
good-egg --help                      # Show help
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (invalid input, API failure, missing token) |

## Python Library

```python
import asyncio
import os
from good_egg import score_pr_author

async def main():
    result = await score_pr_author(
        login="octocat",
        repo_owner="octocat",
        repo_name="Hello-World",
        token=os.environ["GITHUB_TOKEN"],
    )
    print(f"Trust level: {result.trust_level}")
    print(f"Score: {result.normalized_score:.2f}")

asyncio.run(main())
```

See [docs/library.md](docs/library.md) for full API documentation, custom
configuration, error handling, and cache usage.

## MCP Server

Good Egg includes an MCP (Model Context Protocol) server for integration
with AI assistants like Claude.

```bash
pip install good-egg[mcp]
GITHUB_TOKEN=ghp_... good-egg-mcp
```

Add to your Claude Desktop configuration (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "good-egg": {
      "command": "good-egg-mcp",
      "env": {
        "GITHUB_TOKEN": "ghp_your_token_here"
      }
    }
  }
}
```

See [docs/mcp-server.md](docs/mcp-server.md) for tool reference and
Claude Code configuration.

## How It Works

1. **Fetch** -- Retrieves the user's merged pull requests and the metadata
   of repositories they have contributed to via the GitHub API.
2. **Build Graph** -- Constructs a directed graph where nodes represent
   users and repositories, and weighted edges encode contributions. Edge
   weights account for recency (exponential decay) and ecosystem size
   (language normalization).
3. **Score** -- Runs personalised graph scoring seeded from the context
   repository, so contributions to related projects carry more weight.
4. **Classify** -- Normalizes the raw graph score to a 0-1 range and maps
   it to a trust level.

## Trust Levels

| Level | Description |
|-------|-------------|
| **HIGH** | Established contributor with a strong cross-project track record |
| **MEDIUM** | Some contribution history, but limited breadth or recency |
| **LOW** | Little to no prior contribution history -- review manually |
| **UNKNOWN** | Insufficient data to produce a meaningful score |
| **BOT** | Detected bot account (e.g. dependabot, renovate) |

## Configuration

Create a `.good-egg.yml` in your repository root to customize thresholds,
scoring parameters, and more:

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

Environment variables with the `GOOD_EGG_` prefix can override individual
settings. See [docs/configuration.md](docs/configuration.md) for the full
reference, and [examples/.good-egg.yml](examples/.good-egg.yml) for a
complete example config file with all defaults.

## Troubleshooting

### Rate Limits

Good Egg retries automatically on GitHub API rate limits with exponential
backoff. If you see persistent failures:

- Use a GitHub App token instead of `GITHUB_TOKEN` for higher rate limits
  (5000 req/hr vs 1000).
- Reduce `fetch.max_prs` in your config to lower API usage per scored user.

### Required Permissions

| Permission | Required For |
|-----------|-------------|
| `pull-requests: write` | Posting PR comments |
| `checks: write` | Creating check runs (when `check-run: true`) |

### Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Rate limit exhausted` | Too many API calls | Wait for reset or use App token |
| `User not found` | Deleted/renamed account | Action continues with UNKNOWN score |
| `Could not extract PR number` | Not a PR event | Ensure workflow triggers on `pull_request` |
| `Invalid GITHUB_REPOSITORY` | Malformed env var | Check Actions environment |

## License

MIT

---

Egg image CC BY 2.0 (Flickr: renwest)
