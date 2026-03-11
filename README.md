<h1>
  <img src="https://raw.githubusercontent.com/2ndSetAI/good-egg/main/assets/egg.jpg" alt="" width="48" valign="middle">
  Good Egg
</h1>

Trust scoring for GitHub PR authors based on contribution history.

## Why

AI has made mass pull requests trivial to generate, eroding the signal that
a PR represents genuine investment. Good Egg is a data-driven answer: it
mines a contributor's existing track record across the GitHub ecosystem
instead of requiring manual vouching. See
[Methodology](https://github.com/2ndSetAI/good-egg/blob/main/docs/methodology.md) for the full approach or read the
[blog post](https://neotenyai.substack.com/p/scoring-open-source-contributors) for a higher-level overview.

## Quick Start

Try Good Egg without installing anything (requires [uv](https://docs.astral.sh/uv/)):

```bash
# Requires a GitHub personal access token
GITHUB_TOKEN=<token> uvx good-egg score <username> --repo <owner/repo>
```

This runs Good Egg in a temporary environment with no install needed.

## Installation

```bash
pip install good-egg          # Core package
pip install good-egg[mcp]     # With MCP server support
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
jobs:
  score:
    runs-on: ubuntu-latest
    steps:
      - uses: 2ndSetAI/good-egg@v0
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
```

Add `checks: write` to permissions if you enable `check-run: true`.

<details>
<summary>Example PR comment</summary>
<br>
<img src="https://raw.githubusercontent.com/2ndSetAI/good-egg/main/assets/pr-comment-screenshot.png" alt="Good Egg PR comment" width="600">
</details>

See [docs/github-action.md](https://github.com/2ndSetAI/good-egg/blob/main/docs/github-action.md) for inputs, outputs,
and advanced configuration.

## CLI

```bash
good-egg score <username> --repo <owner/repo>
good-egg score octocat --repo octocat/Hello-World --json
good-egg score octocat --repo octocat/Hello-World --verbose
good-egg cache-stats
good-egg cache-clear
good-egg --version
good-egg --help
```

## Python Library

```python
import asyncio
import os

from good_egg import score_pr_author

async def main() -> None:
    result = await score_pr_author(
        login="octocat",
        repo_owner="octocat",
        repo_name="Hello-World",
        token=os.environ["GITHUB_TOKEN"],
    )
    print(f"{result.trust_level}: {result.normalized_score:.2f}")

asyncio.run(main())
```

See [docs/library.md](https://github.com/2ndSetAI/good-egg/blob/main/docs/library.md) for full API documentation.

## MCP Server

```bash
pip install good-egg[mcp]
GITHUB_TOKEN=ghp_... good-egg-mcp
```

Add to Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "good-egg": {
      "command": "good-egg-mcp",
      "env": { "GITHUB_TOKEN": "ghp_your_token_here" }
    }
  }
}
```

See [docs/mcp-server.md](https://github.com/2ndSetAI/good-egg/blob/main/docs/mcp-server.md) for tool reference.

## Scoring Models

Good Egg supports three scoring models:

| Model | Name | Description |
|-------|------|-------------|
| `v3` | Diet Egg (default) | Alltime merge rate as sole signal |
| `v2` | Better Egg | Graph score + merge rate + account age via logistic regression |
| `v1` | Good Egg | Graph-based scoring from contribution history |

v3 is the default. To use an older model, set `scoring_model: v1` or
`scoring_model: v2` in your `.good-egg.yml`, pass `--scoring-model v1` on
the CLI, or set `scoring-model: v1` in the action input. See
[Methodology](https://github.com/2ndSetAI/good-egg/blob/main/docs/methodology.md) for how each model works.

### Fresh Egg Advisory

Accounts less than 365 days old receive a "Fresh Egg" advisory in the
output. This is informational only and does not affect the score. Fresh
accounts correlate with lower merge rates in the validation data.

## How It Works

The default v3 model (Diet Egg) scores contributors by their alltime merge
rate: merged PRs divided by total PRs (merged + closed). Older models (v1,
v2) build a weighted contribution graph and run personalized graph scoring.
See [Methodology](https://github.com/2ndSetAI/good-egg/blob/main/docs/methodology.md) for details.

## Trust Levels

| Level | Description |
|-------|-------------|
| **HIGH** | Established contributor with a strong cross-project track record |
| **MEDIUM** | Some contribution history, but limited breadth or recency |
| **LOW** | Little to no prior contribution history -- review manually |
| **UNKNOWN** | Insufficient data to produce a meaningful score |
| **BOT** | Detected bot account (e.g. dependabot, renovate) |
| **EXISTING_CONTRIBUTOR** | Author already has merged PRs in this repo -- scoring skipped |

## Configuration

```yaml
thresholds:
  high_trust: 0.7
  medium_trust: 0.3
graph_scoring:
  alpha: 0.85
```

Environment variables with the `GOOD_EGG_` prefix can override individual
settings. See [docs/configuration.md](https://github.com/2ndSetAI/good-egg/blob/main/docs/configuration.md) for the full
reference and [examples/.good-egg.yml](https://github.com/2ndSetAI/good-egg/blob/main/examples/.good-egg.yml) for a
complete example.

## Troubleshooting

See [docs/troubleshooting.md](https://github.com/2ndSetAI/good-egg/blob/main/docs/troubleshooting.md) for rate limits,
required permissions, and common errors.

## License

MIT

---

Egg image CC BY 2.0 (Flickr: renwest)
