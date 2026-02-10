<h1>
  <img src="https://raw.githubusercontent.com/2ndSetAI/good-egg/main/assets/egg.jpg" alt="" width="48" valign="middle">
  Good Egg
</h1>

Trust scoring for GitHub PR authors using graph-based analysis of
contribution history.

## Why

AI has made mass pull requests trivial to generate, eroding the signal that
a PR represents genuine investment. Good Egg is a data-driven answer: it
mines a contributor's existing track record across the GitHub ecosystem
instead of requiring manual vouching. See
[Methodology](docs/methodology.md) for the full approach.

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
<img src="assets/pr-comment-screenshot.png" alt="Good Egg PR comment" width="600">
</details>

See [docs/github-action.md](docs/github-action.md) for inputs, outputs,
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

See [docs/library.md](docs/library.md) for full API documentation.

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

See [docs/mcp-server.md](docs/mcp-server.md) for tool reference.

## How It Works

Good Egg builds a weighted contribution graph from a user's merged PRs and
runs personalized graph scoring to produce a trust score relative to your
project. See [Methodology](docs/methodology.md) for details.

## Trust Levels

| Level | Description |
|-------|-------------|
| **HIGH** | Established contributor with a strong cross-project track record |
| **MEDIUM** | Some contribution history, but limited breadth or recency |
| **LOW** | Little to no prior contribution history -- review manually |
| **UNKNOWN** | Insufficient data to produce a meaningful score |
| **BOT** | Detected bot account (e.g. dependabot, renovate) |

## Configuration

```yaml
thresholds:
  high_trust: 0.7
  medium_trust: 0.3
graph_scoring:
  alpha: 0.85
```

Environment variables with the `GOOD_EGG_` prefix can override individual
settings. See [docs/configuration.md](docs/configuration.md) for the full
reference and [examples/.good-egg.yml](examples/.good-egg.yml) for a
complete example.

## Troubleshooting

See [docs/troubleshooting.md](docs/troubleshooting.md) for rate limits,
required permissions, and common errors.

## License

MIT

---

Egg image CC BY 2.0 (Flickr: renwest)
