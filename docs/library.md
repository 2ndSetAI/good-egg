# Python Library Usage

Good Egg can be used as a Python library to integrate trust scoring into
your own applications, bots, or CI pipelines.

## Prerequisites

- Python 3.12 or later
- A GitHub personal access token (classic or fine-grained) with read access
  to public repositories

## Installation

```bash
pip install good-egg
```

## Basic Usage

The main entry point is the `score_pr_author` async function:

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
    print(f"User: {result.user_login}")
    print(f"Trust level: {result.trust_level}")
    print(f"Score: {result.normalized_score:.2f}")
    print(f"Merged PRs: {result.total_merged_prs}")
    print(f"Unique repos: {result.unique_repos_contributed}")

asyncio.run(main())
```

### Function Signature

```python
async def score_pr_author(
    login: str,
    repo_owner: str,
    repo_name: str,
    config: GoodEggConfig | None = None,
    token: str | None = None,
    cache: object | None = None,
) -> TrustScore:
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `login` | `str` | GitHub username to score |
| `repo_owner` | `str` | Owner of the context repository |
| `repo_name` | `str` | Name of the context repository |
| `config` | `GoodEggConfig \| None` | Custom configuration; defaults are used when `None` |
| `token` | `str \| None` | GitHub API token; falls back to `GITHUB_TOKEN` env var |
| `cache` | `object \| None` | `Cache` instance for response caching (see [Cache Usage](#cache-usage)) |

## Custom Configuration

Pass a `GoodEggConfig` to customize scoring behaviour:

```python
from good_egg import GoodEggConfig, score_pr_author

config = GoodEggConfig(
    thresholds={"high_trust": 0.8, "medium_trust": 0.4},
    graph_scoring={"alpha": 0.9},
    recency={"half_life_days": 90},
)

result = await score_pr_author(
    login="octocat",
    repo_owner="octocat",
    repo_name="Hello-World",
    config=config,
)
```

### v2 (Better Egg) Configuration

To use the v2 scoring model, set `scoring_model` on the config:

```python
from good_egg import GoodEggConfig, score_pr_author

config = GoodEggConfig(
    scoring_model="v2",
    v2={
        "graph": {"half_life_days": 180, "max_age_days": 730},
        "features": {"merge_rate": True, "account_age": True},
        "combined_model": {
            "intercept": -0.8094,
            "graph_score_weight": 1.9138,
            "merge_rate_weight": -0.7783,
            "account_age_weight": 0.1493,
        },
    },
)

result = await score_pr_author(
    login="octocat",
    repo_owner="octocat",
    repo_name="Hello-World",
    config=config,
)

# v2 results include component scores
if result.component_scores:
    print(f"Graph score: {result.component_scores['graph_score']:.3f}")
    print(f"Merge rate: {result.component_scores['merge_rate']:.3f}")
    print(f"Log account age: {result.component_scores['log_account_age']:.3f}")
    print(f"Normalized score: {result.normalized_score:.3f}")

print(f"Scoring model: {result.scoring_model}")
```

You can also load configuration from a YAML file:

```python
from good_egg.config import load_config
from good_egg import score_pr_author

config = load_config(".good-egg.yml")
result = await score_pr_author(
    login="octocat",
    repo_owner="octocat",
    repo_name="Hello-World",
    config=config,
)
```

## Return Type: TrustScore

The `score_pr_author` function returns a `TrustScore` Pydantic model with
the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `user_login` | `str` | GitHub username that was scored |
| `context_repo` | `str` | Repository used as scoring context |
| `raw_score` | `float` | Raw graph score before normalization |
| `normalized_score` | `float` | Normalized score (0.0 - 1.0) |
| `trust_level` | `TrustLevel` | HIGH, MEDIUM, LOW, UNKNOWN, or BOT |
| `percentile` | `float` | Percentile rank (0.0 - 1.0) |
| `account_age_days` | `int` | Age of the GitHub account in days |
| `total_merged_prs` | `int` | Total number of merged pull requests |
| `unique_repos_contributed` | `int` | Number of distinct repositories |
| `top_contributions` | `list[ContributionSummary]` | Top repositories contributed to |
| `language_match` | `bool` | Whether the user's top language matches the context repo |
| `flags` | `dict[str, bool]` | Flags (is_bot, is_new_account, etc.) |
| `scoring_model` | `str` | Scoring model used: `v1` or `v2` |
| `component_scores` | `dict[str, float] \| None` | Component breakdown (v2 only): `graph_score`, `merge_rate`, `log_account_age` |
| `scoring_metadata` | `dict[str, Any]` | Internal scoring details |

`TrustScore` is a Pydantic model, so you can serialize it:

```python
# To dict
data = result.model_dump()

# To JSON string
json_str = result.model_dump_json()
```

## Cache Usage

Pass a `Cache` instance to avoid redundant GitHub API calls across
multiple scoring operations:

```python
from good_egg.cache import Cache
from good_egg.config import load_config
from good_egg import score_pr_author

config = load_config()
cache = Cache(ttls=config.cache_ttl.to_seconds())

try:
    result = await score_pr_author(
        login="octocat",
        repo_owner="octocat",
        repo_name="Hello-World",
        config=config,
        cache=cache,
    )
finally:
    cache.close()
```

The cache is backed by SQLite and persists between runs. Cache TTLs are
configured in the `cache_ttl` section of the configuration file.

## Error Handling

Good Egg defines a hierarchy of exceptions in `good_egg.exceptions`:

```python
from good_egg.exceptions import (
    GoodEggError,
    GitHubAPIError,
    RateLimitExhaustedError,
    UserNotFoundError,
    RepoNotFoundError,
    CacheError,
    ConfigError,
    InsufficientDataError,
)
```

### Exception Hierarchy

```
GoodEggError (base)
  GitHubAPIError (status_code, rate_limit_remaining)
    RateLimitExhaustedError (reset_at)
    UserNotFoundError (login)
    RepoNotFoundError (repo)
  CacheError
  ConfigError
  InsufficientDataError
```

### Example

```python
from good_egg import score_pr_author
from good_egg.exceptions import (
    RateLimitExhaustedError,
    UserNotFoundError,
)

try:
    result = await score_pr_author(
        login="octocat",
        repo_owner="octocat",
        repo_name="Hello-World",
    )
except UserNotFoundError as exc:
    print(f"User {exc.login} not found")
except RateLimitExhaustedError as exc:
    print(f"Rate limited until {exc.reset_at.isoformat()}")
except GoodEggError as exc:
    print(f"Scoring failed: {exc}")
```

## Async Patterns

`score_pr_author` is an async function. If you are calling it from
synchronous code, use `asyncio.run()`:

```python
import asyncio
from good_egg import score_pr_author

result = asyncio.run(
    score_pr_author(
        login="octocat",
        repo_owner="octocat",
        repo_name="Hello-World",
    )
)
```

If you already have a running event loop (e.g. inside a web framework),
call it directly with `await`:

```python
result = await score_pr_author(
    login="octocat",
    repo_owner="octocat",
    repo_name="Hello-World",
)
```
