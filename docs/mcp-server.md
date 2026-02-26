# MCP Server

Good Egg includes an MCP (Model Context Protocol) server that exposes trust
scoring tools for AI assistants such as Claude Desktop and Claude Code.

## Installation

The MCP server requires the `mcp` optional dependency:

```bash
pip install good-egg[mcp]
```

## Requirements

A `GITHUB_TOKEN` environment variable must be set with a GitHub personal
access token that has read access to public repositories.

## Running

```bash
GITHUB_TOKEN=ghp_... good-egg-mcp
```

The server uses stdio transport by default and is designed to be launched
by an AI assistant client.

## Claude Desktop Configuration

Add the following to your Claude Desktop config file
(`claude_desktop_config.json`):

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

## Claude Code Configuration

Add to your Claude Code MCP settings (`.mcp.json` in your project root
or `~/.claude/mcp.json` globally):

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

## Tool Reference

| Tool | Description |
|------|-------------|
| `score_user` | Full trust score with all metadata |
| `check_pr_author` | Compact summary: trust level, score, PR count |
| `get_trust_details` | Expanded breakdown with contributions and flags |
| `cache_stats` | Show cache statistics |
| `clear_cache` | Clear cache (optionally by category) |

### score_user

Returns the full trust score as JSON, including all fields from the
`TrustScore` model.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `username` | `string` | Yes | GitHub username to score |
| `repo` | `string` | Yes | Target repository in `owner/repo` format |
| `scoring_model` | `string` | No | Scoring model: `v1` (Good Egg, default) or `v2` (Better Egg) |
| `force_score` | `boolean` | No | Force full scoring even for known contributors (default: `false`) |

**Returns:** Full `TrustScore` JSON with all fields (user_login,
context_repo, raw_score, normalized_score, trust_level, percentile,
account_age_days, total_merged_prs, unique_repos_contributed,
top_contributions, language_match, flags, scoring_model, component_scores,
scoring_metadata). When `scoring_model` is `v2`, the response includes
`component_scores` with graph_score, merge_rate, and log_account_age.

### check_pr_author

Returns a compact summary suitable for quick checks.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `username` | `string` | Yes | GitHub username to check |
| `repo` | `string` | Yes | Target repository in `owner/repo` format |
| `scoring_model` | `string` | No | Scoring model: `v1` (Good Egg, default) or `v2` (Better Egg) |
| `force_score` | `boolean` | No | Force full scoring even for known contributors (default: `false`) |

**Returns (v1):**

```json
{
  "user_login": "octocat",
  "trust_level": "HIGH",
  "normalized_score": 0.82,
  "total_merged_prs": 47
}
```

**Returns (v2):**

```json
{
  "user_login": "octocat",
  "trust_level": "HIGH",
  "normalized_score": 0.82,
  "total_merged_prs": 47,
  "scoring_model": "v2",
  "component_scores": {
    "graph_score": 0.78,
    "merge_rate": 0.91,
    "log_account_age": 3.45
  }
}
```

### get_trust_details

Returns an expanded breakdown with contributions, flags, and metadata.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `username` | `string` | Yes | GitHub username to analyse |
| `repo` | `string` | Yes | Target repository in `owner/repo` format |
| `scoring_model` | `string` | No | Scoring model: `v1` (Good Egg, default) or `v2` (Better Egg) |
| `force_score` | `boolean` | No | Force full scoring even for known contributors (default: `false`) |

**Returns (v1):**

```json
{
  "user_login": "octocat",
  "context_repo": "octocat/Hello-World",
  "trust_level": "HIGH",
  "normalized_score": 0.82,
  "raw_score": 0.0045,
  "account_age_days": 3650,
  "total_merged_prs": 47,
  "unique_repos_contributed": 12,
  "language_match": true,
  "top_contributions": [
    {
      "repo_name": "octocat/Hello-World",
      "pr_count": 15,
      "language": "Python",
      "stars": 1200
    }
  ],
  "flags": {
    "is_bot": false,
    "is_new_account": false
  },
  "scoring_metadata": {}
}
```

**Returns (v2):**

```json
{
  "user_login": "octocat",
  "context_repo": "octocat/Hello-World",
  "trust_level": "HIGH",
  "normalized_score": 0.82,
  "raw_score": 0.0045,
  "account_age_days": 3650,
  "total_merged_prs": 47,
  "unique_repos_contributed": 12,
  "language_match": true,
  "top_contributions": [
    {
      "repo_name": "octocat/Hello-World",
      "pr_count": 15,
      "language": "Python",
      "stars": 1200
    }
  ],
  "flags": {
    "is_bot": false,
    "is_new_account": false
  },
  "scoring_model": "v2",
  "component_scores": {
    "graph_score": 0.78,
    "merge_rate": 0.91,
    "log_account_age": 3.45
  },
  "scoring_metadata": {}
}
```

### cache_stats

Returns cache entry counts, categories, and database size.

**Parameters:** None.

**Returns:**

```json
{
  "total_entries": 42,
  "active_entries": 38,
  "expired_entries": 4,
  "db_size_bytes": 16384,
  "categories": {
    "repo_metadata": 25,
    "user_profile": 8,
    "user_prs": 9
  }
}
```

### clear_cache

Clears cached data. Without a category, removes all expired entries.
With a category, removes all entries in that category.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `category` | `string` | No | Cache category to clear (e.g. `repo_metadata`) |

**Returns (no category):**

```json
{
  "expired_entries_removed": 4
}
```

**Returns (with category):**

```json
{
  "cleared_category": "repo_metadata"
}
```

## Error Handling

When a tool encounters an error, it returns a JSON object with an `error`
field instead of raising an exception:

```json
{
  "error": "Rate limit exhausted. Resets at 2025-01-15T12:00:00"
}
```

This applies to all tools. Common errors include rate limit exhaustion,
user not found, repository not found, and invalid repository format.

## Cache Behaviour

The MCP server creates a fresh cache instance per tool invocation using the
default configuration. Cache data is persisted in a local SQLite database
and shared across invocations. Cache TTLs are controlled by the
`cache_ttl` section of the Good Egg configuration.
