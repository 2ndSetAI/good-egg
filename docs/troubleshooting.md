# Troubleshooting

## Rate Limits

Good Egg retries automatically on GitHub API rate limits with exponential
backoff. If you see persistent failures:

- Use a GitHub App token instead of `GITHUB_TOKEN` for higher rate limits
  (5,000 req/hr vs 1,000).
- Reduce `fetch.max_prs` in your config to lower API usage per scored user.
  See [configuration.md](configuration.md) for details.

## Required Permissions

| Permission | Required For |
|-----------|-------------|
| `pull-requests: write` | Posting PR comments |
| `checks: write` | Creating check runs (when `check-run: true`) |

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Rate limit exhausted` | Too many API calls | Wait for reset or use App token |
| `User not found` | Deleted/renamed account | Action continues with UNKNOWN score |
| `Could not extract PR number` | Not a PR event | Ensure workflow triggers on `pull_request` |
| `Invalid GITHUB_REPOSITORY` | Malformed env var | Check Actions environment |

## Getting Help

If you encounter issues not listed here, please [open an issue](https://github.com/2ndSetAI/good-egg/issues).
