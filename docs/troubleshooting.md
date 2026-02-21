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

## v2 (Better Egg) Scoring

### v2 score differs significantly from v1

This is expected. The v2 model uses a simplified graph (no self-contribution
penalty, no language normalization, no diversity/volume adjustment) and
combines the graph score with merge rate and account age via logistic
regression. The final score is calibrated differently and may be higher or
lower than v1 for the same user.

### Merge rate is 0 or missing

Merge rate requires both merged and closed PR counts. If the GitHub API
returns no closed PRs (e.g. due to rate limits or data availability), merge
rate defaults to 0. This can lower the combined score. Check that your token
has sufficient rate limit remaining.

### "Better Egg" appears in PR comments

PR comments use the "Better Egg" branding when `scoring_model` is set to
`v2`. This is intentional and helps distinguish v2 results from v1. To
switch back, set `scoring_model: v1` or remove the setting (v1 is the
default).

## Getting Help

If you encounter issues not listed here, please [open an issue](https://github.com/2ndSetAI/good-egg/issues).
