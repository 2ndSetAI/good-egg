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

## v3 (Diet Egg) Scoring

### v3 score differs from v1 or v2

This is expected. v3 uses alltime merge rate as the sole signal, with no
graph construction. Prolific contributors who close many of their own PRs
(drafts, experiments) will have lower scores than in v1/v2 because those
closed PRs pull down their merge rate.

### "Diet Egg" appears in PR comments

PR comments use the "Diet Egg" branding when `scoring_model` is set to
`v3` (the default). This is intentional. To use an older model, set
`scoring_model: v1` or `scoring_model: v2`.

### "Fresh Egg" advisory appears

Accounts under 365 days old get a "Fresh Egg" advisory. This is
informational only and does not affect the score. It is not shown for bot
accounts or existing contributors.

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
`v2`. This is intentional and helps distinguish v2 results from v1.

## Getting Help

If you encounter issues not listed here, please [open an issue](https://github.com/2ndSetAI/good-egg/issues).
