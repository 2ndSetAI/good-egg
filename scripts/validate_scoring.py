"""Validation script for Good Egg trust scoring methodology.

Discovers recent PR authors from popular repos, scores them, and produces
aggregate statistics to validate the scoring methodology.

Usage:
    # Small proof-of-concept (~6 users)
    GITHUB_TOKEN=$(gh auth token) uv run python scripts/validate_scoring.py \
        --sample-size 3 --repos langchain-ai/langchain block/goose

    # Full validation (~50-90 users)
    GITHUB_TOKEN=$(gh auth token) uv run python scripts/validate_scoring.py \
        --sample-size 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPOS: dict[str, dict[str, str]] = {
    "langchain-ai/langchain": {"language": "Python", "ecosystem": "AI/ML"},
    "pytorch/pytorch": {"language": "Python", "ecosystem": "AI/ML"},
    "vllm-project/vllm": {"language": "Python", "ecosystem": "AI/ML"},
    "huggingface/transformers": {"language": "Python", "ecosystem": "AI/ML"},
    "denoland/deno": {"language": "Rust", "ecosystem": "Runtime"},
    "astral-sh/ruff": {"language": "Rust", "ecosystem": "Tooling"},
    "kubernetes/kubernetes": {"language": "Go", "ecosystem": "Infra"},
    "vercel/next.js": {"language": "JavaScript", "ecosystem": "Web"},
    "block/goose": {"language": "Rust", "ecosystem": "AI/ML"},
}


def fetch_pr_authors(repo: str, limit: int) -> list[str]:
    """Fetch recent open PR authors from a repo using gh CLI."""
    try:
        result = subprocess.run(
            [
                "gh", "pr", "list",
                "--repo", repo,
                "--state", "open",
                "--limit", str(limit * 3),  # over-fetch to account for bots
                "--json", "author",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"  Warning: gh pr list failed for {repo}: {result.stderr.strip()}")
            return []

        prs = json.loads(result.stdout)
        seen: set[str] = set()
        authors: list[str] = []
        for pr in prs:
            login = pr.get("author", {}).get("login", "")
            if not login or login in seen:
                continue
            # Skip obvious bots
            if "[bot]" in login or login.endswith("-bot"):
                continue
            seen.add(login)
            authors.append(login)
            if len(authors) >= limit:
                break
        return authors

    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        print(f"  Warning: Could not fetch PRs for {repo}: {exc}")
        return []


async def score_all_authors(
    authors_by_repo: dict[str, list[str]], token: str
) -> list[dict[str, Any]]:
    """Score all authors using a single shared client with built-in retry."""
    from good_egg.config import load_config
    from good_egg.github_client import GitHubClient
    from good_egg.scorer import TrustScorer

    results: list[dict[str, Any]] = []
    total = sum(len(a) for a in authors_by_repo.values())
    done = 0

    config = load_config()
    scorer = TrustScorer(config)

    async with GitHubClient(token=token, config=config) as client:
        for repo, authors in authors_by_repo.items():
            owner, name = repo.split("/", 1)
            for login in authors:
                done += 1
                print(f"  [{done}/{total}] Scoring {login} against {repo}...")
                try:
                    user_data = await client.get_user_contribution_data(
                        login, context_repo=repo
                    )
                    result = scorer.score(user_data, repo)
                    results.append({
                        "login": login,
                        "context_repo": repo,
                        "normalized_score": result.normalized_score,
                        "trust_level": result.trust_level.value,
                        "account_age_days": result.account_age_days,
                        "total_merged_prs": result.total_merged_prs,
                        "unique_repos": result.unique_repos_contributed,
                        "language_match": result.language_match,
                        "flags": result.flags,
                        "raw_score": result.raw_score,
                    })
                except Exception as exc:
                    print(f"  Error scoring {login} against {repo}: {exc}")

    return results


def analyze_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate statistics and correlations."""
    if not results:
        return {"error": "No results to analyze"}

    scores = [r["normalized_score"] for r in results]
    n = len(scores)
    mean_score = sum(scores) / n
    variance = sum((s - mean_score) ** 2 for s in scores) / n
    std_dev = variance ** 0.5
    cv = std_dev / mean_score if mean_score > 0 else 0.0

    # Trust level distribution
    level_counts: dict[str, int] = {}
    for r in results:
        level = r["trust_level"]
        level_counts[level] = level_counts.get(level, 0) + 1

    # Language match impact
    match_scores = [r["normalized_score"] for r in results if r["language_match"]]
    no_match_scores = [r["normalized_score"] for r in results if not r["language_match"]]
    mean_match = sum(match_scores) / len(match_scores) if match_scores else 0.0
    mean_no_match = sum(no_match_scores) / len(no_match_scores) if no_match_scores else 0.0

    # Correlations (Pearson)
    def pearson(xs: list[float], ys: list[float]) -> float | None:
        if len(xs) < 3:
            return None
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / n
        sx = (sum((x - mx) ** 2 for x in xs) / n) ** 0.5
        sy = (sum((y - my) ** 2 for y in ys) / n) ** 0.5
        if sx == 0 or sy == 0:
            return None
        return cov / (sx * sy)

    ages = [float(r["account_age_days"]) for r in results]
    prs = [float(r["total_merged_prs"]) for r in results]
    repos = [float(r["unique_repos"]) for r in results]

    corr_age = pearson(ages, scores)
    corr_prs = pearson(prs, scores)
    corr_repos = pearson(repos, scores)

    # Per-repo breakdown
    repo_stats: dict[str, dict[str, Any]] = {}
    for r in results:
        repo = r["context_repo"]
        if repo not in repo_stats:
            repo_stats[repo] = {"scores": [], "count": 0}
        repo_stats[repo]["scores"].append(r["normalized_score"])
        repo_stats[repo]["count"] += 1

    for _repo, stats in repo_stats.items():
        s = stats["scores"]
        stats["mean"] = sum(s) / len(s)
        stats["min"] = min(s)
        stats["max"] = max(s)

    # Surprising lows: users with many PRs but LOW trust
    surprising_lows = [
        r for r in results
        if r["trust_level"] == "LOW" and r["total_merged_prs"] > 10
    ]

    return {
        "total_scored": n,
        "mean_score": round(mean_score, 4),
        "std_dev": round(std_dev, 4),
        "coefficient_of_variation": round(cv, 4),
        "min_score": round(min(scores), 4),
        "max_score": round(max(scores), 4),
        "trust_level_distribution": level_counts,
        "language_match_impact": {
            "with_match_mean": round(mean_match, 4),
            "without_match_mean": round(mean_no_match, 4),
            "with_match_count": len(match_scores),
            "without_match_count": len(no_match_scores),
        },
        "correlations": {
            "account_age_vs_score": round(corr_age, 4) if corr_age is not None else None,
            "merged_prs_vs_score": round(corr_prs, 4) if corr_prs is not None else None,
            "unique_repos_vs_score": round(corr_repos, 4) if corr_repos is not None else None,
        },
        "per_repo": {
            repo: {
                "count": s["count"],
                "mean": round(s["mean"], 4),
                "min": round(s["min"], 4),
                "max": round(s["max"], 4),
            }
            for repo, s in repo_stats.items()
        },
        "surprising_lows": [
            {"login": r["login"], "repo": r["context_repo"], "prs": r["total_merged_prs"]}
            for r in surprising_lows
        ],
    }


def generate_markdown_report(
    analysis: dict[str, Any], results: list[dict[str, Any]]
) -> str:
    """Generate a markdown report from analysis results."""
    lines: list[str] = []
    lines.append("# Good Egg Validation Report")
    lines.append(f"\nGenerated: {datetime.now().isoformat()}")
    lines.append(f"\nTotal authors scored: **{analysis['total_scored']}**")

    # Score distribution
    lines.append("\n## Score Distribution")
    lines.append(f"- Mean: {analysis['mean_score']:.4f}")
    lines.append(f"- Std Dev: {analysis['std_dev']:.4f}")
    lines.append(f"- CV: {analysis['coefficient_of_variation']:.4f}")
    lines.append(f"- Range: [{analysis['min_score']:.4f}, {analysis['max_score']:.4f}]")

    # Trust levels
    lines.append("\n## Trust Level Distribution")
    lines.append("| Level | Count | Percentage |")
    lines.append("|-------|-------|------------|")
    for level, count in sorted(analysis["trust_level_distribution"].items()):
        pct = count / analysis["total_scored"] * 100
        lines.append(f"| {level} | {count} | {pct:.1f}% |")

    # Correlations
    lines.append("\n## Correlations with Score")
    lines.append("| Factor | Pearson r |")
    lines.append("|--------|-----------|")
    corr = analysis["correlations"]
    for name, val in corr.items():
        display_name = name.replace("_vs_score", "").replace("_", " ").title()
        lines.append(f"| {display_name} | {val if val is not None else 'N/A'} |")

    # Language match
    lang = analysis["language_match_impact"]
    lines.append("\n## Language Match Impact")
    lines.append(f"- With language match (n={lang['with_match_count']}): "
                 f"mean = {lang['with_match_mean']:.4f}")
    lines.append(f"- Without language match (n={lang['without_match_count']}): "
                 f"mean = {lang['without_match_mean']:.4f}")
    if lang["with_match_count"] > 0 and lang["without_match_count"] > 0:
        diff = lang["with_match_mean"] - lang["without_match_mean"]
        lines.append(f"- Difference: {diff:+.4f}")

    # Per-repo
    lines.append("\n## Per-Repository Breakdown")
    lines.append("| Repository | Count | Mean | Min | Max |")
    lines.append("|-----------|-------|------|-----|-----|")
    for repo, stats in sorted(analysis["per_repo"].items()):
        lines.append(
            f"| {repo} | {stats['count']} | {stats['mean']:.4f} "
            f"| {stats['min']:.4f} | {stats['max']:.4f} |"
        )

    # Surprising lows
    if analysis["surprising_lows"]:
        lines.append("\n## Surprising Low Scores")
        lines.append("Users with >10 merged PRs but LOW trust:")
        lines.append("| Login | Context Repo | Merged PRs |")
        lines.append("|-------|-------------|------------|")
        for entry in analysis["surprising_lows"]:
            lines.append(f"| {entry['login']} | {entry['repo']} | {entry['prs']} |")
    else:
        lines.append("\n## Surprising Low Scores")
        lines.append("None found - no users with >10 PRs scored LOW.")

    # Methodology quality
    lines.append("\n## Methodology Quality Indicators")
    cv = analysis["coefficient_of_variation"]
    if cv > 0.3:
        lines.append("- Score spread: GOOD (CV > 0.3, scores are well-differentiated)")
    else:
        lines.append(f"- Score spread: NEEDS REVIEW (CV = {cv:.2f}, scores may be too uniform)")

    dist = analysis["trust_level_distribution"]
    if dist.get("HIGH", 0) > 0 and dist.get("LOW", 0) > 0:
        lines.append("- Level balance: GOOD (both HIGH and LOW scores present)")
    else:
        lines.append("- Level balance: NEEDS REVIEW (missing HIGH or LOW levels)")

    # Raw data table
    lines.append("\n## Individual Results")
    lines.append("| Login | Repo | Score | Level | PRs | Repos | Age (d) | Lang Match |")
    lines.append("|-------|------|-------|-------|-----|-------|---------|------------|")
    for r in sorted(results, key=lambda x: x["normalized_score"], reverse=True):
        lines.append(
            f"| {r['login']} | {r['context_repo']} "
            f"| {r['normalized_score']:.4f} | {r['trust_level']} "
            f"| {r['total_merged_prs']} | {r['unique_repos']} "
            f"| {r['account_age_days']} | {'Yes' if r['language_match'] else 'No'} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Good Egg scoring methodology")
    parser.add_argument(
        "--sample-size", type=int, default=5,
        help="Number of PR authors to sample per repo (default: 5)",
    )
    parser.add_argument(
        "--repos", nargs="*", default=None,
        help="Specific repos to validate (default: all 9)",
    )
    parser.add_argument(
        "--output-dir", default="validation",
        help="Output directory (default: validation/)",
    )
    args = parser.parse_args()

    import os
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("Error: GITHUB_TOKEN environment variable required")
        sys.exit(1)

    repos_to_use = args.repos or list(REPOS.keys())
    for repo in repos_to_use:
        if repo not in REPOS:
            print(f"Warning: {repo} not in known repos, using defaults")
            REPOS[repo] = {"language": "Unknown", "ecosystem": "Unknown"}

    # Discover authors
    print("Discovering PR authors...")
    authors_by_repo: dict[str, list[str]] = {}
    for repo in repos_to_use:
        print(f"  Fetching from {repo}...")
        authors = fetch_pr_authors(repo, args.sample_size)
        if authors:
            authors_by_repo[repo] = authors
            print(f"    Found {len(authors)} authors: {', '.join(authors)}")
        else:
            print("    No authors found")

    total_authors = sum(len(a) for a in authors_by_repo.values())
    if total_authors == 0:
        print("Error: No authors discovered. Check gh CLI authentication.")
        sys.exit(1)

    print(f"\nScoring {total_authors} authors...")

    # Score
    results = asyncio.run(score_all_authors(authors_by_repo, token))
    print(f"\nSuccessfully scored {len(results)} / {total_authors} authors")

    # Analyze
    analysis = analyze_results(results)

    # Output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_path = output_dir / f"validation_raw_{timestamp}.json"
    raw_path.write_text(json.dumps({"results": results, "analysis": analysis}, indent=2))
    print(f"\nRaw data: {raw_path}")

    report = generate_markdown_report(analysis, results)
    report_path = output_dir / f"validation_report_{timestamp}.md"
    report_path.write_text(report)
    print(f"Report: {report_path}")

    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Authors scored: {analysis['total_scored']}")
    print(f"Mean score: {analysis['mean_score']:.4f}")
    print(f"Score range: [{analysis['min_score']:.4f}, {analysis['max_score']:.4f}]")
    print(f"CV: {analysis['coefficient_of_variation']:.4f}")
    print(f"Trust levels: {analysis['trust_level_distribution']}")
    corr = analysis["correlations"]
    if corr["merged_prs_vs_score"] is not None:
        print(f"PR count correlation: {corr['merged_prs_vs_score']:.4f}")
    print(f"Surprising lows: {len(analysis['surprising_lows'])}")


if __name__ == "__main__":
    main()
