"""Async GitHub API client for fetching user contribution data."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime

import httpx

from good_egg.cache import Cache
from good_egg.config import GoodEggConfig, load_config
from good_egg.exceptions import GitHubAPIError, RateLimitExhaustedError, UserNotFoundError
from good_egg.formatter import COMMENT_MARKER
from good_egg.models import MergedPR, RepoMetadata, UserContributionData, UserProfile

logger = logging.getLogger(__name__)

_GITHUB_BASE_URL = "https://api.github.com"
_GITHUB_GRAPHQL_URL = f"{_GITHUB_BASE_URL}/graphql"

_COMBINED_QUERY = """
query($login: String!, $cursor: String) {
  user(login: $login) {
    login
    createdAt
    __typename
    followers { totalCount }
    repositories(first: 0) { totalCount }
    pullRequests(first: 100, states: MERGED,
                 orderBy: {field: CREATED_AT, direction: DESC},
                 after: $cursor) {
      totalCount
      pageInfo { hasNextPage endCursor }
      nodes {
        title
        mergedAt
        additions
        deletions
        changedFiles
        repository {
          nameWithOwner
          stargazerCount
          forkCount
          primaryLanguage { name }
          isArchived
          isFork
        }
      }
    }
  }
}
""".strip()

_REPO_METADATA_FRAGMENT = """
  nameWithOwner
  stargazerCount
  forkCount
  primaryLanguage { name }
  isArchived
  isFork
""".strip()


class GitHubClient:
    """Async GitHub API client for fetching user contribution data."""

    def __init__(
        self,
        token: str,
        config: GoodEggConfig | None = None,
        cache: Cache | None = None,
    ) -> None:
        self._token = token
        self._config = config if config is not None else load_config()
        self._cache = cache
        self._client = httpx.AsyncClient(
            base_url=_GITHUB_BASE_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github.v3+json",
            },
            timeout=30.0,
        )

    async def __aenter__(self) -> GitHubClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self._client.aclose()

    async def _graphql(self, query: str, variables: dict[str, object]) -> dict[str, object]:
        """Execute a GraphQL query with error and rate-limit handling.

        Raises:
            UserNotFoundError: If the ``user`` field in the response is null.
            RateLimitExhaustedError: On 403 with a rate-limit message or on 429.
            GitHubAPIError: For any other non-200 response.
        """
        response = await self._client.post(
            _GITHUB_GRAPHQL_URL,
            json={"query": query, "variables": variables},
        )

        if response.status_code in (403, 429):
            body = response.json() if response.content else {}
            message = body.get("message", "")
            if response.status_code == 429 or "rate limit" in message.lower():
                reset_header = response.headers.get("X-RateLimit-Reset")
                if reset_header:
                    reset_at = datetime.fromtimestamp(int(reset_header), tz=UTC)
                else:
                    reset_at = datetime.now(UTC)
                raise RateLimitExhaustedError(reset_at=reset_at)

        if response.status_code != 200:
            remaining = response.headers.get("X-RateLimit-Remaining")
            raise GitHubAPIError(
                message=f"GitHub API returned {response.status_code}",
                status_code=response.status_code,
                rate_limit_remaining=int(remaining) if remaining else None,
            )

        data = response.json()

        # Check for null user field in the response data (only when the
        # query explicitly requested a ``user`` field).
        if "data" in data and "user" in data["data"] and data["data"]["user"] is None:
            login = variables.get("login", "unknown")
            raise UserNotFoundError(login=str(login))

        return data  # type: ignore[return-value]

    _BOT_SUFFIX_RE = re.compile(r"(\[bot\]|-bot|_bot|-app)$", re.IGNORECASE)
    _BOT_PREFIX_RE = re.compile(
        r"^(dependabot|renovate|greenkeeper|snyk-|codecov|stale"
        r"|mergify|allcontributors|github-actions|pre-commit-ci)",
        re.IGNORECASE,
    )

    @staticmethod
    def _is_bot_login(login: str) -> bool:
        """Heuristic bot detection based on login patterns."""
        return bool(
            GitHubClient._BOT_SUFFIX_RE.search(login)
            or GitHubClient._BOT_PREFIX_RE.search(login)
        )

    async def fetch_user_profile(self, login: str) -> UserProfile:
        """Fetch a user profile via GraphQL.

        Bot accounts are detected by ``__typename``, login patterns,
        and heuristics (zero followers + zero repos + old account).
        """
        query = """
        query($login: String!) {
          user(login: $login) {
            login
            createdAt
            __typename
            followers { totalCount }
            repositories(first: 0) { totalCount }
          }
        }
        """
        result = await self._graphql(query, {"login": login})
        user = result["data"]["user"]  # type: ignore[index]

        user_login = str(user["login"])  # type: ignore[index]
        is_bot = (
            user["__typename"] == "Bot"  # type: ignore[index]
            or self._is_bot_login(user_login)
        )

        followers = user["followers"]["totalCount"]  # type: ignore[index]
        public_repos = user["repositories"]["totalCount"]  # type: ignore[index]
        created_at = datetime.fromisoformat(user["createdAt"])  # type: ignore[index]

        # Heuristic: zero followers, zero repos, account older than 1 year
        account_age = (datetime.now(UTC) - created_at).days
        is_suspected = (
            not is_bot
            and followers == 0
            and public_repos == 0
            and account_age > 365
        )

        return UserProfile(
            login=user_login,
            created_at=created_at,
            followers_count=followers,
            public_repos_count=public_repos,
            is_bot=is_bot,
            is_suspected_bot=is_suspected,
        )

    async def fetch_user_merged_prs(
        self, login: str, max_prs: int = 500
    ) -> list[MergedPR]:
        """Fetch merged PRs via GraphQL with pagination (100 per page).

        Stops when *max_prs* is reached or there are no more pages.
        """
        prs: list[MergedPR] = []
        cursor: str | None = None

        while len(prs) < max_prs:
            variables: dict[str, object] = {"login": login}
            if cursor is not None:
                variables["cursor"] = cursor

            result = await self._graphql(_COMBINED_QUERY, variables)
            user_data = result["data"]["user"]  # type: ignore[index]
            pr_connection = user_data["pullRequests"]  # type: ignore[index]

            for node in pr_connection["nodes"]:  # type: ignore[union-attr]
                if len(prs) >= max_prs:
                    break
                repo = node["repository"]  # type: ignore[index]
                prs.append(
                    MergedPR(
                        repo_name_with_owner=repo["nameWithOwner"],
                        title=node["title"],
                        merged_at=datetime.fromisoformat(node["mergedAt"]),
                        additions=node["additions"],
                        deletions=node["deletions"],
                        changed_files=node["changedFiles"],
                    )
                )

            page_info = pr_connection["pageInfo"]  # type: ignore[index]
            if not page_info["hasNextPage"] or len(prs) >= max_prs:  # type: ignore[index]
                break
            cursor = page_info["endCursor"]  # type: ignore[index]

        return prs

    async def fetch_repo_metadata_batch(
        self, repos: list[str]
    ) -> dict[str, RepoMetadata]:
        """Batch-fetch repository metadata using GraphQL aliases.

        Sends up to 50 repositories per query.  Repos that return ``null``
        (inaccessible / deleted) are silently skipped.  Results are cached
        individually when a cache is available.
        """
        all_metadata: dict[str, RepoMetadata] = {}
        to_fetch: list[str] = []

        # Check cache for each repo individually
        for repo_name in repos:
            if self._cache is not None:
                cached = self._cache.get(f"repo_metadata:{repo_name}")
                if cached is not None:
                    all_metadata[repo_name] = RepoMetadata(**cached)
                    continue
            to_fetch.append(repo_name)

        if not to_fetch:
            return all_metadata

        for batch_start in range(0, len(to_fetch), 50):
            batch = to_fetch[batch_start : batch_start + 50]
            alias_parts: list[str] = []
            alias_map: dict[str, str] = {}

            for idx, full_name in enumerate(batch):
                owner, name = full_name.split("/", 1)
                alias = f"repo_{idx}"
                alias_map[alias] = full_name
                alias_parts.append(
                    f'{alias}: repository(owner: "{owner}", name: "{name}") '
                    f"{{ {_REPO_METADATA_FRAGMENT} }}"
                )

            query = "query {\n  " + "\n  ".join(alias_parts) + "\n}"
            result = await self._graphql(query, {})
            data = result["data"]  # type: ignore[index]

            for alias, full_name in alias_map.items():
                repo_data = data.get(alias)  # type: ignore[union-attr]
                if repo_data is None:
                    continue
                primary_lang = repo_data.get("primaryLanguage")
                meta = RepoMetadata(
                    name_with_owner=repo_data["nameWithOwner"],
                    stargazer_count=repo_data["stargazerCount"],
                    fork_count=repo_data["forkCount"],
                    primary_language=primary_lang["name"] if primary_lang else None,
                    is_archived=repo_data["isArchived"],
                    is_fork=repo_data["isFork"],
                )
                all_metadata[full_name] = meta
                if self._cache is not None:
                    self._cache.set(
                        f"repo_metadata:{full_name}",
                        meta.model_dump(),
                        "repo_metadata",
                    )

        return all_metadata

    async def get_user_contribution_data(
        self, login: str, context_repo: str | None = None
    ) -> UserContributionData:
        """Main entry point: fetch profile, merged PRs, and repo metadata.

        Orchestrates:
        1. Fetch profile + first page of merged PRs in a combined query.
        2. Continue paginating PRs if needed.
        3. Collect unique repos from PRs.
        4. Batch-fetch repo metadata for repos not already seen.
        5. If *context_repo* is given and not in the collected repos,
           batch-fetch its metadata so the scorer can resolve its language.
        6. Return :class:`UserContributionData`.
        """
        # Bot app accounts (e.g. "dependabot[bot]") are not real users and
        # cannot be queried via the GitHub GraphQL ``user`` field.  Short-
        # circuit with a synthetic bot profile so the scorer can classify
        # them immediately.
        if self._is_bot_login(login):
            return UserContributionData(
                profile=UserProfile(
                    login=login,
                    created_at=datetime.now(tz=UTC),
                    is_bot=True,
                ),
            )

        # Check cache for full contribution data
        if self._cache is not None:
            cached = self._cache.get(f"contribution_data:{login}")
            if cached is not None:
                contrib_data = UserContributionData(**cached)
                # Even with cached data, ensure context repo metadata is present
                if (
                    context_repo
                    and context_repo not in contrib_data.contributed_repos
                ):
                    ctx_meta = await self.fetch_repo_metadata_batch([context_repo])
                    contrib_data.contributed_repos.update(ctx_meta)
                return contrib_data

        max_prs = self._config.fetch.max_prs

        # Step 1: combined query for profile + first page of PRs
        variables: dict[str, object] = {"login": login}
        result = await self._graphql(_COMBINED_QUERY, variables)
        user_data = result["data"]["user"]  # type: ignore[index]

        # Parse profile with enhanced bot detection
        user_login = str(user_data["login"])  # type: ignore[index]
        is_bot = (
            user_data["__typename"] == "Bot"  # type: ignore[index]
            or self._is_bot_login(user_login)
        )
        followers = user_data["followers"]["totalCount"]  # type: ignore[index]
        public_repos = user_data["repositories"]["totalCount"]  # type: ignore[index]
        created_at = datetime.fromisoformat(user_data["createdAt"])  # type: ignore[index]

        account_age = (datetime.now(UTC) - created_at).days
        is_suspected = (
            not is_bot
            and followers == 0
            and public_repos == 0
            and account_age > 365
        )

        profile = UserProfile(
            login=user_login,
            created_at=created_at,
            followers_count=followers,
            public_repos_count=public_repos,
            is_bot=is_bot,
            is_suspected_bot=is_suspected,
        )

        # Parse first page of PRs and collect inline repo metadata
        pr_connection = user_data["pullRequests"]  # type: ignore[index]
        prs: list[MergedPR] = []
        contributed_repos: dict[str, RepoMetadata] = {}

        for node in pr_connection["nodes"]:  # type: ignore[union-attr]
            if len(prs) >= max_prs:
                break
            repo = node["repository"]  # type: ignore[index]
            repo_name = repo["nameWithOwner"]
            prs.append(
                MergedPR(
                    repo_name_with_owner=repo_name,
                    title=node["title"],
                    merged_at=datetime.fromisoformat(node["mergedAt"]),
                    additions=node["additions"],
                    deletions=node["deletions"],
                    changed_files=node["changedFiles"],
                )
            )
            # Stash repo metadata from inline data
            if repo_name not in contributed_repos:
                primary_lang = repo.get("primaryLanguage")
                contributed_repos[repo_name] = RepoMetadata(
                    name_with_owner=repo_name,
                    stargazer_count=repo["stargazerCount"],
                    fork_count=repo["forkCount"],
                    primary_language=primary_lang["name"] if primary_lang else None,
                    is_archived=repo["isArchived"],
                    is_fork=repo["isFork"],
                )

        # Step 2: paginate remaining PRs
        page_info = pr_connection["pageInfo"]  # type: ignore[index]
        cursor: str | None = page_info["endCursor"]  # type: ignore[index]

        while page_info["hasNextPage"] and len(prs) < max_prs:  # type: ignore[index]
            variables = {"login": login, "cursor": cursor}
            result = await self._graphql(_COMBINED_QUERY, variables)
            user_data = result["data"]["user"]  # type: ignore[index]
            pr_connection = user_data["pullRequests"]  # type: ignore[index]

            for node in pr_connection["nodes"]:  # type: ignore[union-attr]
                if len(prs) >= max_prs:
                    break
                repo = node["repository"]  # type: ignore[index]
                repo_name = repo["nameWithOwner"]
                prs.append(
                    MergedPR(
                        repo_name_with_owner=repo_name,
                        title=node["title"],
                        merged_at=datetime.fromisoformat(node["mergedAt"]),
                        additions=node["additions"],
                        deletions=node["deletions"],
                        changed_files=node["changedFiles"],
                    )
                )
                if repo_name not in contributed_repos:
                    primary_lang = repo.get("primaryLanguage")
                    contributed_repos[repo_name] = RepoMetadata(
                        name_with_owner=repo_name,
                        stargazer_count=repo["stargazerCount"],
                        fork_count=repo["forkCount"],
                        primary_language=primary_lang["name"] if primary_lang else None,
                        is_archived=repo["isArchived"],
                        is_fork=repo["isFork"],
                    )

            page_info = pr_connection["pageInfo"]  # type: ignore[index]
            cursor = page_info["endCursor"]  # type: ignore[index]

        # Step 3 & 4: batch-fetch metadata for repos not already collected
        all_repo_names = {pr.repo_name_with_owner for pr in prs}
        missing_repos = [r for r in all_repo_names if r not in contributed_repos]
        if missing_repos:
            max_to_enrich = self._config.fetch.max_repos_to_enrich
            batch_metadata = await self.fetch_repo_metadata_batch(
                missing_repos[:max_to_enrich]
            )
            contributed_repos.update(batch_metadata)

        # Step 5: ensure context repo metadata is available for personalization
        if context_repo and context_repo not in contributed_repos:
            ctx_meta = await self.fetch_repo_metadata_batch([context_repo])
            contributed_repos.update(ctx_meta)

        contrib_data = UserContributionData(
            profile=profile,
            merged_prs=prs,
            contributed_repos=contributed_repos,
        )

        # Cache the full contribution data
        if self._cache is not None:
            self._cache.set(
                f"contribution_data:{login}",
                contrib_data.model_dump(mode="json"),
                "user_prs",
            )

        return contrib_data

    # ------------------------------------------------------------------
    # REST helpers for PR comments, check runs, etc.
    # ------------------------------------------------------------------

    async def post_pr_comment(
        self, owner: str, repo: str, pr_number: int, body: str
    ) -> dict[str, object]:
        """Post a comment on a pull request.

        Uses the Issues API endpoint:
        ``POST /repos/{owner}/{repo}/issues/{pr_number}/comments``
        """
        response = await self._client.post(
            f"/repos/{owner}/{repo}/issues/{pr_number}/comments",
            json={"body": body},
        )
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    async def update_pr_comment(
        self, owner: str, repo: str, comment_id: int, body: str
    ) -> dict[str, object]:
        """Update an existing PR comment.

        Uses the Issues API endpoint:
        ``PATCH /repos/{owner}/{repo}/issues/comments/{comment_id}``
        """
        response = await self._client.patch(
            f"/repos/{owner}/{repo}/issues/comments/{comment_id}",
            json={"body": body},
        )
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    async def find_existing_comment(
        self, owner: str, repo: str, pr_number: int
    ) -> int | None:
        """Find an existing Good Egg comment on a pull request.

        Searches for :data:`~good_egg.formatter.COMMENT_MARKER` in the body of
        each comment.  Returns the comment ID if found, ``None`` otherwise.
        """
        response = await self._client.get(
            f"/repos/{owner}/{repo}/issues/{pr_number}/comments",
        )
        response.raise_for_status()
        comments: list[dict[str, object]] = response.json()

        for comment in comments:
            body = str(comment.get("body", ""))
            if COMMENT_MARKER in body:
                return int(comment["id"])  # type: ignore[arg-type]

        return None

    async def create_check_run(
        self,
        owner: str,
        repo: str,
        head_sha: str,
        title: str,
        summary: str,
    ) -> dict[str, object]:
        """Create a GitHub Check Run.

        ``POST /repos/{owner}/{repo}/check-runs``
        """
        response = await self._client.post(
            f"/repos/{owner}/{repo}/check-runs",
            json={
                "name": "Good Egg Trust Score",
                "head_sha": head_sha,
                "status": "completed",
                "conclusion": "neutral",
                "output": {
                    "title": title,
                    "summary": summary,
                },
            },
        )
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]
