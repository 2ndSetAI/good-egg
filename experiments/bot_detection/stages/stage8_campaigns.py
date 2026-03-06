from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.bot_detection.cache import BotDetectionDB
from experiments.bot_detection.checkpoint import write_json, write_stage_checkpoint
from experiments.bot_detection.models import StudyConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Monthly rejection rate anomaly detection
# ---------------------------------------------------------------------------

def _compute_monthly_rejection_rates(
    db: BotDetectionDB,
) -> pd.DataFrame:
    """Compute monthly rejection rate per repo.

    Returns a DataFrame with columns: repo, month, n_prs, n_rejected, rejection_rate.
    """
    df = db.con.execute("""
        SELECT
            repo,
            DATE_TRUNC('month', created_at) AS month,
            COUNT(*) AS n_prs,
            SUM(CASE WHEN state != 'MERGED' AND merged_at IS NULL THEN 1 ELSE 0 END)
                AS n_rejected
        FROM prs
        GROUP BY repo, DATE_TRUNC('month', created_at)
        HAVING COUNT(*) >= 5
        ORDER BY repo, month
    """).fetchdf()

    if df.empty:
        df["rejection_rate"] = pd.Series(dtype=float)
        return df

    df["rejection_rate"] = df["n_rejected"] / df["n_prs"]
    return df


def _flag_anomalous_months(
    monthly_df: pd.DataFrame,
    n_stdev: float = 2.0,
) -> pd.DataFrame:
    """Flag months with rejection rate > n_stdev above per-repo mean.

    Adds columns: repo_mean, repo_std, is_anomalous.
    """
    if monthly_df.empty:
        monthly_df["repo_mean"] = pd.Series(dtype=float)
        monthly_df["repo_std"] = pd.Series(dtype=float)
        monthly_df["is_anomalous"] = pd.Series(dtype=bool)
        return monthly_df

    stats = monthly_df.groupby("repo")["rejection_rate"].agg(["mean", "std"])
    stats.columns = ["repo_mean", "repo_std"]
    stats["repo_std"] = stats["repo_std"].fillna(0.0)

    merged = monthly_df.merge(stats, on="repo", how="left")
    merged["is_anomalous"] = (
        merged["rejection_rate"] > merged["repo_mean"] + n_stdev * merged["repo_std"]
    )
    return merged


# ---------------------------------------------------------------------------
# Campaign author identification
# ---------------------------------------------------------------------------

def _find_campaign_authors(
    db: BotDetectionDB,
    anomalous_months: pd.DataFrame,
) -> pd.DataFrame:
    """Find authors who only appear during anomalous months with 0% merge rate.

    Returns a DataFrame with columns: login, repo, month, n_prs, merge_rate.
    """
    if anomalous_months.empty:
        return pd.DataFrame(columns=["login", "repo", "month", "n_prs", "merge_rate"])

    flagged = anomalous_months[anomalous_months["is_anomalous"]]
    if flagged.empty:
        return pd.DataFrame(columns=["login", "repo", "month", "n_prs", "merge_rate"])

    # Get all PRs, compute per-author-per-month stats
    all_prs = db.con.execute("""
        SELECT
            author AS login,
            repo,
            DATE_TRUNC('month', created_at) AS month,
            COUNT(*) AS n_prs,
            COALESCE(SUM(CASE WHEN state = 'MERGED' THEN 1 ELSE 0 END)::DOUBLE
                / NULLIF(COUNT(*), 0), 0) AS merge_rate
        FROM prs
        GROUP BY author, repo, DATE_TRUNC('month', created_at)
    """).fetchdf()

    if all_prs.empty:
        return pd.DataFrame(columns=["login", "repo", "month", "n_prs", "merge_rate"])

    # Which (repo, month) pairs are anomalous
    flagged_keys = set(
        zip(flagged["repo"], flagged["month"], strict=True)
    )

    # Authors who appear during anomalous months
    all_prs["in_anomalous"] = [
        (r, m) in flagged_keys
        for r, m in zip(all_prs["repo"], all_prs["month"], strict=True)
    ]

    # Authors who ONLY appear during anomalous months (for each repo)
    author_any_normal = all_prs[~all_prs["in_anomalous"]].groupby("login").size()
    anomalous_only = set(all_prs["login"]) - set(author_any_normal.index)

    # Filter to those with 0% merge rate
    campaign_df = all_prs[
        all_prs["login"].isin(anomalous_only)
        & all_prs["in_anomalous"]
        & (all_prs["merge_rate"] == 0.0)
    ][["login", "repo", "month", "n_prs", "merge_rate"]].copy()

    return campaign_df


# ---------------------------------------------------------------------------
# Hacktoberfest pattern check
# ---------------------------------------------------------------------------

def _check_october_pattern(
    db: BotDetectionDB,
) -> dict[str, Any]:
    """Check October months for elevated rejection rates (Hacktoberfest)."""
    df = db.con.execute("""
        SELECT
            EXTRACT(YEAR FROM created_at) AS year,
            EXTRACT(MONTH FROM created_at) AS month,
            COUNT(*) AS n_prs,
            SUM(CASE WHEN state = 'MERGED' THEN 1 ELSE 0 END) AS n_merged,
            SUM(CASE WHEN state != 'MERGED' AND merged_at IS NULL THEN 1 ELSE 0 END)
                AS n_rejected,
            COUNT(DISTINCT author) AS n_authors,
            COUNT(DISTINCT repo) AS n_repos
        FROM prs
        GROUP BY EXTRACT(YEAR FROM created_at), EXTRACT(MONTH FROM created_at)
        ORDER BY year, month
    """).fetchdf()

    if df.empty:
        return {"october_months": [], "non_october_avg_rejection_rate": None}

    df["rejection_rate"] = df["n_rejected"] / df["n_prs"]

    october = df[df["month"] == 10]
    non_october = df[df["month"] != 10]

    non_oct_avg = (
        float(non_october["rejection_rate"].mean())
        if not non_october.empty else None
    )

    october_rows = []
    for _, row in october.iterrows():
        october_rows.append({
            "year": int(row["year"]),
            "n_prs": int(row["n_prs"]),
            "n_merged": int(row["n_merged"]),
            "n_rejected": int(row["n_rejected"]),
            "rejection_rate": float(row["rejection_rate"]),
            "n_authors": int(row["n_authors"]),
            "n_repos": int(row["n_repos"]),
        })

    return {
        "october_months": october_rows,
        "non_october_avg_rejection_rate": non_oct_avg,
    }


# ---------------------------------------------------------------------------
# Cross-reference with suspended accounts
# ---------------------------------------------------------------------------

def _cross_reference_suspended(
    campaign_authors: pd.DataFrame,
    db: BotDetectionDB,
) -> dict[str, Any]:
    """Check how many campaign authors have suspended accounts."""
    if campaign_authors.empty:
        return {
            "n_campaign_authors": 0,
            "n_suspended": 0,
            "n_active": 0,
            "n_unknown": 0,
            "suspended_logins": [],
        }

    logins = campaign_authors["login"].unique().tolist()
    if not logins:
        return {
            "n_campaign_authors": 0,
            "n_suspended": 0,
            "n_active": 0,
            "n_unknown": 0,
            "suspended_logins": [],
        }

    placeholders = ", ".join(["?"] * len(logins))
    status_df = db.con.execute(
        f"SELECT login, account_status FROM authors WHERE login IN ({placeholders})",
        logins,
    ).fetchdf()

    status_map = dict(
        zip(status_df["login"], status_df["account_status"], strict=True)
    ) if not status_df.empty else {}

    suspended = [login for login in logins if status_map.get(login) == "suspended"]
    active = [login for login in logins if status_map.get(login) == "active"]
    unknown = [
        login for login in logins
        if login not in status_map or status_map[login] is None
    ]

    return {
        "n_campaign_authors": len(logins),
        "n_suspended": len(suspended),
        "n_active": len(active),
        "n_unknown": len(unknown),
        "suspended_logins": suspended,
    }


# ---------------------------------------------------------------------------
# Named campaign windows
# ---------------------------------------------------------------------------

def _check_campaign_windows(
    db: BotDetectionDB,
    windows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Analyze named campaign windows from config."""
    results = []
    for window in windows:
        name = window.get("name", "unnamed")
        start = window.get("start")
        end = window.get("end")
        if not start or not end:
            continue

        df = db.con.execute("""
            SELECT
                COUNT(*) AS n_prs,
                SUM(CASE WHEN state = 'MERGED' THEN 1 ELSE 0 END) AS n_merged,
                SUM(CASE WHEN state != 'MERGED' AND merged_at IS NULL
                    THEN 1 ELSE 0 END) AS n_rejected,
                COUNT(DISTINCT author) AS n_authors,
                COUNT(DISTINCT repo) AS n_repos
            FROM prs
            WHERE created_at >= ? AND created_at < ?
        """, [start, end]).fetchdf()

        if df.empty or df["n_prs"].iloc[0] == 0:
            results.append({"name": name, "n_prs": 0})
            continue

        row = df.iloc[0]
        n_prs = int(row["n_prs"])
        results.append({
            "name": name,
            "start": start,
            "end": end,
            "n_prs": n_prs,
            "n_merged": int(row["n_merged"]),
            "n_rejected": int(row["n_rejected"]),
            "rejection_rate": float(row["n_rejected"]) / n_prs if n_prs else 0.0,
            "n_authors": int(row["n_authors"]),
            "n_repos": int(row["n_repos"]),
        })

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_stage8(base_dir: Path, config: StudyConfig) -> dict[str, Any]:
    """Detect coordinated spam campaigns from temporal patterns."""
    db_path = base_dir / config.paths.get("local_db", "data/bot_detection.duckdb")
    results_dir = base_dir / config.paths.get("results", "data/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    author_cfg = config.author_analysis
    campaign_windows = author_cfg.get("campaign_windows", [])

    results: dict[str, Any] = {}

    with BotDetectionDB(db_path) as db:
        # Monthly rejection rate analysis
        logger.info("Computing monthly rejection rates...")
        monthly = _compute_monthly_rejection_rates(db)
        flagged = _flag_anomalous_months(monthly)
        n_anomalous = int(flagged["is_anomalous"].sum()) if not flagged.empty else 0
        logger.info("Found %d anomalous repo-months", n_anomalous)

        results["monthly_analysis"] = {
            "n_repo_months": len(monthly),
            "n_anomalous": n_anomalous,
        }

        # Campaign authors
        logger.info("Identifying campaign authors...")
        campaign_authors = _find_campaign_authors(db, flagged)
        n_campaign = len(campaign_authors["login"].unique()) if not campaign_authors.empty else 0
        logger.info("Found %d campaign authors", n_campaign)

        results["campaign_authors"] = {
            "n_authors": n_campaign,
            "n_prs": int(campaign_authors["n_prs"].sum()) if not campaign_authors.empty else 0,
        }

        # Cross-reference with suspended accounts
        logger.info("Cross-referencing with account status...")
        xref = _cross_reference_suspended(campaign_authors, db)
        results["suspended_cross_reference"] = xref

        # October / Hacktoberfest pattern
        logger.info("Checking October pattern...")
        results["october_pattern"] = _check_october_pattern(db)

        # Named campaign windows
        if campaign_windows:
            logger.info("Analyzing %d campaign windows...", len(campaign_windows))
            results["campaign_windows"] = _check_campaign_windows(db, campaign_windows)

    # Write results
    output_path = results_dir / "campaign_results.json"
    write_json(output_path, results)
    logger.info("Campaign results written to %s", output_path)

    # Checkpoint
    write_stage_checkpoint(
        base_dir / "data",
        "stage8",
        {"anomalous_months": n_anomalous, "campaign_authors": n_campaign},
        details={
            "suspended_in_campaigns": xref["n_suspended"],
            "campaign_windows_checked": len(campaign_windows),
        },
    )

    return results
