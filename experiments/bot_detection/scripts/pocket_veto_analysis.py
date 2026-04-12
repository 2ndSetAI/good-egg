"""Pocket-veto investigation for issue #51.

Analyze whether counting stale open PRs as implicit rejections meaningfully
shifts merge-rate distributions and improves the signal's ability to separate
suspended from active GitHub accounts.

Operates entirely on the existing bot_detection DuckDB. Does not fetch from
GitHub. Does not modify src/good_egg/.

Outputs:
  - experiments/bot_detection/data/results/pocket_veto_analysis.json
  - experiments/bot_detection/pocket_veto_findings.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[1]
DB_PATH = BASE / "data" / "bot_detection.duckdb"
ACTIVITY_PATH = BASE / "data" / "open_pr_activity.parquet"
RESULTS_PATH = BASE / "data" / "results" / "pocket_veto_analysis.json"
FINDINGS_PATH = BASE / "pocket_veto_findings.md"

UNIVERSAL_THRESHOLD_DAYS = 90
SENSITIVITY_THRESHOLDS = (30, 60, 90, 180)
SEED = 42

MERGE_RATE_COLS = (
    "merge_rate_v3",
    "merge_rate_universal",
    "merge_rate_per_repo",
    "merge_rate_idle_universal",
    "merge_rate_idle_per_repo",
)


def characterize(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Phase 1: sanity-check the existing outcome/stale_threshold_days columns."""
    totals = con.execute("""
        SELECT state, COUNT(*) FROM prs GROUP BY state ORDER BY state
    """).fetchall()
    outcomes = con.execute("""
        SELECT outcome, COUNT(*) FROM prs GROUP BY outcome ORDER BY outcome
    """).fetchall()
    state_outcome = con.execute("""
        SELECT state, outcome, COUNT(*) FROM prs
        GROUP BY state, outcome ORDER BY state, outcome
    """).fetchall()

    thresh_dist = con.execute("""
        SELECT stale_threshold_days, COUNT(*) AS n
        FROM prs WHERE stale_threshold_days IS NOT NULL
        GROUP BY stale_threshold_days ORDER BY n DESC
    """).fetchall()

    # Per-repo: how does stored stale_threshold_days compare to 2x median
    # time-to-close (the hypothesis in the issue)?
    repo_stats = con.execute("""
        WITH closed AS (
            SELECT repo,
                   EXTRACT(EPOCH FROM (closed_at - created_at))/86400.0 AS ttc_days
            FROM prs
            WHERE state IN ('MERGED', 'CLOSED')
              AND closed_at IS NOT NULL
              AND created_at IS NOT NULL
              AND closed_at > created_at
        ),
        repo_ttc AS (
            SELECT repo, MEDIAN(ttc_days) AS median_ttc_days, COUNT(*) AS n_closed
            FROM closed GROUP BY repo
        ),
        repo_thresh AS (
            SELECT repo,
                   ANY_VALUE(stale_threshold_days) AS stale_threshold_days,
                   COUNT(DISTINCT stale_threshold_days) AS distinct_thresh
            FROM prs
            WHERE stale_threshold_days IS NOT NULL
            GROUP BY repo
        )
        SELECT t.repo, t.stale_threshold_days, t.distinct_thresh,
               r.median_ttc_days, r.n_closed
        FROM repo_thresh t LEFT JOIN repo_ttc r USING (repo)
        ORDER BY t.repo
    """).fetchdf()

    repo_stats["two_x_median"] = 2.0 * repo_stats["median_ttc_days"]
    repo_stats["delta_vs_2x"] = (
        repo_stats["stale_threshold_days"] - repo_stats["two_x_median"]
    )

    # Open PR age distribution (relative to the DB's max created_at as "now")
    age_stats = con.execute("""
        WITH ref AS (SELECT MAX(created_at) AS now FROM prs)
        SELECT
            quantile_cont(
                EXTRACT(EPOCH FROM (ref.now - p.created_at))/86400.0,
                [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
            ) AS quantiles
        FROM prs p, ref WHERE p.state = 'OPEN'
    """).fetchone()

    return {
        "state_totals": dict(totals),
        "outcome_totals": dict(outcomes),
        "state_outcome_crosstab": [
            {"state": s, "outcome": o, "n": n} for s, o, n in state_outcome
        ],
        "stale_threshold_distribution": [
            {"threshold_days": float(t), "n": int(n)} for t, n in thresh_dist
        ],
        "repos_total": int(len(repo_stats)),
        "repos_using_default_30d": int(
            (repo_stats["stale_threshold_days"] == 30.0).sum()
        ),
        "repos_calibrated": int(
            (repo_stats["stale_threshold_days"] != 30.0).sum()
        ),
        "per_repo_calibration_check": {
            "mean_delta_vs_2x_median_ttc": float(
                repo_stats["delta_vs_2x"].dropna().mean()
            ),
            "median_delta_vs_2x_median_ttc": float(
                repo_stats["delta_vs_2x"].dropna().median()
            ),
            "n_repos_with_closed_prs": int(
                repo_stats["median_ttc_days"].notna().sum()
            ),
        },
        "open_pr_age_quantiles_days": {
            label: float(v) for label, v in zip(
                ["p10", "p25", "p50", "p75", "p90", "p95"],
                age_stats[0],
                strict=True,
            )
        },
    }


def build_author_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Phase 2: per-author counts and all five merge-rate definitions.

    Age-based variants use (now - created_at) with 'now' = max(created_at)
    in the DB. Idle-based variants use (fetch_now - updated_at) from the
    open_pr_activity.parquet sidecar; PRs that were OPEN in the DB snapshot
    but are no longer currently open are treated as non-stale (the close or
    merge event since the snapshot is itself evidence of activity).
    """
    has_activity = ACTIVITY_PATH.exists()
    sensitivity_cols = ",\n            ".join(
        f"SUM(CASE WHEN state='OPEN' AND age_days > {d} "
        f"THEN 1 ELSE 0 END) AS open_stale_{d}d"
        for d in SENSITIVITY_THRESHOLDS
    )
    activity_join = ""
    idle_cols = (
        "0 AS open_stale_idle_universal, 0 AS open_stale_idle_per_repo,"
    )
    if has_activity:
        activity_join = f"""
            LEFT JOIN read_parquet('{ACTIVITY_PATH}') oa
              ON oa.repo = aged.repo AND oa.number = aged.number
        """
        idle_cols = f"""
            SUM(CASE
                WHEN state='OPEN' AND oa.updated_at IS NOT NULL
                  AND EXTRACT(EPOCH FROM (oa.fetch_now - oa.updated_at))
                      /86400.0 > {UNIVERSAL_THRESHOLD_DAYS}
                THEN 1 ELSE 0
            END) AS open_stale_idle_universal,
            SUM(CASE
                WHEN state='OPEN' AND oa.updated_at IS NOT NULL
                  AND EXTRACT(EPOCH FROM (oa.fetch_now - oa.updated_at))
                      /86400.0 > COALESCE(stale_threshold_days, 30)
                THEN 1 ELSE 0
            END) AS open_stale_idle_per_repo,
        """
    query = f"""
        WITH ref AS (SELECT MAX(created_at) AS now FROM prs),
        aged AS (
            SELECT p.*,
                   EXTRACT(EPOCH FROM (ref.now - p.created_at))/86400.0 AS age_days
            FROM prs p, ref
        )
        SELECT
            author AS login,
            COUNT(*) AS total_prs,
            SUM(CASE WHEN state='MERGED' THEN 1 ELSE 0 END) AS merged,
            SUM(CASE WHEN state='CLOSED' THEN 1 ELSE 0 END) AS closed,
            SUM(CASE WHEN state='OPEN' THEN 1 ELSE 0 END) AS open_total,
            SUM(CASE WHEN state='OPEN'
                     AND age_days > COALESCE(stale_threshold_days, 30)
                     THEN 1 ELSE 0 END) AS open_stale_per_repo,
            SUM(CASE WHEN state='OPEN' AND age_days > {UNIVERSAL_THRESHOLD_DAYS}
                     THEN 1 ELSE 0 END) AS open_stale_universal,
            {idle_cols}
            {sensitivity_cols},
            MEDIAN(additions) AS median_additions
        FROM aged
        {activity_join}
        GROUP BY author
    """
    df = con.execute(query).fetchdf()

    def compute(col: str, stale_col: str) -> None:
        denom = df["merged"] + df["closed"] + df[stale_col]
        df[col] = np.where(denom > 0, df["merged"] / denom, 0.0)

    df["merge_rate_v3"] = np.where(
        (df["merged"] + df["closed"]) > 0,
        df["merged"] / (df["merged"] + df["closed"]),
        0.0,
    )
    compute("merge_rate_universal", "open_stale_universal")
    compute("merge_rate_per_repo", "open_stale_per_repo")
    compute("merge_rate_idle_universal", "open_stale_idle_universal")
    compute("merge_rate_idle_per_repo", "open_stale_idle_per_repo")
    for d in SENSITIVITY_THRESHOLDS:
        compute(f"merge_rate_universal_{d}d", f"open_stale_{d}d")
    return df


def distribution_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Phase 2 deliverable: summary stats for each merge-rate definition."""

    def summarize(col: str) -> dict[str, float]:
        s = df[col]
        return {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "p10": float(s.quantile(0.10)),
            "p25": float(s.quantile(0.25)),
            "p75": float(s.quantile(0.75)),
            "p90": float(s.quantile(0.90)),
        }

    cols = [
        *MERGE_RATE_COLS,
        *[f"merge_rate_universal_{d}d" for d in SENSITIVITY_THRESHOLDS],
    ]
    return {col: summarize(col) for col in cols}


def shift_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """Phase 3: per-author shift from v3 baseline to each alternative."""
    out: dict[str, Any] = {"n_authors": int(len(df))}
    for alt in [c for c in MERGE_RATE_COLS if c != "merge_rate_v3"]:
        delta = df[alt] - df["merge_rate_v3"]
        out[alt] = {
            "mean_delta": float(delta.mean()),
            "median_delta": float(delta.median()),
            "n_dropped_gt_0.05": int((delta < -0.05).sum()),
            "n_dropped_gt_0.10": int((delta < -0.10).sum()),
            "n_dropped_gt_0.25": int((delta < -0.25).sum()),
            "n_unchanged": int((delta == 0).sum()),
        }
    return out


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    if len(group_a) < 2 or len(group_b) < 2:
        return float("nan")
    pooled = np.sqrt(
        ((len(group_a) - 1) * group_a.var(ddof=1)
         + (len(group_b) - 1) * group_b.var(ddof=1))
        / (len(group_a) + len(group_b) - 2)
    )
    if pooled == 0:
        return float("nan")
    return float((group_a.mean() - group_b.mean()) / pooled)


def cv_auc(
    df: pd.DataFrame,
    merge_rate_col: str,
    n_folds: int = 5,
) -> dict[str, float]:
    """Phase 4: minimal 2-feature LR CV — merge_rate variant + median_additions.

    Mirrors the 2-feature baseline in scripts/refit_bad_egg.py but swaps the
    merge-rate column so all three definitions are evaluated on identical
    labeled-author splits.
    """
    y = (df["account_status"] == "suspended").astype(int).values
    mr = df[merge_rate_col].fillna(0).to_numpy(dtype=float)
    ma = df["median_additions"].fillna(0).to_numpy(dtype=float)
    ma = np.log1p(np.abs(ma)) * np.sign(ma)
    x = np.column_stack([mr, ma])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    fold_aucs: list[float] = []
    for train_idx, test_idx in skf.split(x, y):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])
        model = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=SEED,
        )
        model.fit(x_train, y[train_idx])
        probs = model.predict_proba(x_test)[:, 1]
        oof[test_idx] = probs
        fold_aucs.append(roc_auc_score(y[test_idx], probs))

    mr_susp = mr[y == 1]
    mr_act = mr[y == 0]
    return {
        "cv_auc": float(roc_auc_score(y, oof)),
        "fold_auc_std": float(np.std(fold_aucs)),
        "mean_merge_rate_suspended": float(mr_susp.mean()),
        "mean_merge_rate_active": float(mr_act.mean()),
        "cohens_d_active_vs_suspended": cohens_d(mr_act, mr_susp),
        "n_suspended": int(y.sum()),
        "n_active": int((1 - y).sum()),
    }


def signal_evaluation(
    df: pd.DataFrame, con: duckdb.DuckDBPyConnection,
) -> dict[str, Any]:
    """Phase 4 deliverable: run CV for each merge-rate definition."""
    authors = con.execute(
        "SELECT login, account_status FROM authors"
    ).fetchdf()
    labeled = df.merge(authors, on="login", how="inner")
    labeled = labeled[labeled["account_status"].isin(["active", "suspended"])]
    labeled = labeled.copy()

    results: dict[str, Any] = {
        "n_labeled": int(len(labeled)),
        "n_suspended": int((labeled["account_status"] == "suspended").sum()),
        "n_active": int((labeled["account_status"] == "active").sum()),
    }
    for col in MERGE_RATE_COLS:
        results[col] = cv_auc(labeled, col)
    return results


def pick_example_authors(
    df: pd.DataFrame, con: duckdb.DuckDBPyConnection,
) -> list[dict[str, Any]]:
    """Phase 3 deliverable: handful of authors where definitions disagree."""
    authors = con.execute(
        "SELECT login, account_status FROM authors"
    ).fetchdf()
    merged = df.merge(authors, on="login", how="left")
    merged["shift_universal"] = (
        merged["merge_rate_universal"] - merged["merge_rate_v3"]
    )
    # 3 big-shift authors (most affected) + 3 suspended authors + 2 active
    # high-PR authors.
    big_shift = merged.nsmallest(3, "shift_universal")
    susp = merged[merged["account_status"] == "suspended"].nlargest(
        3, "total_prs",
    )
    act = merged[merged["account_status"] == "active"].nlargest(
        2, "total_prs",
    )
    picks = pd.concat([big_shift, susp, act]).drop_duplicates(subset=["login"])
    cols = [
        "login", "account_status", "total_prs", "merged", "closed",
        "open_total", "open_stale_per_repo", "open_stale_universal",
        "open_stale_idle_universal", "open_stale_idle_per_repo",
        *MERGE_RATE_COLS,
    ]
    return picks[cols].to_dict("records")


def write_findings(results: dict[str, Any]) -> None:
    c = results["characterization"]
    d = results["distributions"]
    s = results["shift_analysis"]
    e = results["signal_evaluation"]

    def fmt(x: float) -> str:
        return f"{x:.4f}"

    lines = [
        "# Pocket Veto Investigation — Findings",
        "",
        "Investigation for issue #51. Does counting stale open PRs as implicit",
        "rejections meaningfully change merge-rate distributions and improve the",
        "signal's ability to separate suspended from active accounts?",
        "",
        "## Dataset",
        "",
        f"- {sum(c['state_totals'].values())} PRs across "
        f"{c['repos_total']} repos",
        f"- State totals: {c['state_totals']}",
        f"- Outcome totals: {c['outcome_totals']}",
        f"- Labeled authors: {e['n_labeled']} "
        f"({e['n_suspended']} suspended, {e['n_active']} active)",
        "",
        "## Staleness definitions compared",
        "",
        "- **v3 (baseline)**: `merged / (merged + closed)` — current scorer.py.",
        f"- **age_universal**: open PR is stale if age > "
        f"{UNIVERSAL_THRESHOLD_DAYS}d since `created_at`.",
        "- **age_per_repo**: open PR is stale if age > that repo's",
        "  `stale_threshold_days` (populated in the DuckDB; default 30d).",
        f"- **idle_universal**: open PR is stale if it is still open AND idle "
        f"> {UNIVERSAL_THRESHOLD_DAYS}d (`fetch_now - updated_at`).",
        "- **idle_per_repo**: same, with the per-repo threshold substituted.",
        "",
        "The `idle_*` variants use a live re-fetch of every DB-OPEN PR's",
        "`updatedAt` (see `fetch_open_pr_activity.py`). PRs that were OPEN at",
        "the snapshot but have since been closed or merged are treated as",
        "non-stale — the close/merge event itself is activity.",
        "",
        "## Calibration sanity check",
        "",
        f"- Repos using the default 30d threshold: {c['repos_using_default_30d']}"
        f" / {c['repos_total']}",
        f"- Repos with a calibrated threshold: {c['repos_calibrated']}",
        "- Per-repo calibrated thresholds vs 2x median time-to-close:",
        f"  mean delta = "
        f"{fmt(c['per_repo_calibration_check']['mean_delta_vs_2x_median_ttc'])}"
        f", median delta = "
        f"{fmt(c['per_repo_calibration_check']['median_delta_vs_2x_median_ttc'])}"
        " (days).",
        "",
        "## Distribution shift",
        "",
        "Mean merge rate across all authors:",
        "",
        "| Definition | mean | median | p10 | p90 |",
        "|---|---|---|---|---|",
        *[
            f"| {label} | "
            f"{fmt(d[col]['mean'])} | "
            f"{fmt(d[col]['median'])} | "
            f"{fmt(d[col]['p10'])} | "
            f"{fmt(d[col]['p90'])} |"
            for label, col in [
                ("v3 baseline", "merge_rate_v3"),
                (f"age_universal ({UNIVERSAL_THRESHOLD_DAYS}d)",
                 "merge_rate_universal"),
                ("age_per_repo", "merge_rate_per_repo"),
                (f"idle_universal ({UNIVERSAL_THRESHOLD_DAYS}d)",
                 "merge_rate_idle_universal"),
                ("idle_per_repo", "merge_rate_idle_per_repo"),
            ]
        ],
        "",
        "Per-author drop from the v3 baseline (n authors, >0.10 / >0.25):",
        "",
        *[
            f"- **{label}**: "
            f"{s[col]['n_dropped_gt_0.10']} / {s[col]['n_dropped_gt_0.25']}"
            for label, col in [
                ("age_universal", "merge_rate_universal"),
                ("age_per_repo", "merge_rate_per_repo"),
                ("idle_universal", "merge_rate_idle_universal"),
                ("idle_per_repo", "merge_rate_idle_per_repo"),
            ]
        ],
        "",
        "## Signal quality vs ground truth",
        "",
        "2-feature logistic regression (merge_rate + log1p(median_additions)),",
        f"5-fold CV on {e['n_labeled']} labeled authors:",
        "",
        "| Definition | CV AUC | Active mean | Suspended mean | Cohen's d |",
        "|---|---|---|---|---|",
        *[
            f"| {label} | "
            f"{fmt(e[col]['cv_auc'])} | "
            f"{fmt(e[col]['mean_merge_rate_active'])} | "
            f"{fmt(e[col]['mean_merge_rate_suspended'])} | "
            f"{fmt(e[col]['cohens_d_active_vs_suspended'])} |"
            for label, col in [
                ("v3 baseline", "merge_rate_v3"),
                ("age_universal", "merge_rate_universal"),
                ("age_per_repo", "merge_rate_per_repo"),
                ("idle_universal", "merge_rate_idle_universal"),
                ("idle_per_repo", "merge_rate_idle_per_repo"),
            ]
        ],
        "",
        "## Recommendation",
        "",
        "See the `recommendation` field in "
        "`data/results/pocket_veto_analysis.json` for the machine-readable",
        "decision logic. Text summary and follow-up branch sketch below.",
        "",
        f"**{results['recommendation']['decision']}** — "
        f"{results['recommendation']['rationale']}",
        "",
        "### Follow-up branch sketch (if adopted)",
        "",
        "- `src/good_egg/github_client.py`: extend `_COMBINED_QUERY` with an",
        "  `openPullRequests` selection that pulls `createdAt`/`updatedAt` for",
        "  each OPEN PR on the scored user (or `totalCount` if we can push the",
        "  staleness filter into the query).",
        "- `src/good_egg/models.py`: add `open_stale_pr_count: int` (or similar)",
        "  to `UserContributionData`.",
        "- `src/good_egg/scorer.py:256-261`: change the `_score_v3` merge-rate",
        "  formula to `merged / (merged + closed + open_stale)`.",
        "- `src/good_egg/config.py`: add the staleness threshold as a tunable",
        "  config value.",
        "- Tests: parallel coverage in `tests/test_scorer.py`.",
        "",
    ]
    FINDINGS_PATH.write_text("\n".join(lines))


def decide(
    e: dict[str, Any], s: dict[str, Any], d: dict[str, Any],
) -> dict[str, Any]:
    """Produce a simple quantitative recommendation."""
    base_auc = e["merge_rate_v3"]["cv_auc"]
    aucs = {col: e[col]["cv_auc"] for col in MERGE_RATE_COLS}

    best_name = "merge_rate_v3"
    for col, auc in aucs.items():
        if col == "merge_rate_v3":
            continue
        if auc > aucs[best_name] + 0.005:
            best_name = col

    cohens = {
        col: e[col]["cohens_d_active_vs_suspended"] for col in MERGE_RATE_COLS
    }

    if best_name == "merge_rate_v3":
        decision = "Keep v3 as-is"
        rationale = (
            f"No variant beats v3 CV AUC {base_auc:.4f} by >0.005 "
            f"(aucs={ {k: round(v, 4) for k, v in aucs.items()} }). "
            f"Cohen's d also fails to improve "
            f"(base={cohens['merge_rate_v3']:.3f}, "
            f"best_alt={max(v for k, v in cohens.items() if k != 'merge_rate_v3'):.3f})."
        )
    else:
        affected = s[best_name]["n_dropped_gt_0.10"]
        decision = f"Adopt {best_name}"
        rationale = (
            f"{best_name} CV AUC {aucs[best_name]:.4f} beats v3 baseline "
            f"{base_auc:.4f} by >0.005. Cohen's d "
            f"{cohens[best_name]:.3f} vs v3 {cohens['merge_rate_v3']:.3f}. "
            f"{affected} authors shift by >0.10 in merge rate."
        )
    return {
        "decision": decision,
        "rationale": rationale,
        "cv_aucs": aucs,
        "cohens_d": cohens,
        "universal_threshold_days": UNIVERSAL_THRESHOLD_DAYS,
    }


def main() -> None:
    print(f"Loading {DB_PATH}")
    con = duckdb.connect(str(DB_PATH), read_only=True)

    print("Phase 1: characterization")
    characterization = characterize(con)
    print(f"  state totals: {characterization['state_totals']}")
    print(f"  outcome totals: {characterization['outcome_totals']}")
    print(
        f"  repos calibrated: {characterization['repos_calibrated']} / "
        f"{characterization['repos_total']}"
    )

    print("Phase 2: per-author features + merge-rate variants")
    if ACTIVITY_PATH.exists():
        print(f"  using idle-time sidecar: {ACTIVITY_PATH.name}")
    else:
        print("  (no idle-time sidecar; idle_* variants will be all-zero)")
    df = build_author_features(con)
    print(f"  {len(df)} authors")
    distributions = distribution_summary(df)
    for col in MERGE_RATE_COLS:
        v = distributions[col]
        print(f"  {col}: mean={v['mean']:.4f} median={v['median']:.4f}")

    print("Phase 3: shift analysis")
    shifts = shift_analysis(df)
    for alt in [c for c in MERGE_RATE_COLS if c != "merge_rate_v3"]:
        print(
            f"  {alt}: mean_delta={shifts[alt]['mean_delta']:+.4f} "
            f"n_dropped>0.10={shifts[alt]['n_dropped_gt_0.10']}"
        )

    print("Phase 4: signal evaluation")
    signal = signal_evaluation(df, con)
    for col in MERGE_RATE_COLS:
        print(
            f"  {col}: CV AUC={signal[col]['cv_auc']:.4f} "
            f"cohens_d={signal[col]['cohens_d_active_vs_suspended']:.4f}"
        )

    examples = pick_example_authors(df, con)
    print("Phase 3: example authors (most-shifted + labeled high-volume)")
    for row in examples:
        print(
            f"  {row['login']:20s} [{row['account_status']}] "
            f"total={row['total_prs']} v3={row['merge_rate_v3']:.3f} "
            f"uni={row['merge_rate_universal']:.3f} "
            f"per_repo={row['merge_rate_per_repo']:.3f}"
        )

    recommendation = decide(signal, shifts, distributions)
    print(f"\nRecommendation: {recommendation['decision']}")
    print(f"  {recommendation['rationale']}")

    results = {
        "universal_threshold_days": UNIVERSAL_THRESHOLD_DAYS,
        "characterization": characterization,
        "distributions": distributions,
        "shift_analysis": shifts,
        "signal_evaluation": signal,
        "example_authors": examples,
        "recommendation": recommendation,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {RESULTS_PATH}")

    write_findings(results)
    print(f"Wrote {FINDINGS_PATH}")

    con.close()


if __name__ == "__main__":
    main()
