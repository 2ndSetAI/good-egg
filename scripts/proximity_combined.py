"""Combined proximity + behavioral model experiment.

Tests H4: Does adding proximity-derived features to a logistic regression
model improve AUC beyond behavioral-features-only baselines?

Reads k-NN and graph results, identifies best proximity methods, then
trains LR on behavioral features + proximity score as additional feature.
"""

from __future__ import annotations

import json
from typing import Any

import duckdb
import networkx as nx
import numpy as np
import pandas as pd
from proximity_common import (
    DB_PATH,
    F10,
    F16,
    RESULTS_DIR,
    delong_auc_test,
    load_all_time_features,
    prepare_features,
)
from proximity_graph_experiment import (
    build_author_repo_data,
    jaccard_max_proximity,
    ppr_proximity,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def compute_knn_scores_cv(
    df: pd.DataFrame,
    feature_list: list[str],
    k: int,
    metric: str,
    seed: int = 42,
) -> np.ndarray:
    """Compute out-of-fold k-NN proximity scores via suspended-only CV."""
    y = (df["account_status"] == "suspended").astype(int).values
    x = prepare_features(df, feature_list)
    susp_idx = np.where(y == 1)[0]
    active_idx = np.where(y == 0)[0]
    n_susp = len(susp_idx)

    use_loo = n_susp < 30
    susp_oof = np.full(n_susp, np.nan)
    active_accum = np.zeros(len(active_idx))
    active_count = np.zeros(len(active_idx))

    if use_loo:
        splits = [(
            np.array([j for j in range(n_susp) if j != i]),
            np.array([i]),
        ) for i in range(n_susp)]
    else:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        splits = list(kf.split(np.arange(n_susp)))

    for train_pos, test_pos in splits:
        seed_susp_idx = susp_idx[train_pos]
        held_susp_idx = susp_idx[test_pos]

        train_idx = np.concatenate([seed_susp_idx, active_idx])
        scaler = StandardScaler()
        scaler.fit(x[train_idx])

        seed_scaled = scaler.transform(x[seed_susp_idx])
        eval_idx = np.concatenate([held_susp_idx, active_idx])
        eval_scaled = scaler.transform(x[eval_idx])

        effective_k = min(k, len(seed_scaled))
        if effective_k == 0:
            continue
        nn = NearestNeighbors(n_neighbors=effective_k, metric=metric)
        nn.fit(seed_scaled)
        dists, _ = nn.kneighbors(eval_scaled)
        scores = -dists.mean(axis=1)

        n_held = len(held_susp_idx)
        susp_oof[test_pos] = scores[:n_held]
        active_accum += scores[n_held:]
        active_count += 1

    safe = np.where(active_count > 0, active_count, 1.0)
    oof = np.full(len(y), np.nan)
    oof[susp_idx] = susp_oof
    oof[active_idx] = active_accum / safe

    return oof


def compute_graph_scores_cv(
    df: pd.DataFrame,
    author_repos: dict[str, set[str]],
    graph: nx.Graph,
    method: str,
    seed: int = 42,
) -> np.ndarray:
    """Compute out-of-fold graph proximity scores via suspended-only CV."""
    author_col = "author" if "author" in df.columns else "login"
    y = (df["account_status"] == "suspended").astype(int).values
    authors = df[author_col].tolist()
    susp_idx = np.where(y == 1)[0]
    active_idx = np.where(y == 0)[0]
    n_susp = len(susp_idx)

    use_loo = n_susp < 30
    susp_oof = np.full(n_susp, np.nan)
    active_accum = np.zeros(len(active_idx))
    active_count = np.zeros(len(active_idx))

    if use_loo:
        splits = [(
            np.array([j for j in range(n_susp) if j != i]),
            np.array([i]),
        ) for i in range(n_susp)]
    else:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        splits = list(kf.split(np.arange(n_susp)))

    for train_pos, test_pos in splits:
        seed_authors = [authors[susp_idx[j]] for j in train_pos]
        eval_idx_arr = np.concatenate([susp_idx[test_pos], active_idx])
        eval_authors = [authors[i] for i in eval_idx_arr]

        if method == "jaccard_max":
            s_repos = {a: author_repos.get(a, set()) for a in seed_authors}
            e_repos = {a: author_repos.get(a, set()) for a in eval_authors}
            scores_dict = jaccard_max_proximity(s_repos, e_repos)
            scores = np.array([scores_dict.get(a, 0.0) for a in eval_authors])
        elif method == "ppr":
            ppr_dict = ppr_proximity(graph, seed_authors)
            scores = np.array([ppr_dict.get(a, 0.0) for a in eval_authors])
        else:
            scores = np.zeros(len(eval_authors))

        n_held = len(test_pos)
        susp_oof[test_pos] = scores[:n_held]
        active_accum += scores[n_held:]
        active_count += 1

    safe = np.where(active_count > 0, active_count, 1.0)
    oof = np.full(len(y), np.nan)
    oof[susp_idx] = susp_oof
    oof[active_idx] = active_accum / safe

    return oof


def lr_with_proximity_cv(
    df: pd.DataFrame,
    feature_list: list[str],
    proximity_scores: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """LR CV with behavioral features + proximity score as extra feature.

    Returns (y_true, oof_probs).
    """
    y = (df["account_status"] == "suspended").astype(int).values
    x_base = prepare_features(df, feature_list)

    # Add proximity as extra column
    prox = proximity_scores.reshape(-1, 1)
    # Fill NaN with median
    median_prox = np.nanmedian(prox)
    prox = np.where(np.isfinite(prox), prox, median_prox)
    x_combined = np.hstack([x_base, prox])

    n_pos = y.sum()
    oof = np.full(len(y), np.nan)

    if n_pos < 30:
        loo = LeaveOneOut()
        for train_idx, test_idx in loo.split(x_combined, y):
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_combined[train_idx])
            x_test = scaler.transform(x_combined[test_idx])
            model = LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=seed,
            )
            model.fit(x_train, y[train_idx])
            oof[test_idx] = model.predict_proba(x_test)[:, 1]
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(x_combined, y):
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_combined[train_idx])
            x_test = scaler.transform(x_combined[test_idx])
            model = LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=seed,
            )
            model.fit(x_train, y[train_idx])
            oof[test_idx] = model.predict_proba(x_test)[:, 1]

    return y, oof


def main() -> None:
    results: dict[str, Any] = {}

    # Load data
    df = load_all_time_features(merged_pr_only=True)
    if "login" in df.columns and "author" not in df.columns:
        df = df.rename(columns={"login": "author"})

    # Determine which feature sets are available
    available_f16 = [f for f in F16 if f in df.columns]
    feature_sets = {"F10": F10, "F16": available_f16}

    # Load graph data
    con = duckdb.connect(str(DB_PATH), read_only=True)
    author_repos, graph = build_author_repo_data(con)
    con.close()

    # Load prior results to find best methods
    knn_path = RESULTS_DIR / "knn_results.json"
    graph_path = RESULTS_DIR / "graph_results.json"

    best_knn_config = {"feature_set": "F10", "k": 5, "metric": "cosine"}
    best_graph_method = "jaccard_max"

    if knn_path.exists():
        with open(knn_path) as f:
            knn_results = json.load(f)
        # Find best k-NN from strategy_b_merged
        stb = knn_results.get("strategy_b_merged", {})
        best_auc = 0.0
        for key, val in stb.items():
            if key.startswith("knn_") and isinstance(val, dict):
                auc = val.get("auc_roc", 0.0)
                if auc > best_auc:
                    best_auc = auc
                    best_knn_config = {
                        "feature_set": val.get("feature_set", "F10"),
                        "k": val.get("k", 5),
                        "metric": val.get("metric", "cosine"),
                    }
        print(f"Best k-NN from prior results: {best_knn_config} "
              f"(AUC={best_auc:.4f})")
    else:
        print("No prior k-NN results found, using defaults")

    if graph_path.exists():
        with open(graph_path) as f:
            graph_results = json.load(f)
        stb = graph_results.get("strategy_b_merged", {})
        best_graph_auc = 0.0
        for method in ["jaccard_max", "jaccard_mean_k5", "ppr"]:
            val = stb.get(method, {})
            auc = val.get("auc_roc", 0.0)
            if auc > best_graph_auc:
                best_graph_auc = auc
                best_graph_method = method
        print(f"Best graph method from prior results: {best_graph_method} "
              f"(AUC={best_graph_auc:.4f})")
    else:
        print("No prior graph results found, using defaults")

    # Compute OOF proximity scores
    print("\nComputing k-NN OOF scores...")
    fs_key = best_knn_config["feature_set"]
    fs_list = feature_sets.get(fs_key, F10)
    knn_oof = compute_knn_scores_cv(
        df, fs_list,
        k=best_knn_config["k"],
        metric=best_knn_config["metric"],
    )

    print("Computing graph OOF scores...")
    graph_oof = compute_graph_scores_cv(
        df, author_repos, graph, best_graph_method,
    )

    # For each behavioral feature set, test: LR alone vs LR + proximity
    for fs_name, fs in feature_sets.items():
        print(f"\n--- Combined models with {fs_name} ---")

        # Behavioral-only baseline
        y_base, oof_base = lr_with_proximity_cv(
            df, fs, np.zeros(len(df)),  # zero proximity = behavioral only
        )
        base_auc = roc_auc_score(y_base, oof_base)
        print(f"  LR({fs_name}) baseline: AUC={base_auc:.4f}")

        # LR + k-NN proximity
        y_knn, oof_knn = lr_with_proximity_cv(df, fs, knn_oof)
        knn_auc = roc_auc_score(y_knn, oof_knn)
        dl_knn = delong_auc_test(y_knn, oof_knn, oof_base)
        print(f"  LR({fs_name}) + k-NN: AUC={knn_auc:.4f} "
              f"(delta={knn_auc - base_auc:+.4f}, "
              f"p={dl_knn['p_value']:.4f})")

        # LR + graph proximity
        y_graph, oof_graph = lr_with_proximity_cv(df, fs, graph_oof)
        graph_auc = roc_auc_score(y_graph, oof_graph)
        dl_graph = delong_auc_test(y_graph, oof_graph, oof_base)
        print(f"  LR({fs_name}) + graph: AUC={graph_auc:.4f} "
              f"(delta={graph_auc - base_auc:+.4f}, "
              f"p={dl_graph['p_value']:.4f})")

        # LR + both
        both_oof = np.column_stack([knn_oof, graph_oof])
        both_median = np.nanmedian(both_oof, axis=0)
        both_oof = np.where(np.isfinite(both_oof), both_oof, both_median)
        # Combine as mean of z-scored proximity scores
        combined_prox = np.nanmean(both_oof, axis=1)

        y_both, oof_both = lr_with_proximity_cv(df, fs, combined_prox)
        both_auc = roc_auc_score(y_both, oof_both)
        dl_both = delong_auc_test(y_both, oof_both, oof_base)
        print(f"  LR({fs_name}) + both: AUC={both_auc:.4f} "
              f"(delta={both_auc - base_auc:+.4f}, "
              f"p={dl_both['p_value']:.4f})")

        results[fs_name] = {
            "baseline_auc": base_auc,
            "knn_combined": {
                "auc": knn_auc,
                "delta": knn_auc - base_auc,
                "delong_p": dl_knn["p_value"],
                "delong_z": dl_knn["z_statistic"],
            },
            "graph_combined": {
                "auc": graph_auc,
                "delta": graph_auc - base_auc,
                "delong_p": dl_graph["p_value"],
                "delong_z": dl_graph["z_statistic"],
            },
            "both_combined": {
                "auc": both_auc,
                "delta": both_auc - base_auc,
                "delong_p": dl_both["p_value"],
                "delong_z": dl_both["z_statistic"],
            },
        }

    results["config"] = {
        "best_knn": best_knn_config,
        "best_graph_method": best_graph_method,
    }

    # Save
    output_path = RESULTS_DIR / "combined_results.json"

    def _default(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=_default)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
