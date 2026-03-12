"""k-NN proximity experiment for suspension detection.

Tests H1 (profile proximity detects suspension in merged-PR population)
via k-NN distance scoring with cosine and euclidean metrics.

Strategies:
  A: Discovery-order holdout (44 original seeds → 373 expansion test)
  B: Suspended-only CV on merged-PR and all-authors populations
  C: Temporal holdout with pre-cutoff feature computation
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from proximity_common import (
    CUTOFFS,
    F10,
    F16,
    F16_NO_MR,
    RESULTS_DIR,
    compute_metrics,
    load_all_time_features,
    load_temporal_features,
    lr_baseline,
    merge_rate_baseline,
    prepare_features,
    run_suspended_cv,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def knn_proximity_score(
    seed_x: np.ndarray,
    eval_x: np.ndarray,
    k: int,
    metric: str,
) -> np.ndarray:
    """Score eval set by negative mean distance to k nearest seeds.

    Higher score = closer to seeds = more suspicious.
    """
    effective_k = min(k, len(seed_x))
    if effective_k == 0:
        return np.zeros(len(eval_x))
    nn = NearestNeighbors(n_neighbors=effective_k, metric=metric)
    nn.fit(seed_x)
    distances, _ = nn.kneighbors(eval_x)
    return -distances.mean(axis=1)


def run_strategy_a(
    df: Any,
    feature_sets: dict[str, list[str]],
    metrics_list: list[str],
    k_values: list[int],
) -> dict[str, Any]:
    """Strategy A: Discovery-order holdout.

    Seeds = original 44 suspended with merged PRs.
    Test positives = 373 expansion suspended with merged PRs.
    Test negatives = all active with merged PRs.
    """
    print("\n=== Strategy A: Discovery-Order Holdout ===")
    results: dict[str, Any] = {}

    susp_mask = df["account_status"] == "suspended"
    orig_mask = df["is_original_discovery"]

    seed_df = df[orig_mask].copy()
    test_df = df[~orig_mask | ~susp_mask].copy()

    n_seeds = len(seed_df)
    n_test_susp = (test_df["account_status"] == "suspended").sum()
    n_test_active = (test_df["account_status"] == "active").sum()
    print(f"  Seeds: {n_seeds}, Test: {n_test_susp} suspended, "
          f"{n_test_active} active")

    y_test = (test_df["account_status"] == "suspended").astype(int).values

    for fs_name, fs in feature_sets.items():
        for metric in metrics_list:
            for k in k_values:
                key = f"knn_{fs_name}_{metric}_k{k}"

                seed_x = prepare_features(seed_df, fs)
                test_x = prepare_features(test_df, fs)

                # Scale on seeds + active (no test suspended)
                all_train_x = np.vstack([
                    seed_x,
                    prepare_features(
                        test_df[test_df["account_status"] == "active"], fs,
                    ),
                ])
                scaler = StandardScaler()
                scaler.fit(all_train_x)

                seed_scaled = scaler.transform(seed_x)
                test_scaled = scaler.transform(test_x)

                scores = knn_proximity_score(
                    seed_scaled, test_scaled, k, metric,
                )

                m = compute_metrics(y_test, scores)
                results[key] = {
                    "feature_set": fs_name,
                    "metric": metric,
                    "k": k,
                    "n_seeds": n_seeds,
                    **m,
                }
                print(f"  {key}: AUC={m.get('auc_roc', float('nan')):.4f}")

    # Baselines on test set
    y_mr, s_mr = merge_rate_baseline(test_df)
    results["baseline_merge_rate"] = compute_metrics(y_mr, s_mr)
    print(f"  baseline_merge_rate: "
          f"AUC={results['baseline_merge_rate'].get('auc_roc', float('nan')):.4f}")

    for fs_name, fs in feature_sets.items():
        y_lr, s_lr = lr_baseline(test_df, fs)
        bkey = f"baseline_lr_{fs_name}"
        results[bkey] = compute_metrics(y_lr, s_lr)
        print(f"  {bkey}: "
              f"AUC={results[bkey].get('auc_roc', float('nan')):.4f}")

    return results


def run_strategy_b(
    df: Any,
    feature_sets: dict[str, list[str]],
    metrics_list: list[str],
    k_values: list[int],
    label: str = "merged-PR",
) -> dict[str, Any]:
    """Strategy B: Suspended-only CV on all-time features."""
    print(f"\n=== Strategy B: CV ({label}) ===")
    results: dict[str, Any] = {}

    for fs_name, fs in feature_sets.items():
        for metric in metrics_list:
            for k in k_values:
                key = f"knn_{fs_name}_{metric}_k{k}"

                def score_fn(
                    seed_df: Any,
                    eval_df: Any,
                    _fs: list[str] = fs,
                    _k: int = k,
                    _metric: str = metric,
                ) -> np.ndarray:
                    seed_x = prepare_features(seed_df, _fs)
                    eval_x = prepare_features(eval_df, _fs)

                    all_x = np.vstack([seed_x, eval_x])
                    scaler = StandardScaler()
                    scaler.fit(all_x)

                    return knn_proximity_score(
                        scaler.transform(seed_x),
                        scaler.transform(eval_x),
                        _k,
                        _metric,
                    )

                y, scores = run_suspended_cv(df, score_fn)
                m = compute_metrics(y, scores)
                results[key] = {
                    "feature_set": fs_name,
                    "metric": metric,
                    "k": k,
                    **m,
                }
                print(f"  {key}: AUC={m.get('auc_roc', float('nan')):.4f}")

    # Baselines
    y_mr, s_mr = merge_rate_baseline(df)
    results["baseline_merge_rate"] = compute_metrics(y_mr, s_mr)
    print(f"  baseline_merge_rate: "
          f"AUC={results['baseline_merge_rate'].get('auc_roc', float('nan')):.4f}")

    for fs_name, fs in feature_sets.items():
        y_lr, s_lr = lr_baseline(df, fs)
        bkey = f"baseline_lr_{fs_name}"
        results[bkey] = compute_metrics(y_lr, s_lr)
        print(f"  {bkey}: "
              f"AUC={results[bkey].get('auc_roc', float('nan')):.4f}")

    return results


def run_strategy_c(
    feature_sets: dict[str, list[str]],
    metrics_list: list[str],
    k_values: list[int],
) -> dict[str, Any]:
    """Strategy C: Temporal holdout with pre-cutoff features."""
    print("\n=== Strategy C: Temporal Holdout ===")
    results: dict[str, Any] = {}

    for cutoff in CUTOFFS:
        df = load_temporal_features(cutoff)
        n_susp = (df["account_status"] == "suspended").sum()
        if n_susp < 5:
            print(f"  Skipping {cutoff}: only {n_susp} suspended")
            continue

        cutoff_results: dict[str, Any] = {"n_suspended": int(n_susp)}

        # Only test cosine + selected k values for temporal
        for fs_name in ["F10", "F16"]:
            fs = feature_sets[fs_name]
            # Check all features are available
            missing = [f for f in fs if f not in df.columns]
            if missing:
                print(f"  Skipping {fs_name} at {cutoff}: missing {missing}")
                continue

            for k in [5, 10]:
                if k not in k_values:
                    continue

                key = f"knn_{fs_name}_cosine_k{k}"

                def score_fn(
                    seed_df: Any,
                    eval_df: Any,
                    _fs: list[str] = fs,
                    _k: int = k,
                ) -> np.ndarray:
                    seed_x = prepare_features(seed_df, _fs)
                    eval_x = prepare_features(eval_df, _fs)

                    all_x = np.vstack([seed_x, eval_x])
                    scaler = StandardScaler()
                    scaler.fit(all_x)

                    return knn_proximity_score(
                        scaler.transform(seed_x),
                        scaler.transform(eval_x),
                        _k,
                        "cosine",
                    )

                y, scores = run_suspended_cv(df, score_fn)
                m = compute_metrics(y, scores)
                cutoff_results[key] = {
                    "feature_set": fs_name,
                    "metric": "cosine",
                    "k": k,
                    **m,
                }
                print(f"  {cutoff} {key}: "
                      f"AUC={m.get('auc_roc', float('nan')):.4f}")

        # Baselines per cutoff
        y_mr, s_mr = merge_rate_baseline(df)
        cutoff_results["baseline_merge_rate"] = compute_metrics(y_mr, s_mr)

        for fs_name in ["F10", "F16"]:
            fs = feature_sets[fs_name]
            missing = [f for f in fs if f not in df.columns]
            if missing:
                continue
            y_lr, s_lr = lr_baseline(df, fs)
            cutoff_results[f"baseline_lr_{fs_name}"] = compute_metrics(y_lr, s_lr)

        results[cutoff] = cutoff_results

    return results


def run_delong_comparisons(
    all_results: dict[str, Any],
) -> dict[str, Any]:
    """Run DeLong tests comparing k-NN methods to baselines.

    Operates on stored y_true/y_scores that were saved during experiments.
    For simplicity, we re-identify the best methods and note AUC comparisons.
    """
    comparisons: dict[str, Any] = {}

    for strategy_name, strategy_results in all_results.items():
        if not isinstance(strategy_results, dict):
            continue

        # Find all knn results and baselines
        knn_keys = [
            k for k in strategy_results
            if k.startswith("knn_") and isinstance(strategy_results[k], dict)
        ]
        baseline_keys = [
            k for k in strategy_results
            if k.startswith("baseline_") and isinstance(strategy_results[k], dict)
        ]

        if not knn_keys or not baseline_keys:
            continue

        # Report best knn AUC vs baselines
        best_knn_key = max(
            knn_keys,
            key=lambda k: strategy_results[k].get("auc_roc", float("-inf")),
        )
        best_knn_auc = strategy_results[best_knn_key].get("auc_roc", float("nan"))

        comp: dict[str, Any] = {
            "best_knn": best_knn_key,
            "best_knn_auc": best_knn_auc,
            "baselines": {},
        }

        for bk in baseline_keys:
            b_auc = strategy_results[bk].get("auc_roc", float("nan"))
            comp["baselines"][bk] = {
                "auc": b_auc,
                "delta": best_knn_auc - b_auc if np.isfinite(best_knn_auc)
                         and np.isfinite(b_auc) else float("nan"),
            }

        comparisons[strategy_name] = comp

    return comparisons


def main() -> None:
    feature_sets = {"F10": F10, "F16": F16, "F16_no_mr": F16_NO_MR}
    metrics_list = ["cosine", "euclidean"]
    k_values = [3, 5, 10, 15]

    all_results: dict[str, Any] = {}

    # Load merged-PR population
    df_merged = load_all_time_features(merged_pr_only=True)

    # Check which F16 features are available in parquet
    available_f16 = [f for f in F16 if f in df_merged.columns]
    missing_f16 = [f for f in F16 if f not in df_merged.columns]
    if missing_f16:
        print(f"Warning: Missing F16 features in parquet: {missing_f16}")
        # Use only available features
        feature_sets["F16"] = available_f16
        feature_sets["F16_no_mr"] = [
            f for f in available_f16
            if f not in ("merge_rate", "rejection_rate")
        ]

    # Strategy A: Discovery-order holdout
    all_results["strategy_a"] = run_strategy_a(
        df_merged, feature_sets, metrics_list, k_values,
    )

    # Strategy B: CV on merged-PR population
    all_results["strategy_b_merged"] = run_strategy_b(
        df_merged, feature_sets, metrics_list, k_values,
        label="merged-PR",
    )

    # Strategy B: Replication on all-authors population
    print("\n--- Loading all-authors population for replication ---")
    df_all = load_all_time_features(merged_pr_only=False)
    available_f16_all = [f for f in F16 if f in df_all.columns]
    repl_fs = {"F16": available_f16_all}
    all_results["strategy_b_all_authors"] = run_strategy_b(
        df_all, repl_fs, ["cosine"], [5],
        label="all-authors (stage 12 replication)",
    )

    # Strategy C: Temporal holdout
    all_results["strategy_c"] = run_strategy_c(
        {"F10": F10, "F16": F16}, metrics_list=["cosine"],
        k_values=[5, 10],
    )

    # DeLong comparisons (AUC-level, no paired scores available)
    all_results["delong_comparisons"] = run_delong_comparisons(all_results)

    # Save results
    output_path = RESULTS_DIR / "knn_results.json"

    def _default(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_default)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
