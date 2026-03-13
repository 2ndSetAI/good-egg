"""Aggregate proximity experiment results and generate analysis report.

Reads all results JSONs, computes summary statistics, DeLong comparisons,
hypothesis verdicts, and writes PROXIMITY_ANALYSIS.md.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from proximity_common import RESULTS_DIR


def load_results() -> dict[str, Any]:
    """Load all result JSON files."""
    data: dict[str, Any] = {}
    for name in ["knn_results", "graph_results", "combined_results"]:
        path = RESULTS_DIR / f"{name}.json"
        if path.exists():
            with open(path) as f:
                data[name] = json.load(f)
            print(f"Loaded {path}")
        else:
            print(f"Warning: {path} not found")
    return data


def extract_auc_table(
    results: dict[str, Any],
    strategy_key: str,
) -> list[dict[str, Any]]:
    """Extract method → AUC rows from a strategy's results."""
    strategy = results.get(strategy_key, {})
    rows: list[dict[str, Any]] = []

    for key, val in strategy.items():
        if not isinstance(val, dict) or "auc_roc" not in val:
            continue
        row: dict[str, Any] = {"method": key}
        row["auc_roc"] = val.get("auc_roc", float("nan"))
        row["auc_pr"] = val.get("auc_pr", float("nan"))
        row["precision_at_25"] = val.get("precision_at_25", float("nan"))
        row["precision_at_50"] = val.get("precision_at_50", float("nan"))
        # Extra metadata
        for meta in ["feature_set", "metric", "k"]:
            if meta in val:
                row[meta] = val[meta]
        rows.append(row)

    rows.sort(key=lambda r: -r.get("auc_roc", 0.0) if np.isfinite(
        r.get("auc_roc", float("nan"))) else float("-inf"))
    return rows


def format_auc_table(rows: list[dict[str, Any]], title: str) -> str:
    """Format AUC table as markdown."""
    if not rows:
        return f"### {title}\n\nNo results available.\n"

    lines = [f"### {title}\n"]
    lines.append("| Method | AUC-ROC | AUC-PR | P@25 | P@50 |")
    lines.append("|--------|---------|--------|------|------|")

    for r in rows:
        auc = r.get("auc_roc", float("nan"))
        apr = r.get("auc_pr", float("nan"))
        p25 = r.get("precision_at_25", float("nan"))
        p50 = r.get("precision_at_50", float("nan"))

        def fmt(v: float) -> str:
            return f"{v:.4f}" if np.isfinite(v) else "—"

        lines.append(
            f"| {r['method']} | {fmt(auc)} | {fmt(apr)} | "
            f"{fmt(p25)} | {fmt(p50)} |"
        )

    lines.append("")
    return "\n".join(lines)


def assess_hypothesis(
    label: str,
    description: str,
    aucs: list[float],
    threshold: float,
    null_description: str,
) -> str:
    """Assess whether hypothesis is supported."""
    valid_aucs = [a for a in aucs if np.isfinite(a)]
    if not valid_aucs:
        return (f"**{label}**: {description}\n\n"
                f"**Verdict**: INCONCLUSIVE — no valid AUC results.\n\n")

    best = max(valid_aucs)
    mean = np.mean(valid_aucs)

    if best > threshold:
        verdict = "SUPPORTED"
        detail = (f"Best AUC = {best:.4f} > {threshold:.2f} threshold. "
                  f"Mean AUC = {mean:.4f}.")
    else:
        verdict = "NOT SUPPORTED"
        detail = (f"Best AUC = {best:.4f} ≤ {threshold:.2f} threshold. "
                  f"{null_description}")

    return (f"**{label}**: {description}\n\n"
            f"**Verdict**: {verdict} — {detail}\n\n")


def generate_report(data: dict[str, Any]) -> str:
    """Generate the full PROXIMITY_ANALYSIS.md report."""
    sections: list[str] = []

    sections.append("# Proximity-Based Suspension Detection — Results\n")
    sections.append(
        "This report summarizes the results of proximity-based methods "
        "for detecting suspended GitHub accounts among authors with merged PRs.\n"
    )

    # --- k-NN Results ---
    knn = data.get("knn_results", {})

    sections.append("## 1. k-NN Proximity Results\n")

    # Strategy A
    rows_a = extract_auc_table(knn, "strategy_a")
    sections.append(format_auc_table(
        rows_a, "Strategy A: Discovery-Order Holdout (k-NN)",
    ))

    # Strategy B merged
    rows_b = extract_auc_table(knn, "strategy_b_merged")
    sections.append(format_auc_table(
        rows_b, "Strategy B: Suspended-Only CV, Merged-PR Population (k-NN)",
    ))

    # Strategy B all-authors replication
    rows_b_all = extract_auc_table(knn, "strategy_b_all_authors")
    sections.append(format_auc_table(
        rows_b_all,
        "Strategy B: Suspended-Only CV, All Authors (Stage 12 Replication)",
    ))

    # Strategy C temporal
    stc = knn.get("strategy_c", {})
    for cutoff, cutoff_data in stc.items():
        if isinstance(cutoff_data, dict):
            rows_c = extract_auc_table({"c": cutoff_data}, "c")
            sections.append(format_auc_table(
                rows_c, f"Strategy C: Temporal Holdout, cutoff={cutoff} (k-NN)",
            ))

    # --- Graph Results ---
    graph = data.get("graph_results", {})

    sections.append("## 2. Graph Proximity Results\n")

    rows_ga = extract_auc_table(graph, "strategy_a")
    sections.append(format_auc_table(
        rows_ga, "Strategy A: Discovery-Order Holdout (Graph)",
    ))

    rows_gb = extract_auc_table(graph, "strategy_b_merged")
    sections.append(format_auc_table(
        rows_gb, "Strategy B: Suspended-Only CV, Merged-PR Population (Graph)",
    ))

    stcg = graph.get("strategy_c", {})
    for cutoff, cutoff_data in stcg.items():
        if isinstance(cutoff_data, dict):
            rows_gc = extract_auc_table({"c": cutoff_data}, "c")
            sections.append(format_auc_table(
                rows_gc,
                f"Strategy C: Temporal Holdout, cutoff={cutoff} (Graph)",
            ))

    # --- Combined Results ---
    combined = data.get("combined_results", {})

    sections.append("## 3. Combined Model Results (H4)\n")

    for fs_name in ["F10", "F16"]:
        fs_data = combined.get(fs_name, {})
        if not fs_data:
            continue

        baseline_auc = fs_data.get("baseline_auc", float("nan"))
        sections.append(f"### {fs_name} Features\n")
        sections.append(f"Behavioral LR baseline: AUC = "
                        f"{baseline_auc:.4f}\n")

        lines = ["| Model | AUC | Delta | DeLong p |"]
        lines.append("|-------|-----|-------|----------|")

        for combo_name in ["knn_combined", "graph_combined", "both_combined"]:
            c = fs_data.get(combo_name, {})
            auc = c.get("auc", float("nan"))
            delta = c.get("delta", float("nan"))
            p = c.get("delong_p", float("nan"))

            def fmt(v: float) -> str:
                return f"{v:.4f}" if np.isfinite(v) else "—"

            lines.append(
                f"| LR + {combo_name.replace('_combined', '')} | "
                f"{fmt(auc)} | {fmt(delta)} | {fmt(p)} |"
            )

        lines.append("")
        sections.append("\n".join(lines))

    # --- Hypothesis Verdicts ---
    sections.append("## 4. Hypothesis Verdicts\n")

    # H1: k-NN on merged-PR pop AUC > 0.55
    h1_aucs = [
        r.get("auc_roc", float("nan"))
        for r in rows_b
        if r["method"].startswith("knn_")
    ]
    sections.append(assess_hypothesis(
        "H1", "Profile proximity detects suspension in merged-PR population "
              "(k-NN AUC > 0.55)",
        h1_aucs, 0.55,
        "No actionable proximity signal in this population.",
    ))

    # H2: Graph proximity AUC > 0.55
    h2_aucs = [
        r.get("auc_roc", float("nan"))
        for r in rows_gb
        if r["method"] in ("jaccard_max", "jaccard_mean_k5", "ppr")
    ]
    sections.append(assess_hypothesis(
        "H2", "Graph-based proximity captures structural signal (AUC > 0.55)",
        h2_aucs, 0.55,
        "Graph proximity does not capture structural signal.",
    ))

    # H3: Strategy A generalizes
    h3_aucs = [
        r.get("auc_roc", float("nan"))
        for r in rows_a
        if r["method"].startswith("knn_")
    ]
    h3_b_aucs = [
        r.get("auc_roc", float("nan"))
        for r in rows_b
        if r["method"].startswith("knn_")
    ]
    valid_h3 = [a for a in h3_aucs if np.isfinite(a)]
    valid_h3b = [a for a in h3_b_aucs if np.isfinite(a)]

    if valid_h3 and valid_h3b:
        best_a = max(valid_h3)
        best_b = max(valid_h3b)
        if best_a >= best_b - 0.03:
            h3_verdict = (f"SUPPORTED — Strategy A AUC ({best_a:.4f}) is within "
                          f"0.03 of Strategy B ({best_b:.4f}), suggesting "
                          f"generalization from biased seeds.")
        else:
            h3_verdict = (f"NOT SUPPORTED — Strategy A AUC ({best_a:.4f}) is "
                          f"substantially lower than Strategy B ({best_b:.4f}), "
                          f"suggesting seed selection bias affects results.")
    else:
        h3_verdict = "INCONCLUSIVE — insufficient data."

    sections.append(
        f"**H3**: Proximity signal is robust to seed selection bias\n\n"
        f"**Verdict**: {h3_verdict}\n\n"
    )

    # H4: Proximity adds incremental value
    h4_results: list[str] = []
    for fs_name in ["F10", "F16"]:
        fs_data = combined.get(fs_name, {})
        for combo in ["knn_combined", "graph_combined", "both_combined"]:
            c = fs_data.get(combo, {})
            p = c.get("delong_p", float("nan"))
            delta = c.get("delta", float("nan"))
            if np.isfinite(p) and p < 0.05 and delta > 0:
                h4_results.append(
                    f"{fs_name}+{combo}: delta={delta:+.4f}, p={p:.4f}"
                )

    if h4_results:
        h4_verdict = "SUPPORTED — " + "; ".join(h4_results)
    else:
        h4_verdict = ("NOT SUPPORTED — No proximity feature significantly "
                      "improves LR AUC (DeLong p > 0.05).")

    sections.append(
        f"**H4**: Proximity adds incremental value to behavioral features\n\n"
        f"**Verdict**: {h4_verdict}\n\n"
    )

    # --- Summary ---
    sections.append("## 5. Summary\n")

    # Find overall best method
    all_aucs: list[tuple[str, str, float]] = []
    for strat_name, rows in [
        ("Strategy A (k-NN)", rows_a),
        ("Strategy B merged (k-NN)", rows_b),
        ("Strategy A (Graph)", rows_ga),
        ("Strategy B merged (Graph)", rows_gb),
    ]:
        for r in rows:
            auc = r.get("auc_roc", float("nan"))
            if np.isfinite(auc) and not r["method"].startswith("baseline"):
                all_aucs.append((strat_name, r["method"], auc))

    if all_aucs:
        all_aucs.sort(key=lambda x: -x[2])
        best_strat, best_method, best_auc = all_aucs[0]
        sections.append(
            f"**Best overall method**: {best_method} ({best_strat}), "
            f"AUC = {best_auc:.4f}\n\n"
        )

        # Compare to random baseline
        if best_auc > 0.55:
            sections.append(
                "The best method exceeds the AUC > 0.55 threshold, suggesting "
                "proximity-based detection has *some* signal on the merged-PR "
                "population. However, the practical value depends on the "
                "magnitude and precision at operational thresholds.\n"
            )
        else:
            sections.append(
                "No method exceeds the AUC > 0.55 threshold on the merged-PR "
                "population. Proximity-based detection does not appear to "
                "rescue what individual behavioral features cannot.\n"
            )

    # Stage 12 replication comparison
    if rows_b_all:
        repl_knn = [
            r for r in rows_b_all
            if r["method"].startswith("knn_")
        ]
        if repl_knn:
            repl_best = max(
                repl_knn, key=lambda r: r.get("auc_roc", 0.0),
            )
            sections.append(
                f"\n**Stage 12 replication** (all-authors, F16, cosine, k=5): "
                f"AUC = {repl_best.get('auc_roc', float('nan')):.4f} "
                f"(original stage 12: 0.595)\n"
            )

    return "\n".join(sections)


def main() -> None:
    data = load_results()

    if not data:
        print("No results found. Run experiment scripts first.")
        return

    report = generate_report(data)

    output_path = RESULTS_DIR / "PROXIMITY_ANALYSIS.md"
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nReport written to {output_path}")

    # Print to console too
    print("\n" + "=" * 80)
    print(report)


if __name__ == "__main__":
    main()
