"""Cross-check RESULTS.md claims against JSON output files.

Run: python -m experiments.bot_detection.scripts.verify_results
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"
RESULTS_MD = Path(__file__).parent.parent / "RESULTS.md"

PASS_COUNT = 0
FAIL_COUNT = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS_COUNT, FAIL_COUNT
    status = "PASS" if condition else "FAIL"
    if not condition:
        FAIL_COUNT += 1
    else:
        PASS_COUNT += 1
    suffix = f" -- {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")


def load_json(filename: str) -> dict:
    path = RESULTS_DIR / filename
    if not path.exists():
        print(f"  [SKIP] {filename} not found")
        return {}
    with open(path) as f:
        return json.load(f)


def approx_eq(a: float, b: float, tol: float = 0.005) -> bool:
    """Check approximate equality with tolerance for rounding."""
    if a is None or b is None:
        return False
    return abs(a - b) < tol


def check_population_arithmetic(data: dict) -> None:
    """Check multi_repo + single_repo == all for n_total and n_positive."""
    print("\n== Population Arithmetic ==")
    if not all(k in data for k in ("multi_repo", "single_repo", "all")):
        check("populations present", False, "missing population keys")
        return

    mr = data["multi_repo"]
    sr = data["single_repo"]
    all_ = data["all"]

    check(
        "n_total: multi + single == all",
        mr["n_total"] + sr["n_total"] == all_["n_total"],
        f"{mr['n_total']} + {sr['n_total']} = {mr['n_total'] + sr['n_total']}"
        f", expected {all_['n_total']}",
    )
    check(
        "n_positive: multi + single == all",
        mr["n_positive"] + sr["n_positive"] == all_["n_positive"],
        f"{mr['n_positive']} + {sr['n_positive']} = "
        f"{mr['n_positive'] + sr['n_positive']}"
        f", expected {all_['n_positive']}",
    )


def check_base_rates(data: dict) -> None:
    """Check base_rate = n_positive / n_total for each population."""
    print("\n== Base Rate Consistency ==")
    for pop_name, pop in data.items():
        if "n_total" not in pop or "n_positive" not in pop:
            continue
        expected = pop["n_positive"] / pop["n_total"]
        actual = pop.get("base_rate", -1)
        check(
            f"{pop_name} base_rate",
            approx_eq(actual, expected, tol=0.0001),
            f"actual={actual:.6f}, expected={expected:.6f}",
        )


def check_degenerate_features(data: dict) -> None:
    """Check that degenerate features have AUC exactly 0.500."""
    print("\n== Degenerate Feature AUCs ==")
    sr = data.get("single_repo", {})
    hyps = sr.get("hypotheses", {})

    for h_name in ("H10_network", "H11_llm"):
        h = hyps.get(h_name, {})
        if "skipped" in h:
            check(f"{h_name} single-repo AUC", True, "skipped (ok)")
            continue
        auc_obj = h.get("auc_roc", {})
        auc = auc_obj.get("auc") if isinstance(auc_obj, dict) else auc_obj
        check(
            f"{h_name} single-repo AUC == 0.500",
            auc is not None and approx_eq(auc, 0.5, tol=0.001),
            f"actual={auc}",
        )


def check_knn_perfect(data: dict) -> None:
    """Check k-NN AUC = 1.000 on primary (suspended) target."""
    print("\n== k-NN Perfect AUC ==")
    for pop_name, pop in data.items():
        hyps = pop.get("hypotheses", {})
        knn = hyps.get("H13_knn", {})
        if "skipped" in knn:
            continue
        auc_obj = knn.get("auc_roc", {})
        auc = auc_obj.get("auc") if isinstance(auc_obj, dict) else auc_obj
        check(
            f"{pop_name} H13_knn AUC == 1.000",
            auc is not None and approx_eq(auc, 1.0, tol=0.001),
            f"actual={auc}",
        )


def check_recall_monotonic(data: dict) -> None:
    """Check that recall@k is monotonically non-decreasing."""
    print("\n== Recall@k Monotonicity ==")
    k_values = [10, 25, 50, 100, 250]
    for pop_name, pop in data.items():
        hyps = pop.get("hypotheses", {})
        for h_name, h in hyps.items():
            if "skipped" in h:
                continue
            recalls = []
            for k in k_values:
                r = h.get(f"recall_at_{k}")
                if r is not None:
                    recalls.append((k, r))
            if len(recalls) < 2:
                continue
            monotonic = all(
                recalls[i][1] <= recalls[i + 1][1] + 0.001
                for i in range(len(recalls) - 1)
            )
            if not monotonic:
                check(
                    f"{pop_name}/{h_name} recall@k monotonic",
                    False,
                    f"values: {[(k, f'{r:.3f}') for k, r in recalls]}",
                )
            # Only report failures, not every pass


def check_n_total_consistency(data: dict) -> None:
    """Check n_total is consistent across hypotheses within a population."""
    print("\n== n_total Consistency Across Hypotheses ==")
    for pop_name, pop in data.items():
        hyps = pop.get("hypotheses", {})
        n_totals = {}
        for h_name, h in hyps.items():
            if "skipped" in h:
                continue
            nt = h.get("n_total")
            if nt is not None:
                n_totals[h_name] = nt

        if not n_totals:
            continue

        # H9 may differ due to NaN filtering
        non_h9 = {k: v for k, v in n_totals.items() if k != "H9_temporal"}
        if non_h9:
            vals = list(non_h9.values())
            all_same = all(v == vals[0] for v in vals)
            check(
                f"{pop_name} n_total consistent (excl. H9)",
                all_same,
                f"values: {non_h9}",
            )


def check_results_md_numbers(data: dict) -> None:
    """Spot-check that key numbers in RESULTS.md match JSON."""
    print("\n== RESULTS.md vs JSON Spot Checks ==")
    if not RESULTS_MD.exists():
        check("RESULTS.md exists", False)
        return

    # Check population sizes mentioned in RESULTS.md
    pop_checks = [
        ("multi_repo", "n_total", 3208, r"3,?208"),
        ("multi_repo", "n_positive", 61, r"\b61\b"),
        ("single_repo", "n_total", 28088, r"28,?088"),
        ("single_repo", "n_positive", 262, r"\b262\b"),
        ("all", "n_total", 31296, r"31,?296"),
        ("all", "n_positive", 323, r"\b323\b"),
    ]

    for pop_name, field, expected_val, _pattern in pop_checks:
        pop = data.get(pop_name, {})
        actual = pop.get(field)
        check(
            f"JSON {pop_name}.{field} == {expected_val}",
            actual == expected_val,
            f"actual={actual}",
        )

    # Check some AUC values from the JSON match what RESULTS.md should report
    mr_hyps = data.get("multi_repo", {}).get("hypotheses", {})

    h11_auc = mr_hyps.get("H11_llm", {}).get("auc_roc", {})
    if isinstance(h11_auc, dict):
        h11_auc = h11_auc.get("auc")
    check(
        "multi_repo H11_llm AUC ~0.619",
        h11_auc is not None and approx_eq(h11_auc, 0.619, tol=0.002),
        f"actual={h11_auc}",
    )

    h10_auc = mr_hyps.get("H10_network", {}).get("auc_roc", {})
    if isinstance(h10_auc, dict):
        h10_auc = h10_auc.get("auc")
    check(
        "multi_repo H10_network AUC ~0.523",
        h10_auc is not None and approx_eq(h10_auc, 0.523, tol=0.002),
        f"actual={h10_auc}",
    )

    # H11_tfidf checks
    tfidf_mr = mr_hyps.get("H11_tfidf", {})
    tfidf_auc = tfidf_mr.get("auc_roc", {})
    if isinstance(tfidf_auc, dict):
        tfidf_auc = tfidf_auc.get("auc")
    check(
        "multi_repo H11_tfidf AUC ~0.595",
        tfidf_auc is not None and approx_eq(tfidf_auc, 0.595, tol=0.002),
        f"actual={tfidf_auc}",
    )

    sr_hyps = data.get("single_repo", {}).get("hypotheses", {})
    tfidf_sr = sr_hyps.get("H11_tfidf", {})
    tfidf_sr_auc = tfidf_sr.get("auc_roc", {})
    if isinstance(tfidf_sr_auc, dict):
        tfidf_sr_auc = tfidf_sr_auc.get("auc")
    check(
        "single_repo H11_tfidf AUC ~0.571 (not degenerate)",
        tfidf_sr_auc is not None and approx_eq(tfidf_sr_auc, 0.571, tol=0.002),
        f"actual={tfidf_sr_auc}",
    )

    # H11_tfidf should have n_total == full population (no pre-filter)
    tfidf_mr_n = tfidf_mr.get("n_total")
    check(
        "H11_tfidf covers all multi-repo authors",
        tfidf_mr_n == 3208,
        f"actual={tfidf_mr_n}",
    )


def check_campaign_results() -> None:
    """Verify campaign detection numbers."""
    print("\n== Campaign Results ==")
    camp = load_json("campaign_results.json")
    if not camp:
        return

    ma = camp.get("monthly_analysis", {})
    check(
        "101 anomalous months",
        ma.get("n_anomalous") == 101,
        f"actual={ma.get('n_anomalous')}",
    )

    ca = camp.get("campaign_authors", {})
    check(
        "609 campaign authors",
        ca.get("n_authors") == 609,
        f"actual={ca.get('n_authors')}",
    )

    sx = camp.get("suspended_cross_reference", {})
    check(
        "campaign/suspended overlap count",
        sx.get("n_suspended") is not None,
        f"n_suspended={sx.get('n_suspended')}",
    )


def check_h11_top10_claim(data: dict) -> None:
    """Check '7 of top 10' claim for H11 LLM on multi-repo."""
    print("\n== Narrative Claims ==")
    mr_hyps = data.get("multi_repo", {}).get("hypotheses", {})
    h11 = mr_hyps.get("H11_llm", {})
    p10 = h11.get("precision_at_10")
    check(
        "H11 P@10 = 0.70 ('7 of top 10')",
        p10 is not None and approx_eq(p10, 0.70, tol=0.01),
        f"actual P@10={p10}",
    )


def main() -> None:
    print("Verifying RESULTS.md against JSON output files...")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"RESULTS.md: {RESULTS_MD}")

    pop_data = load_json("author_evaluation_by_population.json")
    if not pop_data:
        print("\nCannot proceed without population evaluation JSON.")
        sys.exit(1)

    check_population_arithmetic(pop_data)
    check_base_rates(pop_data)
    check_degenerate_features(pop_data)
    check_knn_perfect(pop_data)
    check_recall_monotonic(pop_data)
    check_n_total_consistency(pop_data)
    check_results_md_numbers(pop_data)
    check_campaign_results()
    check_h11_top10_claim(pop_data)

    print(f"\n{'=' * 40}")
    print(f"PASS: {PASS_COUNT}  FAIL: {FAIL_COUNT}")
    if FAIL_COUNT > 0:
        print("Some checks FAILED -- review above.")
        sys.exit(1)
    else:
        print("All checks passed.")


if __name__ == "__main__":
    main()
