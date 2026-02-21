#!/usr/bin/env python3
"""Train v2 (Better Egg) combined model weights from validation study data.

Reads the validation study feature data, fits a logistic regression
on [normalized_score, author_merge_rate, log_account_age_days], and
prints the coefficients for hardcoding into V2CombinedModelConfig.

Requirements: pandas, scikit-learn (from the validation study env).
Run: ``python scripts/train_v2_weights.py``
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

FEATURES_PATH = Path("experiments/validation/data/features/features.parquet")
FEATURE_COLS = ["normalized_score", "author_merge_rate", "log_account_age_days"]


def main() -> None:
    if not FEATURES_PATH.exists():
        print(f"Error: {FEATURES_PATH} not found.", file=sys.stderr)
        print("Run from the repo root with validation data present.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(FEATURES_PATH)
    df["y"] = (df["outcome"] == "merged").astype(int)

    # Use subset with temporally-scoped merge rate data
    mask = df["author_merge_rate"].notna()
    sub = df[mask].copy()
    print(f"Training on {len(sub)} PRs (of {len(df)} total)")
    print(f"Base merge rate: {sub.y.mean():.3f}")

    features = sub[FEATURE_COLS].values
    y = sub["y"].values

    model = LogisticRegression(C=np.inf, max_iter=10000, solver="lbfgs")
    model.fit(features, y)

    print("\n=== v2 Combined Model Coefficients ===")
    print(f"intercept:          {model.intercept_[0]:.6f}")
    for name, coef in zip(FEATURE_COLS, model.coef_[0], strict=True):
        print(f"{name:24s} {coef:.6f}")

    y_pred = model.predict_proba(features)[:, 1]
    print(f"\nCombined AUC: {roc_auc_score(y, y_pred):.4f}")
    print(f"Graph-only AUC: {roc_auc_score(y, sub['normalized_score'].values):.4f}")

    print("\n=== For V2CombinedModelConfig defaults ===")
    print(f"intercept: float = {model.intercept_[0]:.4f}")
    print(f"graph_score_weight: float = {model.coef_[0][0]:.4f}")
    print(f"merge_rate_weight: float = {model.coef_[0][1]:.4f}")
    print(f"account_age_weight: float = {model.coef_[0][2]:.4f}")


if __name__ == "__main__":
    main()
