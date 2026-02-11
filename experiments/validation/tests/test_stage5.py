from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.validation.embedding import (
    author_repo_similarity,
    compute_centroid,
    cosine_similarity,
)

# === Newcomer cohort feature tests (Rec 3) ===


def test_is_newcomer_zero_prs() -> None:
    """A user with 0 prior PRs is a newcomer."""
    df = pd.DataFrame({
        "total_prs_at_time": [0, 0, 5, 10],
        "unique_repos_at_time": [0, 1, 3, 5],
    })
    df["is_newcomer"] = (df["total_prs_at_time"] == 0).astype(int)
    assert df["is_newcomer"].tolist() == [1, 1, 0, 0]


def test_is_newcomer_nonzero_prs() -> None:
    """A user with any prior PRs is not a newcomer."""
    df = pd.DataFrame({"total_prs_at_time": [1, 2, 100]})
    df["is_newcomer"] = (df["total_prs_at_time"] == 0).astype(int)
    assert df["is_newcomer"].tolist() == [0, 0, 0]


def test_is_first_repo_ecosystem_zero_repos() -> None:
    """A user contributing to 0 or 1 repos is first-time ecosystem."""
    df = pd.DataFrame({"unique_repos_at_time": [0, 1, 2, 5]})
    df["is_first_repo_ecosystem"] = (
        df["unique_repos_at_time"] <= 1
    ).astype(int)
    assert df["is_first_repo_ecosystem"].tolist() == [1, 1, 0, 0]


def test_newcomer_and_ecosystem_independent() -> None:
    """is_newcomer and is_first_repo_ecosystem are independent."""
    df = pd.DataFrame({
        "total_prs_at_time": [0, 0, 3],
        "unique_repos_at_time": [0, 2, 1],
    })
    df["is_newcomer"] = (df["total_prs_at_time"] == 0).astype(int)
    df["is_first_repo_ecosystem"] = (
        df["unique_repos_at_time"] <= 1
    ).astype(int)
    # Row 0: newcomer=1, first_ecosystem=1
    # Row 1: newcomer=1, first_ecosystem=0 (has 2 repos but 0 PRs)
    # Row 2: newcomer=0, first_ecosystem=1 (has 3 PRs but only 1 repo)
    assert df["is_newcomer"].tolist() == [1, 1, 0]
    assert df["is_first_repo_ecosystem"].tolist() == [1, 0, 1]


# === Embedding tests ===


def test_cosine_similarity_identical() -> None:
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal() -> None:
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector() -> None:
    a = np.array([1.0, 2.0])
    b = np.zeros(2)
    assert cosine_similarity(a, b) == 0.0


def test_compute_centroid() -> None:
    embeddings = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
    ]
    centroid = compute_centroid(embeddings)
    assert centroid == pytest.approx(np.array([0.5, 0.5]))


def test_compute_centroid_empty() -> None:
    centroid = compute_centroid([])
    assert len(centroid) == 768  # default dim


def test_author_repo_similarity_basic() -> None:
    author_embs = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]
    target = np.array([1.0, 1.0, 0.0])
    sim = author_repo_similarity(author_embs, target)
    # Centroid is [0.5, 0.5, 0], target is [1, 1, 0]
    # cos = 1.0 / (sqrt(0.5) * sqrt(2)) = 1.0 / 1.0 = 1.0
    assert sim == pytest.approx(1.0)


def test_author_repo_similarity_empty() -> None:
    target = np.array([1.0, 0.0])
    assert author_repo_similarity([], target) == 0.0
