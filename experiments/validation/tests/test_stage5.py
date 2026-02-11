from __future__ import annotations

import numpy as np
import pytest

from experiments.validation.embedding import (
    author_repo_similarity,
    compute_centroid,
    cosine_similarity,
)


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
