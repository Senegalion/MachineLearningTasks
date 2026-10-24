import numpy as np
import pytest

from movie_knn.similarity import (
    categorical_similarity,
    numeric_similarity,
    set_similarity,
    standardize,
    text_similarity,
)


def test_set_similarity_is_jaccard():
    sim = set_similarity([frozenset("ab"), frozenset("bc"), frozenset()])
    assert sim[0, 1] == pytest.approx(1 / 3)
    assert sim[0, 0] == 1.0
    assert sim[2, 2] == 0.0


def test_categorical_similarity_ignores_missing():
    sim = categorical_similarity(["en", "en", None, None])
    assert sim[0, 1] == 1.0
    assert sim[2, 3] == 0.0


def test_numeric_similarity_range_and_missing():
    sim = numeric_similarity([1950.0, 2000.0, 2010.0, np.nan])
    assert sim.min() >= 0.0 and sim.max() <= 1.0
    assert not np.isnan(sim).any()
    assert sim[1, 2] > sim[0, 2]


def test_text_similarity_prefers_shared_words():
    sim = text_similarity(["space war ship", "space ship crew", "love story paris", "love story war"])
    assert sim[0, 1] > sim[0, 2]


def test_standardize_off_diagonal_moments():
    rng = np.random.default_rng(0)
    raw = rng.random((10, 10))
    z = standardize(raw)
    off = z[~np.eye(10, dtype=bool)]
    assert off.mean() == pytest.approx(0.0, abs=1e-12)
    assert off.std() == pytest.approx(1.0)
