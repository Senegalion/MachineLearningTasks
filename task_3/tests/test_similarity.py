import numpy as np

from movie_cf.similarity import user_similarity


def _shared(extra_column: bool) -> np.ndarray:
    ratings = np.array([[5.0, 3.0, 1.0], [4.0, 2.0, 0.0]], dtype=float)
    if extra_column:
        ratings = np.column_stack([ratings, [5.0, np.nan]])
    return ratings


def test_pearson_matches_corrcoef_and_ignores_movies_only_one_person_saw():
    rng = np.random.default_rng(0)
    ratings = rng.integers(0, 6, size=(5, 8)).astype(float)
    ratings[rng.random(ratings.shape) < 0.3] = np.nan
    similarity = user_similarity(ratings, "pearson")
    assert np.allclose(similarity, similarity.T)
    assert np.all(np.diag(similarity) == 0)
    for left, right in ((0, 1), (0, 2), (1, 4), (2, 3)):
        shared = np.isfinite(ratings[left]) & np.isfinite(ratings[right])
        if shared.sum() < 2 or np.std(ratings[left, shared]) == 0 or np.std(ratings[right, shared]) == 0:
            assert similarity[left, right] == 0
        else:
            expected = np.corrcoef(ratings[left, shared], ratings[right, shared])[0, 1]
            np.testing.assert_allclose(similarity[left, right], expected, atol=1e-8)

    plain = user_similarity(_shared(False), "pearson")
    with_extra = user_similarity(_shared(True), "pearson")
    np.testing.assert_allclose(plain, with_extra)


def test_mae_uses_only_shared_movies():
    mae = user_similarity(_shared(False), "mae")
    # |5-4| + |3-2| + |1-0| averaged is 1, on a 0–5 scale.
    np.testing.assert_allclose(mae[0, 1], 0.8)
    np.testing.assert_allclose(mae, user_similarity(_shared(True), "mae"))
    assert user_similarity(np.array([[0.0, 5.0], [0.0, 5.0]]), "mae")[0, 1] == 1
    assert user_similarity(np.array([[0.0, 0.0], [5.0, 5.0]]), "mae")[0, 1] == 0


def test_no_shared_movies_and_constant_ratings_are_not_similar():
    disjoint = user_similarity(np.array([[5.0, np.nan], [np.nan, 1.0]]), "pearson")
    assert disjoint[0, 1] == 0
    constant = user_similarity(np.array([[3.0, 3.0, 3.0], [1.0, 2.0, 5.0]]), "pearson")
    assert constant[0, 1] == 0
