import numpy as np
import pandas as pd

from movie_knn.collaborative import RatingPatternSimilarity


def _ratings():
    rng = np.random.default_rng(0)
    rows = [
        {"id": i, "user_id": user, "movie_id": movie, "rating": int(rng.integers(0, 6))}
        for i, (user, movie) in enumerate((u, m) for u in range(5) for m in range(1, 9))
    ]
    return pd.DataFrame(rows)


def test_similarity_for_user_ignores_their_own_ratings():
    ratings = _ratings()
    movie_ids = np.arange(1, 9)
    full = RatingPatternSimilarity(movie_ids).fit(ratings)

    changed = ratings.copy()
    changed.loc[changed["user_id"] == 0, "rating"] = 5 - changed.loc[changed["user_id"] == 0, "rating"]
    altered = RatingPatternSimilarity(movie_ids).fit(changed)

    raw_full = full.for_user(0) * full._std + full._mean
    raw_altered = altered.for_user(0) * altered._std + altered._mean
    np.testing.assert_allclose(raw_full, raw_altered, atol=1e-12)


def test_unknown_user_gets_similarity_from_all_ratings():
    ratings = _ratings()
    model = RatingPatternSimilarity(np.arange(1, 9)).fit(ratings)
    assert model.for_user(999).shape == (8, 8)
    assert not np.allclose(model.for_user(999), model.for_user(0))
