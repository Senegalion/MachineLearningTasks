"""Compare the handwritten user-based kNN with Surprise's KNNBasic on the same ratings.

Surprise covers the weighted-mean aggregation with Pearson similarity. The vote
aggregation and the mean-absolute-difference similarity are not in that library.
"""

import numpy as np
import pandas as pd
from surprise import Dataset, KNNBasic, Reader

from movie_cf.aggregate import round_half_up
from movie_cf.model import UserUserRecommender

RATINGS = pd.DataFrame(
    [
        (1, 10, 5),
        (1, 11, 3),
        (1, 12, 4),
        (2, 10, 4),
        (2, 11, 3),
        (2, 12, 5),
        (2, 13, 5),
        (3, 10, 1),
        (3, 11, 2),
        (3, 12, 1),
        (3, 13, 2),
        (4, 10, 5),
        (4, 11, 4),
        (4, 12, 4),
        (4, 13, 3),
    ],
    columns=["user_id", "movie_id", "rating"],
)
QUERIES = pd.DataFrame({"user_id": [1, 1], "movie_id": [13, 13]})


def test_pearson_weighted_mean_matches_surprise_knn_basic():
    for k in (2, 10):
        ours = UserUserRecommender(similarity="pearson", aggregation="mean", k=k)
        ours.fit(RATINGS[["user_id", "movie_id"]], RATINGS["rating"])
        estimated = ours.estimate(QUERIES.iloc[[0]])

        reader = Reader(rating_scale=(0, 5))
        data = Dataset.load_from_df(RATINGS, reader)
        algo = KNNBasic(k=k, sim_options={"name": "pearson", "user_based": True}, verbose=False)
        algo.fit(data.build_full_trainset())
        surprise = algo.predict(1, 13).est

        np.testing.assert_allclose(estimated[0], surprise, atol=1e-6)
        assert ours.predict(QUERIES.iloc[[0]])[0] == round_half_up([surprise])[0]
