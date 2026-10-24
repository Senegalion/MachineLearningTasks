import numpy as np
import pandas as pd

from movie_knn.evaluation import cross_validate, per_user_folds
from movie_knn.features import SimilarityTensor
from movie_knn.model import KNNRecommender, ModelConfig


def _synthetic_problem(n_movies=40, n_users=6, seed=0):
    """Two movie clusters; every user rates one cluster high and the other low, so kNN should be near perfect."""
    rng = np.random.default_rng(seed)
    cluster = np.arange(n_movies) % 2
    informative = np.where(cluster[:, None] == cluster[None, :], 1.0, -1.0)
    noise = 3 * rng.normal(size=(n_movies, n_movies))
    tensor = SimilarityTensor(("cluster", "noise"), np.arange(1, n_movies + 1), np.stack([informative, noise]))

    rows = []
    for user in range(n_users):
        liked = user % 2
        for movie in range(n_movies):
            rows.append({"id": len(rows), "user_id": user, "movie_id": movie + 1, "rating": 5 if cluster[movie] == liked else 1})
    return tensor, pd.DataFrame(rows)


def test_learns_informative_feature():
    tensor, ratings = _synthetic_problem()
    model = KNNRecommender(tensor, ModelConfig(max_k=10, use_rating_pattern=False)).fit(ratings)
    weights = dict(zip(model.feature_names_, model.global_weights_))
    assert weights["cluster"] > weights["noise"]


def test_cross_validation_on_separable_data():
    tensor, ratings = _synthetic_problem()
    scores = cross_validate(lambda: KNNRecommender(tensor, ModelConfig(max_k=10)), ratings, n_folds=4)
    assert scores["accuracy"].mean() > 0.95


def test_per_user_weights_and_prediction_shape():
    tensor, ratings = _synthetic_problem()
    model = KNNRecommender(tensor, ModelConfig(max_k=10, per_user_weights=True)).fit(ratings)
    predicted = model.predict(ratings)
    assert predicted.index.equals(ratings.index)
    assert set(predicted.unique()) <= {0, 1, 2, 3, 4, 5}
    assert len(model.describe()) == ratings["user_id"].nunique()


def test_folds_are_balanced_per_user():
    _, ratings = _synthetic_problem()
    folds = per_user_folds(ratings, n_folds=4)
    counts = folds.groupby(ratings["user_id"]).value_counts()
    assert counts.min() == counts.max() == 10
