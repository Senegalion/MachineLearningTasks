import numpy as np
import pandas as pd

from movie_cf.aggregate import predict, round_half_up
from movie_cf.model import UserMeanBaseline, UserModeBaseline, UserUserRecommender


def _predict(weights, ratings, means, aggregation, k, modes=None):
    n_users, n_movies = ratings.shape
    users = np.repeat(np.arange(n_users), n_movies)
    movies = np.tile(np.arange(n_movies), n_users)
    flat = predict(weights, ratings, means, means if modes is None else modes, aggregation, k, users, movies)
    return flat.reshape(n_users, n_movies)


def test_mean_uses_only_people_who_saw_the_movie_and_the_closest_k():
    weights = np.zeros((4, 4))
    weights[0, 1] = 1.0
    weights[0, 2] = 0.5
    weights[0, 3] = 10.0
    ratings = np.array([[np.nan], [5.0], [1.0], [np.nan]])
    means = np.full(4, 3.0)

    everyone = _predict(weights, ratings, means, "mean", 10)
    nearest = _predict(weights, ratings, means, "mean", 1)
    np.testing.assert_allclose(everyone[0, 0], (1 * 5 + 0.5 * 1) / 1.5)
    np.testing.assert_allclose(nearest[0, 0], 5.0)


def test_negative_similarity_is_not_a_neighbour():
    weights = np.array([[0.0, -0.8, 0.2], [-0.8, 0.0, 0.4], [0.2, 0.4, 0.0]])
    ratings = np.array([[np.nan, 1.0], [5.0, 1.0], [1.0, 1.0]])
    predicted = _predict(weights, ratings, np.zeros(3), "mean", 2)
    np.testing.assert_allclose(predicted[0, 0], 1.0)


def test_vote_breaks_ties_toward_the_person_mean():
    weights = np.zeros((3, 3))
    weights[0, 1] = 1.0
    weights[0, 2] = 1.0
    ratings = np.array([[np.nan], [5.0], [1.0]])
    high_mean = _predict(weights, ratings, np.array([4.0, 3.0, 3.0]), "vote", 2)
    assert high_mean[0, 0] == 5
    low_mean = _predict(weights, ratings, np.array([1.2, 3.0, 3.0]), "vote", 2)
    assert low_mean[0, 0] == 1


def test_empty_neighbourhood_uses_the_mode_rather_than_the_mean():
    weights = np.zeros((2, 2))
    ratings = np.array([[np.nan], [5.0]])
    means = np.array([2.4, 3.0])
    modes = np.array([5.0, 0.0])
    for aggregation in ("vote", "mean"):
        predicted = _predict(weights, ratings, means, aggregation, 2, modes)
        assert predicted[0, 0] == 5


def test_round_half_up_stays_inside_the_scale():
    assert list(round_half_up(np.array([-0.2, 0.5, 3.5, 5.6]))) == [0, 1, 4, 5]


def test_model_copies_a_similar_person_and_ignores_opposite_taste():
    train = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3],
            "movie_id": [10, 11, 10, 11, 10, 11],
            "rating": [5, 1, 5, 1, 0, 5],
        }
    )
    model = UserUserRecommender(similarity="pearson", aggregation="vote", k=2)
    model.fit(train[["user_id", "movie_id"]], train["rating"])
    queries = pd.DataFrame({"user_id": [1, 3, 1], "movie_id": [10, 10, 99]})
    assert list(model.predict(queries)) == [5, 5, 5]


def test_baselines_follow_each_person():
    train = pd.DataFrame({"user_id": [1, 1, 1, 2], "movie_id": [1, 2, 3, 1], "rating": [5, 5, 1, 0]})
    queries = pd.DataFrame({"user_id": [1, 2], "movie_id": [9, 9]})
    features = ["user_id", "movie_id"]
    assert list(UserModeBaseline().fit(train[features], train["rating"]).predict(queries)) == [5, 0]
    assert list(UserMeanBaseline().fit(train[features], train["rating"]).predict(queries)) == [4, 0]
