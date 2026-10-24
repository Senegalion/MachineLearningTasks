import numpy as np

from movie_knn.knn import METHODS, Method, leave_one_out_predictions, predict_all_k, smooth_over_k


def _prediction(all_k, method, k):
    return all_k[METHODS.index(method), k - 1]


def test_nearest_neighbour_copies_its_rating():
    similarity = np.array([[0.9, 0.1, 0.5]])
    ratings = np.array([5, 0, 2])
    all_k = predict_all_k(similarity, ratings, max_k=3)
    for method in METHODS:
        assert _prediction(all_k, method, 1)[0] == 5


def test_aggregations_over_three_neighbours():
    similarity = np.array([[3.0, 2.0, 1.0]])
    ratings = np.array([1, 1, 5])
    all_k = predict_all_k(similarity, ratings, max_k=3)
    assert _prediction(all_k, Method("vote", "uniform"), 3)[0] == 1
    assert _prediction(all_k, Method("median", "uniform"), 3)[0] == 1
    assert _prediction(all_k, Method("mean", "uniform"), 3)[0] == 2


def test_leave_one_out_never_uses_the_movie_itself():
    similarity = np.eye(4) * 10
    ratings = np.array([5, 0, 0, 0])
    all_k = leave_one_out_predictions(similarity, ratings, max_k=1)
    assert all_k.shape == (len(METHODS), 1, 4)
    assert all_k[:, 0, 0].max() == 0


def test_batched_matches_single_user():
    rng = np.random.default_rng(0)
    similarity = rng.normal(size=(3, 6, 6))
    ratings = rng.integers(0, 6, size=(3, 6))
    batched = leave_one_out_predictions(similarity, ratings, max_k=4)
    for user in range(3):
        single = leave_one_out_predictions(similarity[user], ratings[user], max_k=4)
        np.testing.assert_array_equal(batched[:, :, user], single)


def test_smoothing_keeps_shape_and_constant_rows():
    scores = np.ones((2, 5, 3))
    np.testing.assert_allclose(smooth_over_k(scores, 3), scores)
