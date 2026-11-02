import pandas as pd

from movie_dt.collaborative import RatingFeatures


def test_user_column_is_dropped_and_missing_ratings_use_rater_mean():
    train = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "user_id": [10, 10, 20, 30],
            "movie_id": [1, 2, 1, 2],
            "rating": [5, 1, 3, 5],
        }
    )
    features = RatingFeatures().fit(train)
    values, names = features.for_user(10, [1, 2])
    assert "user:10" not in names
    assert names == ("user:20", "user:30")
    # user 20 rated movie 1 as 3 and never rated movie 2, so movie 2 gets that user's mean (3)
    assert values[0, 0] == 3
    assert values[1, 0] == 3
    assert values[1, 1] == 5
