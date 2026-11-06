import numpy as np
import pandas as pd

from movie_dt.features import TabularFeatures
from movie_dt.model import ModelConfig, PerUserClassifier, UserModeBaseline
from movie_dt.tree import TreeConfig


def test_per_user_tree_fits_independent_classifiers():
    content = TabularFeatures(
        feature_names=("year", "action"),
        movie_ids=np.array([1, 2, 3, 4]),
        values=np.array(
            [
                [1990.0, 1.0],
                [2000.0, 1.0],
                [2010.0, 0.0],
                [2020.0, 0.0],
            ]
        ),
    )
    train = pd.DataFrame(
        {
            "id": range(8),
            "user_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "movie_id": [1, 2, 3, 4, 1, 2, 3, 4],
            "rating": [5, 5, 1, 1, 0, 0, 4, 4],
        }
    )
    model = PerUserClassifier(
        content,
        ModelConfig(
            kind="tree",
            tree=TreeConfig(max_depth=2, min_samples_split=2, min_samples_leaf=1, min_impurity_decrease=0.0),
        ),
    ).fit(train)
    assert list(model.predict(train)) == list(train["rating"])
    assert set(model.models_) == {1, 2}


def test_user_mode_baseline():
    train = pd.DataFrame({"id": [1, 2, 3], "user_id": [7, 7, 7], "movie_id": [1, 2, 3], "rating": [5, 5, 1]})
    queries = pd.DataFrame({"id": [4], "user_id": [7], "movie_id": [4], "rating": [None]})
    predicted = UserModeBaseline().fit(train).predict(queries)
    assert int(predicted.iloc[0]) == 5
