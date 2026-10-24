from collections.abc import Callable
from typing import Protocol

import numpy as np
import pandas as pd

from movie_knn.config import RANDOM_STATE


class Recommender(Protocol):
    def fit(self, train: pd.DataFrame) -> "Recommender": ...

    def predict(self, queries: pd.DataFrame) -> pd.Series: ...


def metrics(truth: pd.Series, predicted: pd.Series) -> dict[str, float]:
    error = predicted.to_numpy() - truth.to_numpy()
    return {
        "accuracy": float(np.mean(error == 0)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }


def per_user_folds(ratings: pd.DataFrame, n_folds: int, seed: int = RANDOM_STATE) -> pd.Series:
    """Assigns folds within each user, so every user keeps the same share of ratings in every training split."""
    rng = np.random.default_rng(seed)
    shuffled_rank = ratings.groupby("user_id")["id"].transform(lambda ids: rng.permutation(len(ids)))
    return shuffled_rank % n_folds


def cross_validate(
    make_model: Callable[[], Recommender], ratings: pd.DataFrame, n_folds: int = 5, seed: int = RANDOM_STATE
) -> pd.DataFrame:
    """Outer cross-validation. Any hyperparameter search inside the model only sees the training part of each fold."""
    folds = per_user_folds(ratings, n_folds, seed)
    results = []
    for fold in range(n_folds):
        train, test = ratings[folds != fold], ratings[folds == fold]
        predicted = make_model().fit(train).predict(test)
        results.append({"fold": fold, **metrics(test["rating"], predicted)})
    return pd.DataFrame(results).set_index("fold")
