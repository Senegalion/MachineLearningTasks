from collections.abc import Callable
from typing import Protocol

import numpy as np
import pandas as pd

from movie_dt.config import RANDOM_STATE


class Recommender(Protocol):
    def fit(self, train: pd.DataFrame) -> "Recommender": ...

    def predict(self, queries: pd.DataFrame) -> pd.Series: ...


def metrics(truth: pd.Series, predicted: pd.Series) -> dict[str, float]:
    error = predicted.to_numpy() - truth.to_numpy()
    absolute_error = np.abs(error)
    return {
        "accuracy": float(np.mean(error == 0)),
        "accuracy_pm1": float(np.mean(absolute_error <= 1)),
        "mae": float(np.mean(absolute_error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }


def per_user_folds(ratings: pd.DataFrame, n_folds: int) -> pd.Series:
    """Assigns folds within each user, so every user keeps the same share of ratings in every training split."""
    rng = np.random.default_rng(RANDOM_STATE)
    shuffled_rank = ratings.groupby("user_id")["id"].transform(lambda ids: rng.permutation(len(ids)))
    return shuffled_rank % n_folds


def cross_validate(make_model: Callable[[], Recommender], ratings: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame:
    """Cross-validation on per-user folds."""
    folds = per_user_folds(ratings, n_folds)
    results = []
    for fold in range(n_folds):
        train, test = ratings[folds != fold], ratings[folds == fold]
        predicted = make_model().fit(train).predict(test)
        results.append({"fold": fold, **metrics(test["rating"], predicted)})
    return pd.DataFrame(results).set_index("fold")
