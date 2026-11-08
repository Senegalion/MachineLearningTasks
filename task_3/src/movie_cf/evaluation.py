from collections.abc import Callable

import numpy as np
import pandas as pd

from movie_cf.config import RANDOM_STATE
from movie_cf.model import FEATURES


class PerUserKFold:
    """Cross-validation splitter that keeps every person in every training fold."""

    def __init__(self, n_splits: int = 5) -> None:
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        frame = pd.DataFrame({"user_id": np.asarray(X["user_id"]), "id": np.arange(len(X))})
        labels = per_user_folds(frame, self.n_splits).to_numpy()
        for fold in range(self.n_splits):
            yield np.flatnonzero(labels != fold), np.flatnonzero(labels == fold)


def metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = np.asarray(predicted) - np.asarray(truth)
    absolute = np.abs(error)
    return {
        "accuracy": float(np.mean(error == 0)),
        "accuracy_pm1": float(np.mean(absolute <= 1)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }


def objective(accuracy: float, mae: float) -> float:
    """Accuracy, with MAE as a tie-break so a noisier exact-match model does not win."""
    return accuracy - 0.01 * mae


def per_user_folds(ratings: pd.DataFrame, n_folds: int, seed: int = RANDOM_STATE) -> pd.Series:
    """Assigns folds within each person, so every person keeps ratings in every training split."""
    rng = np.random.default_rng(seed)
    # Shuffle inside each person, then deal those ratings out round-robin across folds.
    shuffled_rank = ratings.groupby("user_id")["id"].transform(lambda ids: rng.permutation(len(ids)))
    return shuffled_rank % n_folds


def cross_validate(make_model: Callable, ratings: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame:
    folds = per_user_folds(ratings, n_folds)
    results = []
    for fold in range(n_folds):
        train, test = ratings[folds != fold], ratings[folds == fold]
        predicted = make_model().fit(train[FEATURES], train["rating"]).predict(test[FEATURES])
        results.append({"fold": fold, **metrics(test["rating"].to_numpy(), predicted)})
    return pd.DataFrame(results).set_index("fold")
