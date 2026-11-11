import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import cross_validate as sklearn_cross_validate

from movie_cf.config import RANDOM_STATE
from movie_cf.model import FEATURES

SCORING = {
    "accuracy": "accuracy",
    "accuracy_pm1": make_scorer(lambda truth, predicted: np.mean(np.abs(predicted - truth) <= 1)),
    "mae": make_scorer(mean_absolute_error),
    "rmse": make_scorer(root_mean_squared_error),
}


def cross_validator(user_ids, n_folds: int = 5) -> PredefinedSplit:
    """Folds are dealt inside each person, so every person stays in every training split."""
    rng = np.random.default_rng(RANDOM_STATE)
    frame = pd.DataFrame({"user_id": np.asarray(user_ids), "row": np.arange(len(user_ids))})
    # Shuffle inside each person, then deal those ratings out round-robin across folds.
    shuffled_rank = frame.groupby("user_id")["row"].transform(lambda rows: rng.permutation(len(rows)))
    return PredefinedSplit((shuffled_rank % n_folds).to_numpy())


def cross_validate(make_model, ratings: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame:
    scores = sklearn_cross_validate(
        make_model(),
        ratings[FEATURES],
        ratings["rating"],
        cv=cross_validator(ratings["user_id"], n_folds),
        scoring=SCORING,
    )
    return pd.DataFrame({name: scores[f"test_{name}"] for name in SCORING}).rename_axis("fold")
