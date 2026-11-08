import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from movie_cf.aggregate import AGGREGATIONS, predict, round_half_up
from movie_cf.similarity import SIMILARITIES, RatingMatrix, mode, user_similarity

FEATURES = ["user_id", "movie_id"]


class UserUserRecommender(BaseEstimator):
    """User-based kNN over people who rated the same movies."""

    def __init__(self, similarity: str = "mae", aggregation: str = "vote", k: int = 2) -> None:
        self.similarity = similarity
        self.aggregation = aggregation
        self.k = k

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> "UserUserRecommender":
        if self.similarity not in SIMILARITIES:
            raise ValueError(f"Unknown similarity {self.similarity!r}. Expected one of {SIMILARITIES}.")
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(f"Unknown aggregation {self.aggregation!r}. Expected one of {AGGREGATIONS}.")
        if self.k < 1:
            raise ValueError("k must be at least 1.")
        ratings = np.asarray(y)
        frame = pd.DataFrame(
            {"user_id": np.asarray(X["user_id"]), "movie_id": np.asarray(X["movie_id"]), "rating": ratings}
        )
        self.matrix_ = RatingMatrix.from_frame(frame)
        # A negative similarity never votes or enters the average.
        self.weights_ = np.nan_to_num(np.clip(user_similarity(self.matrix_.values, self.similarity), 0.0, None))
        np.fill_diagonal(self.weights_, 0.0)
        self.fallback_ = mode(ratings)
        return self

    def estimate(self, X: pd.DataFrame) -> np.ndarray:
        """Unrounded prediction. With no eligible neighbour this is the person's most common rating."""
        if not hasattr(self, "matrix_"):
            raise RuntimeError("Call fit before estimate.")
        users = pd.Series(np.asarray(X["user_id"])).map(self.matrix_.user_index)
        movies = pd.Series(np.asarray(X["movie_id"])).map(self.matrix_.movie_index)
        known_user = users.notna().to_numpy()
        known_both = (users.notna() & movies.notna()).to_numpy()

        # Unknown person: global mode. Known person but unseen movie: that person's mode.
        raw = np.full(len(X), self.fallback_, dtype=float)
        if known_user.any():
            raw[known_user] = self.matrix_.user_modes[users[known_user].to_numpy(dtype=int)]
        if known_both.any():
            raw[known_both] = predict(
                self.weights_,
                self.matrix_.values,
                self.matrix_.user_means,
                self.matrix_.user_modes,
                self.aggregation,
                self.k,
                users[known_both].to_numpy(dtype=int),
                movies[known_both].to_numpy(dtype=int),
            )
        return raw

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return round_half_up(self.estimate(X))


class UserModeBaseline(BaseEstimator):
    """Predicts each person's most frequent training rating."""

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> "UserModeBaseline":
        ratings = np.asarray(y)
        frame = pd.DataFrame({"user_id": np.asarray(X["user_id"]), "rating": ratings})
        self.modes_ = frame.groupby("user_id")["rating"].agg(mode)
        self.fallback_ = mode(ratings)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return pd.Series(np.asarray(X["user_id"])).map(self.modes_).fillna(self.fallback_).astype(int).to_numpy()


class UserMeanBaseline(BaseEstimator):
    """Predicts each person's rounded mean training rating."""

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> "UserMeanBaseline":
        ratings = np.asarray(y, dtype=float)
        frame = pd.DataFrame({"user_id": np.asarray(X["user_id"]), "rating": ratings})
        self.means_ = frame.groupby("user_id")["rating"].mean()
        self.fallback_ = float(ratings.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        means = pd.Series(np.asarray(X["user_id"])).map(self.means_).fillna(self.fallback_)
        return round_half_up(means.to_numpy())
