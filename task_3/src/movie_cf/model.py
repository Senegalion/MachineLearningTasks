import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from movie_cf.aggregate import predict, round_half_up
from movie_cf.similarity import RatingMatrix, mode, user_similarity

FEATURES = ["user_id", "movie_id"]


class UserUserRecommender(BaseEstimator):
    """User-based kNN over people who rated the same movies."""

    def __init__(self, similarity: str = "mae", aggregation: str = "vote", k: int = 2) -> None:
        self.similarity = similarity
        self.aggregation = aggregation
        self.k = k

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> "UserUserRecommender":
        ratings = np.asarray(y)
        frame = pd.DataFrame(
            {"user_id": np.asarray(X["user_id"]), "movie_id": np.asarray(X["movie_id"]), "rating": ratings}
        )
        # Build users x movies matrix
        self.matrix_ = RatingMatrix.from_frame(frame)
        # Compute user similarity matrix (users x users)
        self.weights_ = np.clip(user_similarity(self.matrix_.values, self.similarity), 0.0, None)
        # Fallback is the mode of all ratings of all users, it is used if the 
        # prediction is not possible, because no user and no movie were seen in the training set
        self.fallback_ = mode(ratings)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Rounded prediction. With no eligible neighbour this is the person's most common rating."""
        if not hasattr(self, "matrix_"):
            raise RuntimeError("Call fit before estimate.")
        # Map user/movie ids to their row/column in the matrix
        users = pd.Series(np.asarray(X["user_id"])).map(self.matrix_.user_index)
        movies = pd.Series(np.asarray(X["movie_id"])).map(self.matrix_.movie_index)
        # Filter out users/movies that were not in the training set
        known_user = users.notna().to_numpy()
        known_both = (known_user & movies.notna().to_numpy())

        # Unknown person: global mode.
        prediction = np.full(len(X), self.fallback_, dtype=float)
        # Known person but unseen movie: that person's mode.
        prediction[known_user] = self.matrix_.user_modes[users[known_user].to_numpy(dtype=int)]
        # Predict the rating for a known person and a known movie
        prediction[known_both] = predict(
            self.weights_,
            self.matrix_.values,
            self.matrix_.user_modes,
            self.aggregation,
            self.k,
            users[known_both].to_numpy(dtype=int),
            movies[known_both].to_numpy(dtype=int),
        )
        return prediction


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
