"""Movie similarity based on how other users rated the movies (adjusted cosine of user-centered ratings).

For a given user the similarity is always computed without that user's own ratings. At prediction time the user's
rating of the target movie is unknown, so excluding their column makes leave-one-out validation match that setting
and prevents the held-out rating from leaking into the similarity matrix.
"""

import numpy as np
import pandas as pd


class RatingPatternSimilarity:
    def __init__(self, movie_ids: np.ndarray) -> None:
        self.movie_ids = movie_ids

    def fit(self, train: pd.DataFrame) -> "RatingPatternSimilarity":
        ratings = train.pivot(index="movie_id", columns="user_id", values="rating").reindex(self.movie_ids)
        centered = ratings - ratings.mean(axis=0)
        self._centered = centered.fillna(0.0).to_numpy()
        self._user_column = pd.Series(np.arange(centered.shape[1]), index=centered.columns)
        self._dot = self._centered @ self._centered.T
        self._squared_norm = np.diag(self._dot).copy()

        full = self._cosine(self._dot, self._squared_norm)
        off_diagonal = full[~np.eye(len(full), dtype=bool)]
        self._mean, self._std = off_diagonal.mean(), off_diagonal.std()
        return self

    def for_user(self, user_id: int, rows: np.ndarray | None = None, columns: np.ndarray | None = None) -> np.ndarray:
        """Standardized similarity between movies (by position) computed without the given user's ratings."""
        rows = np.arange(len(self.movie_ids)) if rows is None else rows
        columns = rows if columns is None else columns
        own = self._own_ratings(user_id)
        dot = self._dot[np.ix_(rows, columns)] - np.outer(own[rows], own[columns])
        squared_norm = self._squared_norm - own**2
        similarity = self._cosine(dot, squared_norm[rows], squared_norm[columns])
        return (similarity - self._mean) / self._std

    def _own_ratings(self, user_id: int) -> np.ndarray:
        if user_id not in self._user_column.index:
            return np.zeros(len(self.movie_ids))
        return self._centered[:, self._user_column[user_id]]

    @staticmethod
    def _cosine(dot: np.ndarray, row_norm: np.ndarray, column_norm: np.ndarray | None = None) -> np.ndarray:
        column_norm = row_norm if column_norm is None else column_norm
        denominator = np.sqrt(np.outer(row_norm, column_norm))
        return np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator > 0)
