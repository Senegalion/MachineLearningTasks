"""Other users' ratings of a movie, used as extra numeric features for a given person's tree."""

import numpy as np
import pandas as pd


class RatingFeatures:
    def fit(self, train: pd.DataFrame) -> "RatingFeatures":
        pivot = train.pivot(index="movie_id", columns="user_id", values="rating")
        self.movie_ids = pivot.index.to_numpy()
        self.user_ids = pivot.columns.to_numpy()
        self.ratings = pivot.to_numpy(dtype=np.float64)
        user_means = np.nanmean(self.ratings, axis=0)
        global_mean = np.nanmean(self.ratings)
        self.user_means = np.where(np.isnan(user_means), global_mean, user_means)
        filled = self.ratings.copy()
        missing = np.isnan(filled)
        filled[missing] = self.user_means[np.nonzero(missing)[1]]
        self.filled = filled
        self._movie_row = pd.Series(np.arange(len(self.movie_ids)), index=self.movie_ids)
        self._user_column = pd.Series(np.arange(len(self.user_ids)), index=self.user_ids)
        return self

    def for_user(self, user_id: int, movie_ids: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
        rows = self._movie_row.reindex(movie_ids)
        present = rows.notna().to_numpy()
        row_index = rows.fillna(0).astype(int).to_numpy()

        if user_id in self._user_column.index:
            exclude = int(self._user_column[user_id])
            columns = np.arange(len(self.user_ids)) != exclude
        else:
            columns = np.ones(len(self.user_ids), dtype=bool)

        values = np.tile(self.user_means[columns], (len(movie_ids), 1))
        if present.any():
            values[present] = self.filled[np.ix_(row_index[present], columns)]
        names = tuple(f"user:{other}" for other in self.user_ids[columns])
        return values, names
