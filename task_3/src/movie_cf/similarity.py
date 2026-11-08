import numpy as np
import pandas as pd

from movie_cf.config import RATING_MAX, RATING_MIN

SIMILARITIES = ("pearson", "mae")
RATING_SPAN = RATING_MAX - RATING_MIN


class RatingMatrix:
    """Users × movies matrix. Missing ratings stay NaN and never enter a similarity."""

    def __init__(self, values: np.ndarray, user_index: pd.Series, movie_index: pd.Series) -> None:
        self.values = values
        self.user_index = user_index
        self.movie_index = movie_index
        self.user_means = np.nanmean(values, axis=1)
        self.user_modes = np.array([mode(row) for row in values], dtype=float)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "RatingMatrix":
        table = frame.pivot(index="user_id", columns="movie_id", values="rating")
        table = table.sort_index().sort_index(axis=1)
        values = table.to_numpy(dtype=float)
        user_index = pd.Series(np.arange(len(table.index)), index=table.index)
        movie_index = pd.Series(np.arange(len(table.columns)), index=table.columns)
        return cls(values, user_index, movie_index)


def mode(values: np.ndarray) -> int:
    """Most common rating. Ties go to the higher rating."""
    observed = np.asarray(values, dtype=float)
    observed = observed[np.isfinite(observed)].astype(int)
    counts = np.bincount(observed, minlength=RATING_MAX + 1)
    return int(np.flatnonzero(counts == counts.max())[-1])


def user_similarity(ratings: np.ndarray, kind: str) -> np.ndarray:
    """Pairwise person similarity. The diagonal is 0, so a person is not their own neighbour."""
    if kind == "pearson":
        similarity = _pearson(ratings)
    elif kind == "mae":
        similarity = _mae(ratings)
    else:
        raise ValueError(f"Unknown similarity {kind!r}. Expected one of {SIMILARITIES}.")
    np.clip(similarity, -1.0, 1.0, out=similarity)
    np.fill_diagonal(similarity, 0.0)
    return similarity

def _pearson(ratings: np.ndarray) -> np.ndarray:
    """Pearson correlation on the movies both people rated.

    Means are the averages on that shared set. A pair with no shared movie, or with a
    constant rating on the overlap, gets similarity 0.
    """
    mask = np.isfinite(ratings)
    present = mask.astype(float)
    filled = np.where(mask, ratings, 0.0)
    # overlap[i, j] counts movies both rated. The other products are sums over that same set:
    # missing ratings were filled with 0 and then multiplied by the other person's mask.
    overlap = present @ present.T
    sum_ratings = filled @ present.T
    sum_squares = (filled**2) @ present.T
    dot = filled @ filled.T

    mean_i = np.divide(sum_ratings, overlap, out=np.zeros_like(sum_ratings), where=overlap > 0)
    mean_j = mean_i.T
    covariance = dot - overlap * mean_i * mean_j
    variance_i = np.maximum(sum_squares - overlap * mean_i**2, 0.0)
    variance_j = variance_i.T
    denominator = np.sqrt(variance_i * variance_j)
    return np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 1e-12)

def _mae(ratings: np.ndarray) -> np.ndarray:
    """1 − (mean absolute difference) / rating span, on movies both people rated."""
    mask = np.isfinite(ratings)
    present = mask.astype(float)
    overlap = present @ present.T
    # One movie at a time: add |ra − rb| for every pair who both rated it.
    absolute_sum = np.zeros((ratings.shape[0], ratings.shape[0]), dtype=float)
    for movie in range(ratings.shape[1]):
        rated_by = np.flatnonzero(mask[:, movie])
        values = ratings[rated_by, movie]
        absolute_sum[np.ix_(rated_by, rated_by)] += np.abs(values[:, None] - values[None, :])
    mean_absolute = np.divide(absolute_sum, overlap, out=np.zeros_like(absolute_sum), where=overlap > 0)
    return np.where(overlap > 0, 1.0 - mean_absolute / RATING_SPAN, 0.0)
