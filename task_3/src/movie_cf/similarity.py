import numpy as np
import pandas as pd

from movie_cf.config import RATING_MAX, RATING_MIN

SIMILARITIES = ("cosine", "mae", "pearson", "spearman")
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
    return int(pd.Series(np.asarray(values, dtype=float)).dropna().mode().iloc[-1])


def user_similarity(ratings: np.ndarray, kind: str) -> np.ndarray:
    """Pairwise person similarity. The diagonal is 0, so a person is not their own neighbour.

    pandas compares each pair only on the movies both rated. No shared movie, or a correlation
    that is undefined because a rating is constant, gets similarity 0.
    """
    if kind not in SIMILARITIES:
        raise ValueError(f"Unknown similarity {kind!r}. Expected one of {SIMILARITIES}.")
    people = pd.DataFrame(np.asarray(ratings, dtype=float).T)
    if kind in ("pearson", "spearman"):
        # One shared movie makes a correlation undefined. Cosine and mean absolute error are not.
        similarity = people.corr(method=kind, min_periods=2)
    else:
        similarity = people.corr(method=_cosine_similarity if kind == "cosine" else _mae_similarity)
    similarity = np.nan_to_num(similarity.to_numpy(dtype=float), nan=0.0)
    np.clip(similarity, -1.0, 1.0, out=similarity)
    np.fill_diagonal(similarity, 0.0)
    return similarity


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine similarity. pandas has already dropped movies only one person rated."""
    if len(left) == 0:
        return 0.0
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _mae_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """1 − (mean absolute error) / rating span. pandas has already dropped movies only one person rated."""
    if len(left) == 0:
        return 0.0
    return 1.0 - float(np.mean(np.abs(left - right))) / RATING_SPAN
