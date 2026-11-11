import numpy as np

from movie_cf.config import RATING_MAX, RATING_MIN

AGGREGATIONS = ("mean", "vote")


def round_half_up(values: np.ndarray) -> np.ndarray:
    # 0.5 rounds away from zero, then the result is clipped to the rating scale.
    return np.clip(np.floor(np.asarray(values, dtype=float) + 0.5), RATING_MIN, RATING_MAX).astype(int)


def predict(
    weights: np.ndarray,
    ratings: np.ndarray,
    user_means: np.ndarray,
    user_modes: np.ndarray,
    aggregation: str,
    k: int,
    users: np.ndarray,
    movies: np.ndarray,
) -> np.ndarray:
    """Predict each pair from the k most similar people who rated that movie.

    A neighbour needs a positive similarity. Nobody eligible means the person's most common rating.
    """
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"Unknown aggregation {aggregation!r}. Expected one of {AGGREGATIONS}.")
    # Closest person first. Stable, so equal similarities keep their original order.
    order = np.argsort(-weights, axis=1, kind="stable")
    predicted = np.empty(len(users), dtype=float)
    for index, (person, movie) in enumerate(zip(users, movies)):
        person, movie = int(person), int(movie)
        neighbours = _nearest(order[person], weights[person], ratings[:, movie], int(k))
        if len(neighbours) == 0:
            predicted[index] = user_modes[person]
        elif aggregation == "mean":
            predicted[index] = np.average(ratings[neighbours, movie], weights=weights[person, neighbours])
        else:
            predicted[index] = _vote(ratings[neighbours, movie], weights[person, neighbours], user_means[person])
    return predicted


def _nearest(ranked_people: np.ndarray, weight_row: np.ndarray, movie_ratings: np.ndarray, k: int) -> list[int]:
    """First k people in `ranked_people` who have a positive weight and a rating for this movie."""
    chosen = []
    for other in ranked_people:
        if weight_row[other] <= 0:
            break
        if np.isfinite(movie_ratings[other]):
            chosen.append(int(other))
            if len(chosen) == k:
                break
    return chosen


def _vote(neighbour_ratings: np.ndarray, neighbour_weights: np.ndarray, user_mean: float) -> float:
    """Rating with the largest total similarity. A tie goes to the rating closest to this person's mean.

    Equal distance keeps the lower rating.
    """
    totals = np.zeros(RATING_MAX + 1)
    for rating, weight in zip(neighbour_ratings, neighbour_weights):
        totals[int(rating)] += weight
    tied = np.flatnonzero(totals == totals.max())
    closest = tied[np.abs(tied - user_mean) == np.abs(tied - user_mean).min()]
    return float(closest[0])
