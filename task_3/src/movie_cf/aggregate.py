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
    """Predict the listed pairs from the k nearest people with positive similarity who rated that movie.

    Nobody eligible means the person's most common rating.
    """
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"Unknown aggregation {aggregation!r}. Expected one of {AGGREGATIONS}.")
    mask = np.isfinite(ratings)
    k = min(int(k), ratings.shape[0] - 1)
    predicted = np.empty(len(users), dtype=float)
    for person in np.unique(users):
        rows = np.flatnonzero(users == person)
        predicted[rows] = _person_scores(
            weights[int(person)],
            ratings,
            mask,
            user_means[int(person)],
            user_modes[int(person)],
            aggregation,
            k,
            movies[rows],
        )
    return predicted


def _person_scores(weight_row, ratings, mask, user_mean, user_mode, aggregation, k, columns):
    # One row of similarities: this person against everyone else, closest first.
    # Stable sort so equal similarities keep their original order.
    order = np.argsort(-weight_row, kind="stable")
    ranked_weights = weight_row[order]
    # columns are the movies asked about. Each column is scored independently.
    neighbour_ratings = ratings[np.ix_(order, columns)]
    neighbour_mask = mask[np.ix_(order, columns)]
    # A neighbour counts only with a positive similarity and a rating for that movie.
    eligible = neighbour_mask & (ranked_weights[:, None] > 0)
    # cumsum numbers the eligible neighbours from the closest one, so later ones drop out.
    chosen = eligible & (np.cumsum(eligible, axis=0) <= k)
    chosen_weights = np.where(chosen, ranked_weights[:, None], 0.0)
    if aggregation == "mean":
            return _mean(chosen_weights, neighbour_ratings, neighbour_mask, user_mode)
    elif aggregation == "vote":
            return _vote(chosen_weights, neighbour_ratings, neighbour_mask, user_mean, user_mode)
    else:
        raise ValueError(f"Unknown aggregation {aggregation!r}. Expected one of {AGGREGATIONS}.")


def _mean(chosen_weights, neighbour_ratings, neighbour_mask, user_mode):
    # Missing ratings are 0 here, but their weight is already 0, so they add nothing.
    observed = np.where(neighbour_mask, neighbour_ratings, 0.0)
    numer = (chosen_weights * observed).sum(axis=0)
    denom = chosen_weights.sum(axis=0)
    # No eligible neighbour: fall back to this person's most common rating.
    return np.divide(numer, denom, out=np.full(denom.shape, user_mode), where=denom > 0)


def _vote(chosen_weights, neighbour_ratings, neighbour_mask, user_mean, user_mode):
    classes = np.arange(RATING_MIN, RATING_MAX + 1)
    # votes[rating] is the total similarity of neighbours who gave that rating.
    votes = [
        (chosen_weights * (neighbour_mask & (neighbour_ratings == rating))).sum(axis=0) for rating in classes
    ]
    votes = np.stack(votes)
    total = votes.sum(axis=0)
    # Among the ratings tied for the largest total, pick the one closest to this person's mean.
    distance = np.abs(classes[:, None] - user_mean)
    preference = np.where(votes == votes.max(axis=0, keepdims=True), -distance, -np.inf)
    choice = preference.argmax(axis=0).astype(float)
    # classes are 0..5, so the winning row index is the rating itself.
    return np.where(total > 0, choice, user_mode)
