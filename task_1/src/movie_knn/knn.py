from dataclasses import dataclass

import numpy as np

from movie_knn.config import RATINGS

CLASSES = np.asarray(RATINGS)


@dataclass(frozen=True)
class Method:
    aggregation: str
    weighting: str

    def __str__(self) -> str:
        return f"{self.aggregation}/{self.weighting}"


AGGREGATIONS = ("vote", "median", "mean")
WEIGHTINGS = ("uniform", "similarity")
METHODS = tuple(Method(aggregation, weighting) for aggregation in AGGREGATIONS for weighting in WEIGHTINGS)


def predict_all_k(similarity: np.ndarray, ratings: np.ndarray, max_k: int) -> np.ndarray:
    max_k = min(max_k, similarity.shape[-1])
    order = np.argsort(-similarity, axis=-1, kind="stable")[..., :max_k]
    neighbour_similarity = np.take_along_axis(similarity, order, axis=-1)
    neighbour_ratings = np.take_along_axis(ratings[..., None, :], order, axis=-1)
    one_hot = neighbour_ratings[..., None] == CLASSES

    predictions = {}
    for weighting in WEIGHTINGS:
        if weighting == "uniform":
            weights = np.ones_like(neighbour_similarity)
        else:
            weights = np.exp(neighbour_similarity - neighbour_similarity[..., :1])
        histogram = np.cumsum(one_hot * weights[..., None], axis=-2)
        total = histogram.sum(axis=-1)
        mean = np.cumsum(weights * neighbour_ratings, axis=-1) / total
        tie_break = 1e-9 * np.abs(CLASSES - mean[..., None])

        predictions[Method("vote", weighting)] = CLASSES[np.argmax(histogram - tie_break, axis=-1)]
        predictions[Method("median", weighting)] = CLASSES[
            np.argmax(np.cumsum(histogram, axis=-1) >= total[..., None] / 2, axis=-1)
        ]
        predictions[Method("mean", weighting)] = np.clip(np.floor(mean + 0.5), CLASSES[0], CLASSES[-1]).astype(int)

    stacked = np.stack([predictions[method] for method in METHODS])
    return np.moveaxis(stacked, -1, 1)


def leave_one_out_predictions(similarity: np.ndarray, ratings: np.ndarray, max_k: int) -> np.ndarray:
    n = similarity.shape[-1]
    masked = np.where(np.eye(n, dtype=bool), -np.inf, similarity)
    return predict_all_k(masked, ratings, min(max_k, n - 1))


def objective(predictions: np.ndarray, truth: np.ndarray, mae_penalty: float) -> np.ndarray:
    accuracy = (predictions == truth).mean(axis=-1)
    mae = np.abs(predictions - truth).mean(axis=-1)
    return accuracy - mae_penalty * mae


def smooth_over_k(scores: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return scores
    pad = window // 2
    padded = np.pad(scores, [(0, 0), (pad, pad)] + [(0, 0)] * (scores.ndim - 2), mode="edge")
    kernel = np.ones(window) / window
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded)
