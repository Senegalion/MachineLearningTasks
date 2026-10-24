import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from movie_knn.collaborative import RatingPatternSimilarity
from movie_knn.features import SimilarityTensor
from movie_knn.knn import METHODS, Method, leave_one_out_predictions, objective, predict_all_k, smooth_over_k

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    max_k: int = 60
    k_smoothing: int = 3
    mae_penalty: float = 0.01
    use_rating_pattern: bool = True
    learn_global_weights: bool = True
    weight_grid: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0)
    global_passes: int = 3
    per_user_k: bool = True
    per_user_weights: bool = False
    per_user_passes: int = 1
    per_user_min_gain: float = 0.02


@dataclass(frozen=True)
class UserParams:
    method: Method
    k: int
    weights: np.ndarray


@dataclass(frozen=True)
class _UserBatch:
    """Users with the same number of training ratings, so their data stacks into dense arrays."""

    user_ids: np.ndarray
    positions: np.ndarray
    ratings: np.ndarray
    feature_similarity: np.ndarray = field(repr=False)

    @property
    def size(self) -> int:
        return len(self.user_ids)


class KNNRecommender:
    """Per-user kNN over movies; each user gets their own neighbourhood size, aggregation and optionally feature weights.

    Movie similarity is a weighted average of standardized per-feature similarities: the fixed content features
    (TMDB) and, optionally, the rating pattern of other users, which is learned from the training ratings.
    Hyperparameters are selected with leave-one-out validation on the user's own training ratings.
    """

    RATING_PATTERN = "rating_pattern"

    def __init__(self, content: SimilarityTensor, config: ModelConfig | None = None) -> None:
        self.content = content
        self.config = config or ModelConfig()
        self.feature_names_: tuple[str, ...] = content.feature_names
        self.global_weights_: np.ndarray | None = None
        self._rating_pattern: RatingPatternSimilarity | None = None
        self.user_params_: dict[int, UserParams] = {}
        self._train: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._fallback_rating = 0

    def fit(self, train: pd.DataFrame) -> "KNNRecommender":
        if self.config.use_rating_pattern:
            self._rating_pattern = RatingPatternSimilarity(self.content.movie_ids).fit(train)
            self.feature_names_ = (*self.content.feature_names, self.RATING_PATTERN)
        batches = self._make_batches(train)
        self._train = {
            user_id: (batch.positions[i], batch.ratings[i])
            for batch in batches
            for i, user_id in enumerate(batch.user_ids)
        }
        self._fallback_rating = int(train["rating"].mode().iloc[0])

        weights = np.ones(len(self.feature_names_))
        if self.config.learn_global_weights:
            weights = self._learn_global_weights(batches, weights)
        self.global_weights_ = weights

        for batch in batches:
            user_weights = np.tile(weights, (batch.size, 1))
            if self.config.per_user_weights:
                user_weights = self._learn_user_weights(batch, user_weights)
            self.user_params_.update(self._select_neighbourhood(batch, user_weights))
        return self

    def predict(self, queries: pd.DataFrame) -> pd.Series:
        predictions = pd.Series(self._fallback_rating, index=queries.index, dtype=int)
        for user_id, rows in queries.groupby("user_id"):
            if user_id not in self.user_params_:
                logger.warning("User %s has no training ratings; using the global mode", user_id)
                continue
            params = self.user_params_[user_id]
            train_positions, train_ratings = self._train[user_id]
            query_positions = self.content.positions(rows["movie_id"].to_numpy())
            similarity = self._user_similarity(user_id, params.weights, query_positions, train_positions)
            all_k = predict_all_k(similarity, train_ratings, params.k)
            predictions.loc[rows.index] = all_k[METHODS.index(params.method), params.k - 1]
        return predictions

    def describe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "k": params.k,
                "method": str(params.method),
                **dict(zip(self.feature_names_, params.weights)),
            }
            for params in self.user_params_.values()
        ).set_index(pd.Index(list(self.user_params_), name="user_id"))

    def _make_batches(self, train: pd.DataFrame) -> list[_UserBatch]:
        sizes = train.groupby("user_id").size()
        batches = []
        for _, user_ids in sizes.groupby(sizes):
            rows = train[train["user_id"].isin(user_ids.index)].sort_values(["user_id", "movie_id"])
            n_users, n_movies = len(user_ids), int(user_ids.iloc[0])
            positions = self.content.positions(rows["movie_id"].to_numpy()).reshape(n_users, n_movies)
            ratings = rows["rating"].to_numpy().reshape(n_users, n_movies)
            feature_similarity = self.content.values[:, positions[:, :, None], positions[:, None, :]]
            if self._rating_pattern is not None:
                rating_pattern = np.stack(
                    [self._rating_pattern.for_user(u, p) for u, p in zip(user_ids.index, positions)]
                )
                feature_similarity = np.concatenate([feature_similarity, rating_pattern[None]])
            batches.append(
                _UserBatch(user_ids.index.to_numpy(), positions, ratings, feature_similarity.astype(np.float32))
            )
        return batches

    def _user_similarity(self, user_id: int, weights: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
        n_content = self.content.n_features
        content = self.content.values[:, rows][:, :, columns]
        similarity = np.tensordot(weights[:n_content], content, axes=1)
        if self._rating_pattern is not None:
            similarity += weights[n_content] * self._rating_pattern.for_user(user_id, rows, columns)
        return similarity / weights.sum()

    def _validation_scores(self, batch: _UserBatch, user_weights: np.ndarray) -> np.ndarray:
        """Smoothed leave-one-out objective with shape (n_methods, n_k, n_users)."""
        normalized = user_weights / user_weights.sum(axis=1, keepdims=True)
        similarity = np.einsum("uf,funm->unm", normalized, batch.feature_similarity)
        predictions = leave_one_out_predictions(similarity, batch.ratings, self.config.max_k)
        scores = objective(predictions, batch.ratings[None, None], self.config.mae_penalty)
        return smooth_over_k(scores, self.config.k_smoothing)

    def _best_scores(self, batch: _UserBatch, user_weights: np.ndarray) -> np.ndarray:
        scores = self._validation_scores(batch, user_weights)
        if self.config.per_user_k:
            return scores.max(axis=(0, 1))
        return scores.reshape(-1, batch.size)[np.argmax(scores.mean(axis=2))]

    def _mean_best_score(self, batches: list[_UserBatch], weights: np.ndarray) -> float:
        total = sum(self._best_scores(b, np.tile(weights, (b.size, 1))).sum() for b in batches)
        return total / sum(b.size for b in batches)

    def _learn_global_weights(self, batches: list[_UserBatch], weights: np.ndarray) -> np.ndarray:
        """Coordinate ascent over a grid of feature weights shared by all users."""
        best = self._mean_best_score(batches, weights)
        for _ in range(self.config.global_passes):
            improved = False
            for feature in range(len(weights)):
                for value in self.config.weight_grid:
                    candidate = weights.copy()
                    candidate[feature] = value
                    if value == weights[feature] or candidate.sum() == 0:
                        continue
                    score = self._mean_best_score(batches, candidate)
                    if score > best + 1e-9:
                        best, weights, improved = score, candidate, True
            logger.info("Global weights: score %.4f, %s", best, dict(zip(self.feature_names_, weights.tolist())))
            if not improved:
                break
        return weights

    def _learn_user_weights(self, batch: _UserBatch, user_weights: np.ndarray) -> np.ndarray:
        """Per-user coordinate ascent starting from the global weights.

        A change is accepted only if it beats the current score by per_user_min_gain, which limits overfitting
        to the few ratings available for a single user.
        """
        best = self._best_scores(batch, user_weights)
        for _ in range(self.config.per_user_passes):
            for feature in range(user_weights.shape[1]):
                for value in self.config.weight_grid:
                    candidate = user_weights.copy()
                    candidate[:, feature] = value
                    valid = candidate.sum(axis=1) > 0
                    candidate[~valid] = user_weights[~valid]
                    score = self._best_scores(batch, candidate)
                    accept = valid & (score > best + self.config.per_user_min_gain)
                    user_weights[accept] = candidate[accept]
                    best = np.where(accept, score, best)
        return user_weights

    def _select_neighbourhood(self, batch: _UserBatch, user_weights: np.ndarray) -> dict[int, UserParams]:
        scores = self._validation_scores(batch, user_weights)
        n_k = scores.shape[1]
        if self.config.per_user_k:
            best = scores.reshape(-1, batch.size).argmax(axis=0)
        else:
            best = np.full(batch.size, np.argmax(scores.mean(axis=2)))
        return {
            int(user_id): UserParams(METHODS[best[i] // n_k], int(best[i] % n_k) + 1, user_weights[i])
            for i, user_id in enumerate(batch.user_ids)
        }


class UserModeBaseline:
    """Predicts each user's most frequent training rating."""

    def fit(self, train: pd.DataFrame) -> "UserModeBaseline":
        self.modes_ = train.groupby("user_id")["rating"].agg(lambda r: r.mode().max())
        return self

    def predict(self, queries: pd.DataFrame) -> pd.Series:
        return queries["user_id"].map(self.modes_).astype(int)


class UserMeanBaseline:
    """Predicts each user's rounded mean training rating."""

    def fit(self, train: pd.DataFrame) -> "UserMeanBaseline":
        self.means_ = train.groupby("user_id")["rating"].mean()
        return self

    def predict(self, queries: pd.DataFrame) -> pd.Series:
        return np.floor(queries["user_id"].map(self.means_) + 0.5).astype(int)
