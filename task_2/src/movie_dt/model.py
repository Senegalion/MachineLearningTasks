import logging
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from movie_dt.collaborative import RatingFeatures
from movie_dt.features import TabularFeatures
from movie_dt.tree import DecisionTree, ForestConfig, RandomForest, TreeConfig, majority_label

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    tree: TreeConfig = TreeConfig()
    forest: ForestConfig = ForestConfig()
    use_collaborative: bool = True
    kind: str = "tree"


class PerUserClassifier:
    """Fits an independent classifier on the movies each person has already rated."""

    def __init__(self, content: TabularFeatures, config: ModelConfig | None = None) -> None:
        self.content = content
        self.config = config or ModelConfig()
        self.models_: dict[int, DecisionTree | RandomForest] = {}
        self.modes_: dict[int, int] = {}
        self.feature_names_: dict[int, tuple[str, ...]] = {}
        self._ratings: RatingFeatures | None = None
        self._fallback_rating = 0

    def fit(self, train: pd.DataFrame) -> "PerUserClassifier":
        self._ratings = RatingFeatures().fit(train) if self.config.use_collaborative else None
        self._fallback_rating = int(train["rating"].mode().iloc[0])
        self.models_.clear()
        self.modes_.clear()
        self.feature_names_.clear()

        for user_id, rows in train.groupby("user_id"):
            movie_ids = rows["movie_id"].to_numpy()
            y = rows["rating"].to_numpy(dtype=int)
            x, names = self._design_matrix(int(user_id), movie_ids)
            self.feature_names_[int(user_id)] = names
            self.modes_[int(user_id)] = majority_label(y)
            self.models_[int(user_id)] = self._make_estimator().fit(x, y)
        return self

    def predict(self, queries: pd.DataFrame) -> pd.Series:
        predictions = pd.Series(self._fallback_rating, index=queries.index, dtype=int)
        for user_id, rows in queries.groupby("user_id"):
            user_id = int(user_id)
            if user_id not in self.models_:
                logger.warning("User %s has no training ratings; using the global mode", user_id)
                continue
            x, _names = self._design_matrix(user_id, rows["movie_id"].to_numpy())
            predictions.loc[rows.index] = self.models_[user_id].predict(x)
        return predictions

    def _make_estimator(self) -> DecisionTree | RandomForest:
        if self.config.kind == "forest":
            return RandomForest(self.config.forest)
        return DecisionTree(self.config.tree)

    def _design_matrix(self, user_id: int, movie_ids: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
        content = self.content.rows(movie_ids)
        names = self.content.feature_names
        if self._ratings is None:
            return content, names
        ratings, rating_names = self._ratings.for_user(user_id, movie_ids)
        return np.hstack([content, ratings]), names + rating_names


def tree_recommender(content: TabularFeatures, **overrides) -> PerUserClassifier:
    config = ModelConfig(kind="tree")
    if overrides:
        config = replace(config, tree=replace(config.tree, **overrides))
    return PerUserClassifier(content, config)


def forest_recommender(content: TabularFeatures, **overrides) -> PerUserClassifier:
    config = ModelConfig(kind="forest")
    if overrides:
        config = replace(config, forest=replace(config.forest, **overrides))
    return PerUserClassifier(content, config)


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
