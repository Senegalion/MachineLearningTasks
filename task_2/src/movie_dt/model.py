import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from movie_dt.features import TabularFeatures
from movie_dt.tree import DecisionTree, ForestConfig, RandomForest, TreeConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    tree: TreeConfig = TreeConfig()
    forest: ForestConfig = ForestConfig()
    kind: str = "tree"


class PerUserClassifier:
    """One tree or forest per person, trained on that person's rated movies."""

    def __init__(self, content: TabularFeatures, config: ModelConfig) -> None:
        self.content = content
        self.config = config
        self.models_: dict[int, DecisionTree | RandomForest] = {}
        self._unseen_user_rating = 0

    def fit(self, train: pd.DataFrame) -> "PerUserClassifier":
        self._unseen_user_rating = int(train["rating"].mode().iloc[0])
        self.models_.clear()
        for user_id, rows in train.groupby("user_id"):
            user_id = int(user_id)
            x = self.content.rows(rows["movie_id"].to_numpy())
            y = rows["rating"].to_numpy(dtype=int)
            self.models_[user_id] = self._make_estimator().fit(x, y)
        return self

    def predict(self, queries: pd.DataFrame) -> pd.Series:
        predictions = pd.Series(self._unseen_user_rating, index=queries.index, dtype=int)
        for user_id, rows in queries.groupby("user_id"):
            user_id = int(user_id)
            if user_id not in self.models_:
                logger.warning("User %s has no training ratings", user_id)
                continue
            x = self.content.rows(rows["movie_id"].to_numpy())
            predictions.loc[rows.index] = self.models_[user_id].predict(x)
        return predictions

    def _make_estimator(self) -> DecisionTree | RandomForest:
        if self.config.kind == "forest":
            return RandomForest(self.config.forest)
        return DecisionTree(self.config.tree)


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
