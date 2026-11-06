"""Binary CART-style classifier: each node tests one feature with `x_j <= threshold`."""

from dataclasses import dataclass, field

import numpy as np

from movie_dt.config import N_CLASSES, RANDOM_STATE, RATINGS

CLASSES = np.asarray(RATINGS)


def gini(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    probabilities = counts / total
    return float(1.0 - np.dot(probabilities, probabilities))


def majority_label(labels: np.ndarray) -> int:
    counts = np.bincount(labels, minlength=N_CLASSES)
    return majority_from_counts(counts, labels.mean() if len(labels) else 0.0)


def majority_from_counts(counts: np.ndarray, mean: float) -> int:
    tied = np.flatnonzero(counts == counts.max())
    return int(tied[np.argmin(np.abs(tied - mean))])


def majority_vote(predictions: np.ndarray) -> np.ndarray:
    """Majority class along axis 0; ties go to the label closest to the mean prediction."""
    counts = np.stack([(predictions == label).sum(axis=0) for label in CLASSES])
    means = predictions.mean(axis=0)
    best = counts.max(axis=0)
    scores = np.where(counts == best, -np.abs(CLASSES[:, None] - means), -np.inf)
    return CLASSES[scores.argmax(axis=0)]


@dataclass
class Node:
    prediction: int
    n_samples: int
    class_counts: np.ndarray
    feature_index: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None


@dataclass(frozen=True)
class TreeConfig:
    max_depth: int = 4
    min_samples_split: int = 8
    min_samples_leaf: int = 4
    min_impurity_decrease: float = 1e-4
    max_features: int | float | str | None = None
    random_state: int = RANDOM_STATE


def _max_features(n_features: int, spec: int | float | str | None) -> int:
    if spec is None:
        return n_features
    if spec == "sqrt":
        value = int(np.sqrt(n_features))
    elif spec == "log2":
        value = int(np.log2(n_features)) if n_features > 1 else 1
    elif isinstance(spec, float) and spec < 1:
        value = int(np.ceil(spec * n_features))
    else:
        value = int(spec)
    return max(1, min(n_features, value))


def _best_split(
    x: np.ndarray,
    y: np.ndarray,
    feature_indices: np.ndarray,
    min_leaf: int,
    min_decrease: float,
) -> tuple[int, float, float] | None:
    n_samples = len(y)
    if n_samples < 2 * min_leaf:
        return None

    parent_counts = np.bincount(y, minlength=N_CLASSES).astype(np.float64)
    parent_gini = gini(parent_counts)
    subset = x[:, feature_indices]
    order = np.argsort(subset, axis=0, kind="mergesort")
    sorted_values = np.take_along_axis(subset, order, axis=0)
    sorted_y = y[order]

    left = np.stack([(sorted_y == label).cumsum(axis=0) for label in CLASSES], axis=0)
    n_left = np.arange(1, n_samples, dtype=np.float64)[:, None]
    n_right = n_samples - n_left
    left_at_cut = left[:, :-1, :]
    right_at_cut = parent_counts[:, None, None] - left_at_cut
    left_gini = 1.0 - np.sum((left_at_cut / n_left) ** 2, axis=0)
    right_gini = 1.0 - np.sum((right_at_cut / n_right) ** 2, axis=0)
    gain = parent_gini - (n_left * left_gini + n_right * right_gini) / n_samples

    valid = sorted_values[:-1] != sorted_values[1:]
    valid[: min_leaf - 1] = False
    valid[n_samples - min_leaf :] = False
    gain = np.where(valid, gain, -np.inf)
    index = np.unravel_index(int(np.argmax(gain)), gain.shape)
    best_gain = float(gain[index])
    if not np.isfinite(best_gain) or best_gain <= min_decrease:
        return None
    row, column = (int(index[0]), int(index[1]))
    threshold = 0.5 * (sorted_values[row, column] + sorted_values[row + 1, column])
    return int(feature_indices[column]), float(threshold), best_gain


class DecisionTree:
    def __init__(self, config: TreeConfig | None = None, rng: np.random.Generator | None = None) -> None:
        self.config = config or TreeConfig()
        self.rng = rng if rng is not None else np.random.default_rng(self.config.random_state)
        self.root_: Node | None = None
        self.n_features_: int = 0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DecisionTree":
        y = np.asarray(y, dtype=int)
        self.n_features_ = x.shape[1]
        self.root_ = self._grow(x, y, depth=0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.fromiter((self._predict_one(row) for row in x), dtype=int, count=len(x))

    def _predict_one(self, row: np.ndarray) -> int:
        node = self.root_
        assert node is not None
        while not node.is_leaf:
            node = node.left if row[node.feature_index] <= node.threshold else node.right
        return node.prediction

    def _grow(self, x: np.ndarray, y: np.ndarray, depth: int) -> Node:
        counts = np.bincount(y, minlength=N_CLASSES)
        node = Node(majority_from_counts(counts, y.mean()), len(y), counts)
        pure = counts.max() == len(y)
        if pure or depth >= self.config.max_depth or len(y) < self.config.min_samples_split:
            return node

        n_consider = _max_features(x.shape[1], self.config.max_features)
        if n_consider < x.shape[1]:
            features = self.rng.choice(x.shape[1], size=n_consider, replace=False)
        else:
            features = np.arange(x.shape[1])

        split = _best_split(x, y, features, self.config.min_samples_leaf, self.config.min_impurity_decrease)
        if split is None:
            return node

        feature, threshold, _gain = split
        left_mask = x[:, feature] <= threshold
        if left_mask.sum() < self.config.min_samples_leaf or (~left_mask).sum() < self.config.min_samples_leaf:
            return node

        node.feature_index = feature
        node.threshold = threshold
        node.left = self._grow(x[left_mask], y[left_mask], depth + 1)
        node.right = self._grow(x[~left_mask], y[~left_mask], depth + 1)
        return node

    def depth(self) -> int:
        def walk(node: Node | None) -> int:
            if node is None or node.is_leaf:
                return 0
            return 1 + max(walk(node.left), walk(node.right))

        return walk(self.root_)


@dataclass(frozen=True)
class ForestConfig(TreeConfig):
    n_estimators: int = 40
    bootstrap: bool = True
    max_features: int | float | str | None = "sqrt"
    max_depth: int = 4
    min_samples_split: int = 8
    min_samples_leaf: int = 4


@dataclass
class RandomForest:
    config: ForestConfig = field(default_factory=ForestConfig)
    trees_: list[DecisionTree] = field(default_factory=list)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RandomForest":
        y = np.asarray(y, dtype=int)
        rng = np.random.default_rng(self.config.random_state)
        self.trees_ = []
        n = len(y)
        for _ in range(self.config.n_estimators):
            tree_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
            if self.config.bootstrap:
                sample = tree_rng.integers(0, n, size=n)
                x_i, y_i = x[sample], y[sample]
            else:
                x_i, y_i = x, y
            tree = DecisionTree(self.config, rng=tree_rng)
            tree.fit(x_i, y_i)
            self.trees_.append(tree)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        votes = np.stack([tree.predict(x) for tree in self.trees_])
        return majority_vote(votes)
