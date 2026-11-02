import numpy as np

from movie_dt.tree import DecisionTree, ForestConfig, RandomForest, TreeConfig, majority_vote


def test_pure_labels_make_a_leaf():
    x = np.array([[0.0], [1.0], [2.0]])
    y = np.array([4, 4, 4])
    tree = DecisionTree(TreeConfig(max_depth=3, min_samples_split=2, min_samples_leaf=1)).fit(x, y)
    assert tree.root_.is_leaf
    np.testing.assert_array_equal(tree.predict(x), y)


def test_threshold_split_recovers_a_step_function():
    x = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([1, 1, 1, 5, 5, 5])
    tree = DecisionTree(TreeConfig(max_depth=1, min_samples_split=2, min_samples_leaf=1, min_impurity_decrease=0.0)).fit(x, y)
    assert not tree.root_.is_leaf
    assert tree.root_.feature_index == 0
    assert 1.5 < tree.root_.threshold < 3.5
    np.testing.assert_array_equal(tree.predict(x), y)


def test_binary_condition_sends_low_values_left():
    x = np.array([[0.0], [1.0]])
    y = np.array([0, 5])
    tree = DecisionTree(TreeConfig(max_depth=1, min_samples_split=2, min_samples_leaf=1, min_impurity_decrease=0.0)).fit(x, y)
    assert tree.predict(np.array([[-1.0]]))[0] == 0
    assert tree.predict(np.array([[10.0]]))[0] == 5


def test_majority_vote_breaks_ties_toward_the_mean():
    votes = np.array(
        [
            [3, 5],
            [4, 1],
            [4, 1],
        ]
    )
    np.testing.assert_array_equal(majority_vote(votes), np.array([4, 1]))


def test_random_forest_averages_diverse_trees():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 3))
    y = (x[:, 0] > 0).astype(int) * 2 + (x[:, 1] > 0).astype(int)
    forest = RandomForest(
        ForestConfig(
            n_estimators=12,
            max_depth=3,
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=True,
            max_features="sqrt",
            random_state=0,
        )
    ).fit(x, y)
    predicted = forest.predict(x)
    assert (predicted == y).mean() > 0.7
    assert len(forest.trees_) == 12
