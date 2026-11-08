import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from movie_cf.aggregate import AGGREGATIONS
from movie_cf.evaluation import PerUserKFold, metrics, objective
from movie_cf.model import FEATURES, UserUserRecommender
from movie_cf.similarity import SIMILARITIES

K_VALUES = (1, 2, 5, 10, 20, 40)


def selection_score(truth, predicted) -> float:
    scores = metrics(truth, predicted)
    return objective(scores["accuracy"], scores["mae"])


def exact_accuracy(truth, predicted) -> float:
    return metrics(truth, predicted)["accuracy"]


def search(ratings: pd.DataFrame, n_folds: int = 5) -> GridSearchCV:
    """Choose similarity, aggregation, and k with scikit-learn's grid search."""
    grid = GridSearchCV(
        UserUserRecommender(),
        param_grid={"similarity": list(SIMILARITIES), "aggregation": list(AGGREGATIONS), "k": list(K_VALUES)},
        scoring={
            "score": make_scorer(selection_score),
            "accuracy": make_scorer(exact_accuracy),
            "neg_mae": "neg_mean_absolute_error",
        },
        refit="score",
        cv=PerUserKFold(n_folds),
    )
    return grid.fit(ratings[FEATURES], ratings["rating"])


def results_table(grid: GridSearchCV) -> pd.DataFrame:
    table = pd.DataFrame(grid.cv_results_)
    table = table[
        [
            "param_similarity",
            "param_aggregation",
            "param_k",
            "mean_test_accuracy",
            "std_test_accuracy",
            "mean_test_neg_mae",
            "mean_test_score",
        ]
    ].rename(
        columns={
            "param_similarity": "similarity",
            "param_aggregation": "aggregation",
            "param_k": "k",
            "mean_test_accuracy": "accuracy",
            "std_test_accuracy": "accuracy_std",
            "mean_test_neg_mae": "neg_mae",
            "mean_test_score": "score",
        }
    )
    table["mae"] = -table["neg_mae"]
    return table.sort_values(["score", "mae", "k"], ascending=[False, True, True], kind="mergesort", ignore_index=True)
