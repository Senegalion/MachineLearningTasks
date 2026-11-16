import pandas as pd
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import GridSearchCV

from movie_cf.aggregate import AGGREGATIONS
from movie_cf.evaluation import cross_validator
from movie_cf.model import FEATURES, UserUserRecommender
from movie_cf.similarity import SIMILARITIES

K_VALUES = (1, 2, 3, 5, 10, 20, 40, 80)
POWERS = (1.0, 2.0, 4.0, 8.0)


def search(ratings: pd.DataFrame, n_folds: int = 5) -> GridSearchCV:
    """Choose similarity, aggregation, k, and the weight power by exact-match accuracy."""
    grid = GridSearchCV(
        UserUserRecommender(),
        param_grid={
            "similarity": list(SIMILARITIES),
            "aggregation": list(AGGREGATIONS),
            "k": list(K_VALUES),
            "power": list(POWERS),
        },
        scoring={"accuracy": "accuracy", "mae": make_scorer(mean_absolute_error)},
        refit="accuracy",
        cv=cross_validator(ratings["user_id"], n_folds),
        n_jobs=-1,
    )
    return grid.fit(ratings[FEATURES], ratings["rating"])


def results_table(grid: GridSearchCV) -> pd.DataFrame:
    rows = pd.DataFrame(grid.cv_results_)
    table = pd.DataFrame(
        {
            "similarity": rows["param_similarity"],
            "aggregation": rows["param_aggregation"],
            "k": rows["param_k"],
            "power": rows["param_power"],
            "accuracy": rows["mean_test_accuracy"],
            "accuracy_std": rows["std_test_accuracy"],
            "mae": rows["mean_test_mae"],
        }
    )
    return table.sort_values(
        ["accuracy", "mae", "k", "power"], ascending=[False, True, True, True], kind="mergesort", ignore_index=True
    )
