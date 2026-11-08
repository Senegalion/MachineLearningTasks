from pathlib import Path

import pandas as pd

from movie_cf import config

RATING_COLUMNS = ["id", "user_id", "movie_id", "rating"]


def load_ratings(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=";",
        header=None,
        names=RATING_COLUMNS,
        na_values=["NULL"],
        dtype={"id": int, "user_id": int, "movie_id": int, "rating": "Int64"},
    )


def load_train(path: Path = config.TRAIN_PATH) -> pd.DataFrame:
    return load_ratings(path).astype({"rating": int})


def load_task(path: Path = config.TASK_PATH) -> pd.DataFrame:
    return load_ratings(path)


def save_submission(task: pd.DataFrame, predictions: pd.Series, path: Path) -> None:
    submission = task.copy()
    submission["rating"] = predictions.reindex(submission.index).astype(int)
    submission[RATING_COLUMNS].to_csv(path, sep=";", header=False, index=False)
