from pathlib import Path

import pandas as pd

from movie_knn import config

RATING_COLUMNS = ["id", "user_id", "movie_id", "rating"]


def load_movies(path: Path = config.MOVIES_PATH) -> pd.DataFrame:
    # movie.csv uses classic Mac (CR) line endings, which pandas does not split on.
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    rows = [line.split(";", 2) for line in text.splitlines() if line.strip()]
    movies = pd.DataFrame(rows, columns=["movie_id", "tmdb_id", "title"])
    return movies.astype({"movie_id": int, "tmdb_id": int}).set_index("movie_id")


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
