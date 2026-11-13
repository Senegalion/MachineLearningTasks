from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "movie"
TRAIN_PATH = DATA_DIR / "train.csv"
TASK_PATH = DATA_DIR / "task.csv"

SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"
GRID_RESULTS_PATH = PROJECT_ROOT / "resources" / "cv_grid.csv"

RATINGS = (0, 1, 2, 3, 4, 5)
RATING_MIN = RATINGS[0]
RATING_MAX = RATINGS[-1]
RANDOM_STATE = 42

# Replaced by the winner of `movie_cf tune` (scikit-learn grid search).
SELECTED_SIMILARITY = "mae"
SELECTED_AGGREGATION = "vote_weighted"
SELECTED_K = 2
