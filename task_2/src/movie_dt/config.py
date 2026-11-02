from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "movie"
MOVIES_PATH = DATA_DIR / "movie.csv"
TRAIN_PATH = DATA_DIR / "train.csv"
TASK_PATH = DATA_DIR / "task.csv"

TMDB_CACHE_DIR = PROJECT_ROOT / "resources" / "tmdb"
SUBMISSION_TREE_PATH = PROJECT_ROOT / "submission_tree.csv"
SUBMISSION_FOREST_PATH = PROJECT_ROOT / "submission_forest.csv"
TREE_IMAGE_PATH = PROJECT_ROOT / "tree.png"

RATINGS = (0, 1, 2, 3, 4, 5)
N_CLASSES = len(RATINGS)
RANDOM_STATE = 42
