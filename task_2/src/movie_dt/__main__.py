import argparse
import logging
from pathlib import Path

import pandas as pd

from movie_dt import config, data, tmdb
from movie_dt.evaluation import cross_validate
from movie_dt.features import TabularFeatures, extract_features
from movie_dt.model import ModelConfig, PerUserClassifier, UserMeanBaseline, UserModeBaseline
from movie_dt.visualize import draw_tree

logger = logging.getLogger("movie_dt")

CONTENT: TabularFeatures | None = None

BASELINES = {
    "user-mode": UserModeBaseline,
    "user-mean": UserMeanBaseline,
}


def load_content() -> TabularFeatures:
    global CONTENT
    if CONTENT is None:
        movies = data.load_movies()
        CONTENT = TabularFeatures.from_raw(extract_features(movies, tmdb.load_cached(movies["tmdb_id"])))
        logger.info("TMDB features: %d movies × %d columns", len(CONTENT.movie_ids), len(CONTENT.feature_names))
    return CONTENT


def experiment_factories() -> dict:
    content = load_content()
    return {
        "tree": lambda: PerUserClassifier(content, ModelConfig(kind="tree")),
        "forest": lambda: PerUserClassifier(content, ModelConfig(kind="forest")),
    }


def fetch(args: argparse.Namespace) -> None:
    tmdb.fetch_movies(data.load_movies()["tmdb_id"], force=args.force)


def evaluate(args: argparse.Namespace) -> None:
    train = data.load_train()
    factories = {**BASELINES, **experiment_factories()}
    names = args.only or list(factories)
    summary = {}
    for name in names:
        logger.info("Evaluating %s", name)
        scores = cross_validate(factories[name], train, n_folds=args.folds)
        summary[name] = scores.mean().to_dict() | {"accuracy_std": scores["accuracy"].std()}
    report = pd.DataFrame(summary).T[["accuracy", "accuracy_std", "accuracy_pm1", "mae", "rmse"]]
    print(report.round(4).to_string())


def _fit(kind: str) -> PerUserClassifier:
    return experiment_factories()[kind]().fit(data.load_train())


def _choose_user(model: PerUserClassifier) -> int:
    def score(user_id: int) -> tuple[int, int, int]:
        tree = model.models_[user_id]
        root = tree.root_
        return (int(not root.is_leaf), int((root.class_counts > 0).sum()), tree.depth())

    return max(model.models_, key=score)


def _save_tree_image(model: PerUserClassifier, user_id: int | None, path: Path) -> None:
    if not model.models_:
        raise RuntimeError("No fitted trees to draw.")
    if user_id is None:
        user_id = _choose_user(model)
    root = model.models_[user_id].root_
    draw_tree(root, model.content.feature_names, path, title=f"Decision tree for user {user_id}")
    logger.info("Saved decision tree for user %s to %s", user_id, path)


def predict(args: argparse.Namespace) -> None:
    task = data.load_task()
    tree_model = _fit("tree")
    forest_model = _fit("forest")
    data.save_submission(task, tree_model.predict(task), args.tree_output)
    data.save_submission(task, forest_model.predict(task), args.forest_output)
    logger.info("Saved tree predictions to %s", args.tree_output)
    logger.info("Saved forest predictions to %s", args.forest_output)
    _save_tree_image(tree_model, args.tree_user, args.tree_image)


def draw(args: argparse.Namespace) -> None:
    _save_tree_image(_fit("tree"), args.tree_user, args.tree_image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="movie_dt", description="Per-user decision tree and random forest rating prediction.")
    commands = parser.add_subparsers(dest="command", required=True)

    fetch_parser = commands.add_parser("fetch", help="download movie details from TMDB into the local cache")
    fetch_parser.add_argument("--force", action="store_true", help="re-download already cached movies")
    fetch_parser.set_defaults(handler=fetch)

    evaluate_parser = commands.add_parser("evaluate", help="compare models with cross-validation on train.csv")
    evaluate_parser.add_argument("--folds", type=int, default=5)
    evaluate_parser.add_argument("--only", nargs="+", choices=[*BASELINES, "tree", "forest"])
    evaluate_parser.set_defaults(handler=evaluate)

    predict_parser = commands.add_parser("predict", help="fit on train.csv and write both submission files plus tree.png")
    predict_parser.add_argument("--tree-output", type=Path, default=config.SUBMISSION_TREE_PATH)
    predict_parser.add_argument("--forest-output", type=Path, default=config.SUBMISSION_FOREST_PATH)
    predict_parser.add_argument("--tree-image", type=Path, default=config.TREE_IMAGE_PATH)
    predict_parser.add_argument("--tree-user", type=int, help="user id whose fitted tree is drawn in tree.png")
    predict_parser.set_defaults(handler=predict)

    draw_parser = commands.add_parser("draw", help="fit the per-user trees and write tree.png")
    draw_parser.add_argument("--tree-image", type=Path, default=config.TREE_IMAGE_PATH)
    draw_parser.add_argument("--tree-user", type=int, help="user id to draw")
    draw_parser.set_defaults(handler=draw)

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
