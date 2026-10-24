import argparse
import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd

from movie_knn import config, data, tmdb
from movie_knn.evaluation import cross_validate
from movie_knn.features import SimilarityTensor, extract_features
from movie_knn.model import KNNRecommender, ModelConfig, UserMeanBaseline, UserModeBaseline

logger = logging.getLogger("movie_knn")

DEFAULT_MODEL = ModelConfig()

BASELINES = {
    "user-mode": UserModeBaseline,
    "user-mean": UserMeanBaseline,
}

EXPERIMENTS = {
    "knn-tmdb-only": replace(DEFAULT_MODEL, use_rating_pattern=False),
    "knn-equal-weights": replace(DEFAULT_MODEL, learn_global_weights=False),
    "knn-shared-k": replace(DEFAULT_MODEL, per_user_k=False),
    "knn-global-weights": DEFAULT_MODEL,
    "knn-per-user-weights": replace(DEFAULT_MODEL, per_user_weights=True),
}


def build_tensor() -> SimilarityTensor:
    movies = data.load_movies()
    details = tmdb.load_cached(movies["tmdb_id"])
    return SimilarityTensor.from_features(extract_features(movies, details))


def fetch(args: argparse.Namespace) -> None:
    tmdb.fetch_movies(data.load_movies()["tmdb_id"], force=args.force)


def evaluate(args: argparse.Namespace) -> None:
    tensor = build_tensor()
    train = data.load_train()
    factories = {
        **BASELINES,
        **{name: (lambda cfg=cfg: KNNRecommender(tensor, cfg)) for name, cfg in EXPERIMENTS.items()},
    }
    names = args.only or list(factories)
    summary = {}
    for name in names:
        logger.info("Evaluating %s", name)
        scores = cross_validate(factories[name], train, n_folds=args.folds)
        summary[name] = scores.mean().to_dict() | {"accuracy_std": scores["accuracy"].std()}
    report = pd.DataFrame(summary).T[["accuracy", "accuracy_std", "mae", "rmse"]]
    print(report.round(4).to_string())


def predict(args: argparse.Namespace) -> None:
    tensor = build_tensor()
    model = KNNRecommender(tensor, EXPERIMENTS[args.model]).fit(data.load_train())
    task = data.load_task()
    data.save_submission(task, model.predict(task), args.output)
    logger.info("Saved %d predictions to %s", len(task), args.output)
    if args.params:
        model.describe().to_csv(args.params)
        logger.info("Saved per-user hyperparameters to %s", args.params)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="movie_knn", description="Per-user kNN movie rating prediction.")
    commands = parser.add_subparsers(dest="command", required=True)

    fetch_parser = commands.add_parser("fetch", help="download movie details from TMDB into the local cache")
    fetch_parser.add_argument("--force", action="store_true", help="re-download already cached movies")
    fetch_parser.set_defaults(handler=fetch)

    evaluate_parser = commands.add_parser("evaluate", help="compare models with cross-validation on train.csv")
    evaluate_parser.add_argument("--folds", type=int, default=5)
    evaluate_parser.add_argument("--only", nargs="+", choices=[*BASELINES, *EXPERIMENTS], help="models to evaluate")
    evaluate_parser.set_defaults(handler=evaluate)

    predict_parser = commands.add_parser("predict", help="fit on train.csv and fill in task.csv")
    predict_parser.add_argument("--model", choices=list(EXPERIMENTS), default="knn-global-weights")
    predict_parser.add_argument("--output", type=Path, default=config.SUBMISSION_PATH)
    predict_parser.add_argument("--params", type=Path, help="optional CSV with the selected per-user hyperparameters")
    predict_parser.set_defaults(handler=predict)

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
