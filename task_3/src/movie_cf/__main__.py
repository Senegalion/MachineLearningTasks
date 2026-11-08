import argparse
import logging
from pathlib import Path

import pandas as pd

from movie_cf import config, data
from movie_cf.evaluation import cross_validate
from movie_cf.model import FEATURES, UserMeanBaseline, UserModeBaseline, UserUserRecommender
from movie_cf.tuning import results_table, search

logger = logging.getLogger("movie_cf")

BASELINES = {
    "user-mode": UserModeBaseline,
    "user-mean": UserMeanBaseline,
}
SELECTED_NAME = f"{config.SELECTED_SIMILARITY}-{config.SELECTED_AGGREGATION}"
SELECTED = {
    "similarity": config.SELECTED_SIMILARITY,
    "aggregation": config.SELECTED_AGGREGATION,
    "k": config.SELECTED_K,
}


def evaluate(args: argparse.Namespace) -> None:
    train = data.load_train()
    factories = {
        **BASELINES,
        SELECTED_NAME: lambda: UserUserRecommender(**SELECTED),
    }
    names = args.only or list(factories)
    summary = {}
    for name in names:
        logger.info("Evaluating %s", name)
        scores = cross_validate(factories[name], train, n_folds=args.folds)
        summary[name] = scores.mean().to_dict() | {"accuracy_std": float(scores["accuracy"].std())}
    report = pd.DataFrame(summary).T[["accuracy", "accuracy_std", "accuracy_pm1", "mae", "rmse"]]
    print(report.round(4).to_string())


def tune(args: argparse.Namespace) -> None:
    grid = search(data.load_train(), n_folds=args.folds)
    table = results_table(grid)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(table.round(4).to_string(index=False))
    best = grid.best_params_
    logger.info(
        "Best configuration: similarity=%s aggregation=%s k=%s", best["similarity"], best["aggregation"], best["k"]
    )
    logger.info("Saved %d configurations to %s", len(table), args.output)


def predict(args: argparse.Namespace) -> None:
    train = data.load_train()
    model = UserUserRecommender(**SELECTED).fit(train[FEATURES], train["rating"])
    task = data.load_task()
    predictions = pd.Series(model.predict(task[FEATURES]), index=task.index)
    data.save_submission(task, predictions, args.output)
    logger.info(
        "Saved %d predictions to %s (%s/%s/k=%s)",
        len(task),
        args.output,
        SELECTED["similarity"],
        SELECTED["aggregation"],
        SELECTED["k"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="movie_cf", description="User-based kNN movie rating prediction.")
    commands = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = commands.add_parser("evaluate", help="compare models with cross-validation on train.csv")
    evaluate_parser.add_argument("--folds", type=int, default=5)
    evaluate_parser.add_argument("--only", nargs="+", choices=[*BASELINES, SELECTED_NAME], help="models to evaluate")
    evaluate_parser.set_defaults(handler=evaluate)

    tune_parser = commands.add_parser("tune", help="grid-search similarity, aggregation, and k")
    tune_parser.add_argument("--folds", type=int, default=5)
    tune_parser.add_argument("--output", type=Path, default=config.GRID_RESULTS_PATH)
    tune_parser.set_defaults(handler=tune)

    predict_parser = commands.add_parser("predict", help="fit on train.csv and fill in task.csv")
    predict_parser.add_argument("--output", type=Path, default=config.SUBMISSION_PATH)
    predict_parser.set_defaults(handler=predict)

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
