# Movie recommendations with Decision Trees and Random Forest

Predicts how each user rates a movie (0–5) with a **separate classifier per person**.
Two models are trained: a single binary decision tree and a random forest. A movie is
described by features downloaded from [TMDB](https://www.themoviedb.org/) and by the
ratings it received from other users in `train.csv`.

## Setup

```bash
python3 -m pip install -r requirements.txt
cp .env.example .env   # paste a TMDB API key (v3 key or v4 read access token)
```

The TMDB responses used to produce the submission are already cached in `resources/tmdb/`, so the key is only
needed to download them again.

## Usage

```bash
make fetch        # download movie details from TMDB into resources/tmdb/
make evaluate     # 5-fold cross-validation of trees, forests and baselines on train.csv
make submission   # fit on train.csv, write submission_tree.csv, submission_forest.csv and tree.png
make tree         # fit trees only and write tree.png
make test         # unit tests
make zip          # build code.zip
```

The same commands are available directly: `PYTHONPATH=src python3 -m movie_dt {fetch,evaluate,predict,draw} --help`.
Pick a person with `make tree` or `PYTHONPATH=src python3 -m movie_dt draw --tree-user 19`.
The input files are expected in `movie/` (`movie.csv`, `train.csv`, `task.csv`).

## Method

Each person has 90 training ratings. A classifier is induced on that person's labelled movies
only, then applied to the movies whose ratings are missing in `task.csv`.

### Features

The TMDB attributes collected for each movie are encoded as columns a tree can split on:

| Type        | Source                                                                                  | Encoding                                      |
|-------------|-----------------------------------------------------------------------------------------|-----------------------------------------------|
| set         | genres, keywords, top-8 cast, directors, writers, composers, companies, countries       | binary indicators (min document frequency 3)  |
| categorical | original language, collection                                                           | binary indicators                             |
| numeric     | year, runtime, vote average, vote count (log), budget (log)                             | raw value, missing filled with the median     |
| text        | tagline + overview                                                                      | top TF-IDF terms                              |
| ratings     | ratings of the movie by every other user                                                | numeric; missing filled with that user's mean |

The target user's own ratings are never used as features, so a split cannot leak the label.

### Decision tree

The implementation is a binary CART classifier:

* **Condition set.** Every node tests `x_j ≤ t`. Binary indicators use a threshold of 0.5
  (feature present or not); numeric and collaborative columns use the midpoint between two
  neighbouring observed values.
* **Split selection.** Among those conditions, the split with the largest **Gini gain** on
  the six rating classes is kept.
* **Stopping.** Induction stops at `max_depth = 5`, when a node has fewer than 8 samples,
  when a child would have fewer than 3 samples, when the node is pure, or when the Gini
  gain is below `10⁻⁴`.
* **Leaf prediction.** Majority class; ties go to the rating closest to the node mean.

### Random forest

Diversity of the ensemble comes from two independent randomizations:

* **Bootstrap samples** of the person's 90 movies (bagging).
* A **random subset of features** at every split (`√p` columns).

Forty trees are grown (`max_depth = 6`). The forest prediction is a **majority vote**; ties
again go to the rating closest to the mean of the votes.

### Validation

Generalization is measured with 5-fold cross-validation **split within each user**, so every
person keeps the same share of ratings in every training fold. Collaborative features are
rebuilt from the training part of the fold only.

## Results

5-fold cross-validation on `train.csv` (folds split within each user, mean over folds):

| Model              | Description                                              | Accuracy | MAE    | RMSE   |
|--------------------|----------------------------------------------------------|----------|--------|--------|
| `user-mode`        | baseline: user's most frequent rating                    | 0.4113   | 1.0526 | 1.5682 |
| `user-mean`        | baseline: user's rounded mean rating                     | 0.3241   | 0.9721 | 1.2933 |
| `tree-tmdb-only`   | single tree on TMDB columns only                         | 0.3558   | 1.1506 | 1.6306 |
| `forest-tmdb-only` | random forest on TMDB columns only                       | 0.4128   | 1.0378 | 1.5464 |
| `tree`             | single tree, TMDB + other users' ratings                 | 0.4198   | 0.9951 | 1.4873 |
| `forest`           | random forest, TMDB + other users' ratings               | **0.4387** | **0.9564** | **1.4583** |

`submission_tree.csv` is produced with `tree`, `submission_forest.csv` with `forest`.
`tree.png` is the fitted tree of user 19 (an arbitrary person whose ratings use several classes).

Observations:

* TMDB columns alone are weak for a single tree (they overfit the 90 labels) and only match
  the mode baseline once bagged in a forest.
* Other users' ratings are the features the trees actually split on; the forest then
  improves on a single tree by averaging out the unstable splits.
* The forest is the most accurate of the two required models and also has the lowest MAE.

## Project layout

```
src/movie_dt/
  config.py         paths and constants
  data.py           reading the input CSV files and writing submissions
  tmdb.py           TMDB API client with a local JSON cache
  features.py       TMDB extraction and tabular encoding
  collaborative.py  other users' ratings as extra columns
  tree.py           binary decision tree and random forest
  model.py          per-user classifiers and baselines
  visualize.py      drawing a fitted tree to PNG
  evaluation.py     metrics and per-user cross-validation
  __main__.py       command line interface
tests/              unit tests
resources/tmdb/     cached TMDB responses
```
