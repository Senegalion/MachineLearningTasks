# Movie recommendations from person similarity

Predicts how each person rates a movie (0–5) with user-based k-nearest neighbours. Two people are similar when their ratings agree on the movies both of them have seen. The missing rating is then taken from the nearest of those people who have seen the movie. Movie descriptions are not used.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Usage

```bash
make evaluate    # 5-fold cross-validation on train.csv
make tune        # grid-search similarity, aggregation, and k
make submission  # fit on train.csv and write submission.csv
make zip         # build code.zip
```

`PYTHONPATH=src python3 -m movie_cf {evaluate,tune,predict} --help`.
Input files live in `movie/` (`train.csv`, `task.csv`).

## Method

Each person has 90 ratings in `train.csv` and 30 missing ratings in `task.csv`. This is the memory-based user–user approach: the rating matrix itself is the model, and a prediction looks up similar people at query time.

### Similarity

Two people are compared only on the movies both rated.

| Name | Definition |
|------|------------|
| `mse` | `1 - mean((rating_a - rating_b)²) / 25` |
| `mae` | `1 - mean(\|rating_a - rating_b\|) / 5` |
| `pearson` | Pearson correlation on the shared movies |
| `spearman` | Spearman rank correlation on the shared movies |
| `kendall` | Kendall rank correlation on the shared movies |

A negative similarity is dropped. The person's own ratings are not used as a neighbour.

### Prediction

Take the `k` people with the largest positive similarity who have rated the movie.

| Aggregation | Prediction |
|-------------|------------|
| `mean` | unweighted average of those ratings |
| `mean_weighted` | similarity-weighted average of those ratings |
| `vote` | unweighted vote. The rating that appears most often wins |
| `vote_weighted` | weighted vote. The rating with the largest total similarity wins |

A vote tie goes to the rating closest to this person's most common rating. Equal distance keeps the lower rating. The result is clipped to 0–5 and rounded half up. If nobody eligible rated the movie, the prediction is this person's most common rating. A tie between several of their ratings uses the highest one.

`mean_weighted` with Pearson similarity is the same estimator as Surprise's `KNNBasic`. At `k = 1` every aggregation returns that single neighbour, so the four rows are identical.

### Choosing k, similarity, and aggregation

`movie_cf tune` runs `sklearn.model_selection.GridSearchCV` over every similarity, every aggregation, and a list of `k`. Folds are split inside each person. The chosen model is the one with the highest exact-match accuracy. The run in `resources/cv_grid.csv` used `similarity ∈ {mse, mae, pearson, spearman, kendall}`, `aggregation ∈ {mean, mean_weighted, vote, vote_weighted}`, and `k ∈ {1, 2, 3, 4, 5, 8, 10, 15, 20}` (220 configurations).

## Results

5-fold cross-validation on `train.csv` (mean over folds). `mae` with `vote_weighted` and `k = 2` won the grid search. The next `mae` row is `k = 1`, where aggregation does not matter. The other rows are the remaining `mae` aggregations at `k = 2`, then the best `k = 2` configuration of each other similarity:

| Similarity | Aggregation | k | Accuracy | Accuracy std | MAE |
|------------|-------------|---|----------|--------------|-----|
| `mae` | `vote_weighted` | 2 | **0.5212** | 0.0063 | 0.7962 |
| `mae` | any | 1 | 0.5204 | 0.0070 | 0.7990 |
| `mae` | `vote` | 2 | 0.5122 | 0.0039 | 0.7834 |
| `mae` | `mean_weighted` | 2 | 0.4812 | 0.0057 | 0.7902 |
| `mae` | `mean` | 2 | 0.4592 | 0.0039 | 0.8239 |
| `mse` | `vote_weighted` | 2 | 0.4940 | 0.0066 | 0.8018 |
| `kendall` | `vote` | 2 | 0.4453 | 0.0034 | 0.9288 |
| `pearson` | `vote` | 2 | 0.4449 | 0.0023 | 0.9279 |
| `spearman` | `vote` | 2 | 0.4422 | 0.0024 | 0.9373 |

Accuracy falls as `k` grows: the same `mae` / `vote_weighted` model is 0.5034 at `k = 3` and 0.3551 at `k = 40`. Correlation similarities top out at 0.445. The grid search selects `mae`, `vote_weighted`, and `k = 2`. `config.py` still builds `submission.csv` with `mae`, `vote`, and `k = 2` (accuracy 0.5122). The full grid is in `resources/cv_grid.csv`.

## Project layout

```
src/movie_cf/
  __main__.py       command line
  aggregate.py      mean, weighted mean, vote, and weighted vote of the k neighbours
  config.py         paths, the rating scale, and the selected parameters
  data.py           CSV input and the submission
  evaluation.py     metrics and per-person folds
  model.py          the kNN recommender and the two baselines
  similarity.py     mean-squared-error, mean-absolute-error, Pearson, Spearman, and Kendall similarity
  tuning.py         scikit-learn grid search
movie/              train.csv and task.csv
resources/cv_grid.csv
```
