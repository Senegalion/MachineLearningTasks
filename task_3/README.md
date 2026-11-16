# Movie recommendations from person similarity

Predicts how each person rates a movie (0–5) with user-based k-nearest neighbours. Two people are similar when their ratings agree on the movies both of them have seen. The missing rating is then taken from the nearest of those people who have seen the movie. Movie descriptions are not used.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Usage

```bash
make evaluate    # 5-fold cross-validation on train.csv
make tune        # grid-search similarity, aggregation, k, and power
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
| `exact` | share of the shared movies on which both gave exactly the same rating |
| `mae` | `1 - mean(\|rating_a - rating_b\|) / 5` |
| `mse` | `1 - mean((rating_a - rating_b)²) / 25` |
| `pearson` | Pearson correlation on the shared movies |

In every measure, people are more similar the more alike their ratings of the shared movies are.

A negative similarity is dropped. The person's own ratings are not used as a neighbour.

### Prediction

Take the `k` people with the largest positive similarity who have rated the movie. A person who has not rated the movie is never used for it. Each chosen neighbour has the weight `similarity ** power`: a higher power lets the closest people decide, while the more distant ones only break ties. The power does not change who the nearest neighbours are, and the unweighted `mean` and `vote` ignore it.

| Aggregation | Prediction |
|-------------|------------|
| `mean` | unweighted average of those ratings |
| `mean_weighted` | similarity-weighted average of those ratings |
| `vote` | unweighted vote. The rating that appears most often wins |
| `vote_weighted` | weighted vote. The rating with the largest total similarity wins |

A vote tie goes to the rating closest to this person's most common rating. Equal distance keeps the lower rating. The result is clipped to 0–5 and rounded half up. If nobody eligible rated the movie, the prediction is this person's most common rating. A tie between several of their ratings uses the highest one.

At `k = 1` every aggregation returns that single neighbour, so the four aggregations give identical results.

### Choosing similarity, aggregation, k, and power

`movie_cf tune` runs `sklearn.model_selection.GridSearchCV` with 5-fold cross-validation on `train.csv` over `similarity ∈ {exact, mae, mse, pearson}`, `aggregation ∈ {mean, mean_weighted, vote, vote_weighted}`, `k ∈ {1, 2, 3, 5, 10, 20, 40, 80}`, and `power ∈ {1, 2, 4, 8}` (512 configurations). Folds are split inside each person, so every person keeps ratings in every training split. The chosen model is the one with the highest exact-match accuracy on the held-out folds.

## Results

5-fold cross-validation on `train.csv` (mean over folds). `exact` with `vote_weighted`, `k = 40`, and `power = 4` won the grid search, and `submission.csv` is built with exactly this configuration (`config.py`). The first rows are the best configuration of each similarity, then the two constant baselines (`python -m movie_cf evaluate`):

| Similarity | Aggregation | k | power | Accuracy | Accuracy std | MAE |
|------------|-------------|---|-------|----------|--------------|-----|
| `exact` | `vote_weighted` | 40 | 4 | **0.5511** | 0.0036 | 0.7653 |
| `mae` | `vote_weighted` | 2 | 1 | 0.5212 | 0.0063 | 0.7962 |
| `mse` | `vote_weighted` | 2 | 1 | 0.4940 | 0.0066 | 0.8018 |
| `pearson` | `vote` | 2 | 1 | 0.4449 | 0.0023 | 0.9279 |
| baseline: person's most common rating | | | | 0.4113 | 0.0040 | 1.0526 |
| baseline: person's rounded mean | | | | 0.3241 | 0.0070 | 0.9721 |

Why `exact` and a high power work here: many people in `train.csv` have a near copy. 105 of the 358 people have someone who gave the same rating on every movie they share, and every pair shares at least 22 movies. The best predictor is therefore what the people who rate exactly like this person gave. With `power = 1` the many more distant neighbours outvote those near copies (`exact`, `vote_weighted`, `k = 40`: 0.4749), `power = 2` gives 0.5316, `power = 4` gives 0.5511, and `power = 8` gives 0.5432. With `power = 4`, accuracy is 0.5467 at `k = 10`, 0.5502 at `k = 20`, 0.5511 at `k = 40`, and 0.5507 at `k = 80`, so the choice is not a lucky spike. The full grid is in `resources/cv_grid.csv`.

## Project layout

```
src/movie_cf/
  __main__.py       command line
  aggregate.py      mean, weighted mean, vote, and weighted vote of the k neighbours
  config.py         paths, the rating scale, and the selected parameters
  data.py           CSV input and the submission
  evaluation.py     metrics and per-person folds
  model.py          the kNN recommender and the two baselines
  similarity.py     exact-agreement, mean-absolute-error, mean-squared-error, and Pearson similarity
  tuning.py         scikit-learn grid search
movie/              train.csv and task.csv
resources/cv_grid.csv
```
