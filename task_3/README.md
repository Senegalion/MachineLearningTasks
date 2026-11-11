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
| `cosine` | cosine similarity on the shared movies |
| `mae` | `1 - mean(\|rating_a - rating_b\|) / 5` |
| `pearson` | Pearson correlation on the shared movies. Means are the averages on that shared set |
| `spearman` | Spearman rank correlation on the shared movies |

A negative similarity is dropped. The person's own ratings are not used as a neighbour.

### Prediction

Take the `k` people with the largest positive similarity who have rated the movie.

| Aggregation | Prediction |
|-------------|------------|
| `mean` | similarity-weighted average of those ratings |
| `vote` | weighted vote. The class with the largest total similarity wins. A tie goes to the class closest to this person's mean rating |

The result is clipped to 0–5 and rounded half up. If nobody eligible rated the movie, the prediction is this person's most common rating. A tie between several of their ratings uses the highest one.

`mean` with Pearson similarity is the same estimator as Surprise's `KNNBasic`.

### Choosing k, similarity, and aggregation

`movie_cf tune` runs `sklearn.model_selection.GridSearchCV` over `similarity ∈ {cosine, mae, pearson, spearman}`, `aggregation ∈ {mean, vote}`, and `k ∈ {1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 40}`. Folds are split inside each person. The chosen model is the one with the highest exact-match accuracy.

## Results

5-fold cross-validation on `train.csv` (mean over folds). `mae-vote` with `k = 2` won the grid search. The other rows are the two constant baselines, plus the other similarity and the other aggregation at that same `k`:

| Model | Description | Accuracy | Accuracy ±1 | MAE | RMSE |
|-------|-------------|----------|-------------|-----|------|
| `user-mode` | person's most common rating | 0.4113 | 0.7097 | 1.0526 | 1.5682 |
| `user-mean` | person's rounded mean | 0.3241 | 0.7533 | 0.9721 | 1.2933 |
| `pearson-vote` | Pearson similarity, vote of 2 neighbours | 0.4279 | 0.6941 | 1.0678 | 1.6044 |
| `mae-mean` | absolute-difference similarity, weighted average of 2 | 0.4812 | 0.8014 | 0.7902 | 1.2302 |
| `mae-vote` | absolute-difference similarity, vote of 2 neighbours | **0.5207** | 0.7884 | 0.7955 | 1.3056 |

`submission.csv` is produced with `mae-vote` and `k = 2`. The full grid is in `resources/cv_grid.csv`.

## Project layout

```
src/movie_cf/
  __main__.py       command line
  aggregate.py      weighted mean and weighted vote of the k neighbours
  config.py         paths, the rating scale, and the selected parameters
  data.py           CSV input and the submission
  evaluation.py     metrics and per-person folds
  model.py          the kNN recommender and the two baselines
  similarity.py     cosine, mean-absolute-error, Pearson, and Spearman similarity
  tuning.py         scikit-learn grid search
movie/              train.csv and task.csv
resources/cv_grid.csv
```
