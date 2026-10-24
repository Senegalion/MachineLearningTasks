# Movie recommendations with per-user kNN

Predicts how each user rates a movie (0–5) using a separate kNN model for every user. A movie is described by
features downloaded from [TMDB](https://www.themoviedb.org/) and by the pattern of ratings it received from other
users in `train.csv`.

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
make evaluate     # 5-fold cross-validation of all model variants on train.csv
make submission   # fit on train.csv, write submission.csv and resources/user_params.csv
make test         # unit tests
make zip          # build code.zip
```

The same commands are available directly: `PYTHONPATH=src python3 -m movie_knn {fetch,evaluate,predict} --help`.
The input files are expected in `movie/` (`movie.csv`, `train.csv`, `task.csv`).

## Method

### Movie representation and similarity

| Type        | Features                                                                                  | Similarity                                         |
|-------------|-------------------------------------------------------------------------------------------|----------------------------------------------------|
| set         | genres, keywords, top-8 cast, directors, writers, composers, production companies, countries | Jaccard index                                      |
| categorical | original language, collection                                                             | equality                                           |
| numeric     | release year, runtime, vote average, vote count (log), budget (log)                       | `1 - |a - b| / spread`, spread = 5–95 percentile range |
| text        | tagline + overview                                                                        | cosine of TF-IDF vectors                           |
| ratings     | ratings of the movie by other users (centered by user mean)                               | adjusted cosine                                    |

Each per-feature similarity matrix is standardized (z-score over off-diagonal entries), so the features have a
comparable scale, and the movie similarity is their weighted average.

The rating-pattern similarity for a user is always computed **without that user's ratings**. When predicting, the
user's rating of the target movie is unknown; excluding the user's column makes leave-one-out validation reproduce
that situation instead of leaking the held-out rating into the similarity.

### Prediction

For a user, the `k` training movies most similar to the target movie are taken and their ratings are aggregated
with a vote, a weighted median or a rounded mean, each with uniform weights or weights `exp(s - s_max)`.

### Model selection

* Feature weights shared by all users are learned by coordinate ascent over the grid `{0, 0.5, 1, 2, 4}`.
* For every user, `k ∈ [1, 60]` and the aggregation are selected by leave-one-out validation on that user's
  ratings. The objective is accuracy with a small MAE penalty as a tie-breaker; scores are smoothed over
  neighbouring `k` so that an isolated lucky `k` is not chosen.
* Optionally, feature weights can be tuned per user as well (`knn-per-user-weights`), accepting a change only
  when it improves the user's validation score by a margin.

All variants are compared with an outer 5-fold cross-validation split within each user, so the hyperparameter
search never sees the evaluation fold.

## Results

5-fold cross-validation on `train.csv` (folds split within each user, mean over folds):

| Model                  | Description                                              | Accuracy | MAE    | RMSE   |
|------------------------|----------------------------------------------------------|----------|--------|--------|
| `user-mode`            | baseline: user's most frequent rating                    | 0.4113   | 1.0526 | 1.5682 |
| `user-mean`            | baseline: user's rounded mean rating                     | 0.3241   | 0.9721 | 1.2933 |
| `knn-tmdb-only`        | TMDB features only, learned weights, per-user k          | 0.4059   | 1.0282 | 1.5212 |
| `knn-equal-weights`    | all features with equal weights, per-user k              | 0.4096   | 1.0074 | 1.4956 |
| `knn-shared-k`         | learned weights, one k and aggregation for all users     | 0.4263   | 0.9757 | 1.4693 |
| `knn-global-weights`   | learned weights, per-user k and aggregation              | **0.4296** | **0.9251** | **1.3945** |
| `knn-per-user-weights` | per-user weights and per-user k                          | 0.4224   | 0.9479 | 1.4184 |

`submission.csv` is produced with `knn-global-weights`. Observations:

* Content features from TMDB alone do not beat the per-user mode baseline: taste of a single user is hard to
  explain with genre, cast or year. Each of them alone scores about 0.38–0.40 in leave-one-out validation.
* The rating-pattern similarity is the most informative component, and the weight search assigns it the highest
  weight; of the TMDB features, vote count and the overview text are kept.
* Choosing `k` and the aggregation per user helps, while also tuning feature weights per user overfits the 90
  ratings available for each person.

The selected per-user `k`, aggregation and feature weights are saved in `resources/user_params.csv`.

## Project layout

```
src/movie_knn/
  config.py         paths and constants
  data.py           reading the input CSV files and writing the submission
  tmdb.py           TMDB API client with a local JSON cache
  features.py       feature extraction from TMDB data, per-feature similarity tensor
  similarity.py     similarity functions for set, categorical, numeric and text features
  collaborative.py  rating-pattern similarity computed from the training ratings
  knn.py            vectorized kNN prediction for all k and aggregation methods at once
  model.py          per-user kNN recommender with hyperparameter selection, baselines
  evaluation.py     metrics and per-user cross-validation
  __main__.py       command line interface
tests/              unit tests
resources/tmdb/     cached TMDB responses
```
