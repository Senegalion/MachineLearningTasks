# Movie recommendations with Decision Trees and Random Forest

Predicts how each user rates a movie (0–5). A **separate** binary decision tree and a
**separate** random forest are trained for every person, using that person's known
ratings as labels and a few [TMDB](https://www.themoviedb.org/) attributes as features.

## Setup

```bash
python3 -m pip install -r requirements.txt
cp .env.example .env   # paste a TMDB API key (v3 key or v4 read access token)
```

The TMDB responses used to produce the submission are already cached in `resources/tmdb/`.

## Usage

```bash
make fetch        # download movie details from TMDB into resources/tmdb/
make evaluate     # 5-fold cross-validation on train.csv
make submission   # write submission_tree.csv, submission_forest.csv and tree.png
make tree         # write tree.png
make zip          # code.zip: source, data (movie/) and TMDB cache (resources/)
```

`PYTHONPATH=src python3 -m movie_dt {fetch,evaluate,predict,draw} --help`.
Input files live in `movie/` (`movie.csv`, `train.csv`, `task.csv`).

## Method

Each person has 90 rated movies. Those 90 rows train that person's classifier:

- **X:** TMDB description of the movie
- **y:** the rating this person gave it (0–5)

The fitted model then rates the movies that are missing in `task.csv`.

### Features

| Column | Encoding |
|--------|----------|
| genres | one 0/1 column per genre |
| directors | one 0/1 column per director |
| cast | one 0/1 column per person in the top 3 billed cast |
| year | release year |
| runtime | length in minutes |
| vote_average | TMDB public score |
| vote_count | number of TMDB votes |
| budget | production budget (USD) |
| revenue | box office revenue (USD) |
| revenue_budget_ratio | revenue / budget |

Missing numeric values are filled with the median over movies.

### Decision tree

A binary CART tree grown on those 90 movies:

* A node tests `x_j ≤ t` (for a 0/1 flag, `t = 0.5`).
* The split with the largest **Gini gain** on the six rating classes is kept.
* Growth stops at depth 2, at a pure node, when a node has fewer than 32 samples,
  when a child would have fewer than 16, or when the gain is tiny. With only 90 movies
  per person, deeper trees overfit: in CV (leaf ≥ 16) depth 2 gave 0.379 accuracy,
  depth 3 gave 0.377, and depth 4 / leaf 4 only 0.355.
* A leaf predicts the majority rating; ties go to the class closest to the leaf mean.

### Random forest

40 trees (depth ≤ 4, leaf ≥ 4 — averaging compensates for the extra variance). Each tree sees a bootstrap sample of the 90 movies and, at every split,
only `√p` random features. The forest prediction is a majority vote.

### Validation

5-fold cross-validation, folds split **within each user**. A prediction is exact when it matches the rating, and counts for **accuracy ±1** when it is at most one point away.

## Results

5-fold cross-validation on `train.csv` (mean over folds):

| Model       | Description                 | Accuracy | Accuracy +-1 | MAE    | RMSE   |
|-------------|-----------------------------|----------|--------------|--------|--------|
| `user-mode` | person's most common rating | 0.4113   | 0.7097       | 1.0526 | 1.5682 |
| `user-mean` | person's rounded mean       | 0.3241   | 0.7533       | 0.9721 | 1.2933 |
| `tree`      | per-user decision tree      | 0.3790   | 0.6968       | 1.0944 | 1.5813 |
| `forest`    | per-user random forest      | 0.4145   | 0.7168       | 1.0333 | 1.5437 |

`submission_tree.csv` / `submission_forest.csv` come from `tree` / `forest`.
`tree.png` shows one person's fitted tree.

## Project layout

```
src/movie_dt/
  config.py         paths
  data.py           CSV input/output
  tmdb.py           TMDB cache
  features.py       TMDB → numeric table
  tree.py           binary decision tree and random forest
  model.py          one classifier per user, baselines
  visualize.py      tree.png
  evaluation.py     metrics and per-user CV
  __main__.py       CLI
```
