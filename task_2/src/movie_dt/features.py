from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

TOP_CAST = 8


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    extract: Callable[[dict], object]


def _names(items: Iterable[dict], key: str = "name") -> frozenset[str]:
    return frozenset(item[key] for item in items)


def _genre_names(details: dict) -> frozenset[str]:
    return _names(details["genres"])


def _directors(details: dict) -> frozenset[str]:
    crew = details.get("credits", {}).get("crew", [])
    return _names(member for member in crew if member.get("job") == "Director")


def _top_cast(details: dict) -> frozenset[str]:
    cast = sorted(details.get("credits", {}).get("cast", []), key=lambda member: member["order"])
    return _names(cast[:TOP_CAST])


def _year(details: dict) -> float:
    release_date = details.get("release_date") or ""
    return float(release_date[:4]) if release_date[:4].isdigit() else np.nan


def _runtime(details: dict) -> float:
    return float(details["runtime"]) if details.get("runtime") else np.nan


def _vote_average(details: dict) -> float:
    return float(details["vote_average"]) if details.get("vote_average") else np.nan


def _vote_count(details: dict) -> float:
    return float(details["vote_count"]) if details.get("vote_count") else np.nan


def _popularity(details: dict) -> float:
    return float(details["popularity"]) if details.get("popularity") else np.nan


def _in_collection(details: dict) -> float:
    return 1.0 if details.get("belongs_to_collection") else 0.0


def _adult(details: dict) -> float:
    return 1.0 if details.get("adult") else 0.0


FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("genres", _genre_names),
    FeatureSpec("directors", _directors),
    FeatureSpec("cast", _top_cast),
    FeatureSpec("year", _year),
    FeatureSpec("runtime", _runtime),
    FeatureSpec("vote_average", _vote_average),
    FeatureSpec("vote_count", _vote_count),
    FeatureSpec("popularity", _popularity),
    FeatureSpec("in_collection", _in_collection),
    FeatureSpec("adult", _adult),
)


def extract_features(movies: pd.DataFrame, details: dict[int, dict]) -> pd.DataFrame:
    rows = {
        movie_id: {spec.name: spec.extract(details[tmdb_id]) for spec in FEATURES}
        for movie_id, tmdb_id in movies["tmdb_id"].items()
    }
    return pd.DataFrame.from_dict(rows, orient="index")


def _fill_numeric(series: pd.Series) -> np.ndarray:
    x = series.to_numpy(dtype=np.float64)
    median = np.nanmedian(x)
    if np.isnan(median):
        median = 0.0
    return np.where(np.isnan(x), median, x)


def _one_hot(values: pd.Series, prefix: str) -> tuple[np.ndarray, list[str]]:
    counts: Counter[str] = Counter()
    for items in values:
        if items:
            counts.update(items)
    vocab = list(counts)
    matrix = np.zeros((len(values), len(vocab)), dtype=np.float64)
    lookup = {name: j for j, name in enumerate(vocab)}
    for i, items in enumerate(values):
        if not items:
            continue
        for name in items:
            matrix[i, lookup[name]] = 1.0
    return matrix, [f"{prefix}:{name}" for name in vocab]


NUMERIC = ("year", "runtime", "vote_average", "vote_count", "popularity")
BINARY = ("in_collection", "adult")
SETS = (("genres", "genre"), ("directors", "director"), ("cast", "cast"))


@dataclass(frozen=True)
class TabularFeatures:
    feature_names: tuple[str, ...]
    movie_ids: np.ndarray
    values: np.ndarray

    @classmethod
    def from_raw(cls, raw: pd.DataFrame) -> "TabularFeatures":
        blocks: list[np.ndarray] = []
        names: list[str] = []
        for column, prefix in SETS:
            matrix, column_names = _one_hot(raw[column], prefix)
            blocks.append(matrix)
            names.extend(column_names)
        for column in NUMERIC:
            blocks.append(_fill_numeric(raw[column]))
            names.append(column)
        for column in BINARY:
            blocks.append(raw[column].to_numpy(dtype=np.float64))
            names.append(column)
        return cls(tuple(names), raw.index.to_numpy(), np.column_stack(blocks))

    def rows(self, movie_ids: np.ndarray) -> np.ndarray:
        lookup = pd.Series(np.arange(len(self.movie_ids)), index=self.movie_ids)
        return self.values[lookup.loc[movie_ids].to_numpy()]
