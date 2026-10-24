from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from movie_knn.similarity import SIMILARITY_BY_TYPE, standardize

TOP_CAST = 8


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: str
    extract: Callable[[dict], object]


def _names(items: Iterable[dict], key: str = "name") -> frozenset[str]:
    return frozenset(item[key] for item in items)


def _crew(details: dict, predicate: Callable[[dict], bool]) -> frozenset[str]:
    return _names(member for member in details["credits"]["crew"] if predicate(member))


def _cast(details: dict) -> frozenset[str]:
    cast = sorted(details["credits"]["cast"], key=lambda member: member["order"])
    return _names(cast[:TOP_CAST])


def _year(details: dict) -> float:
    release_date = details.get("release_date") or ""
    return float(release_date[:4]) if release_date[:4].isdigit() else np.nan


def _positive(key: str) -> Callable[[dict], float]:
    return lambda details: float(details[key]) if details.get(key) else np.nan


FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("genres", "set", lambda d: _names(d["genres"])),
    FeatureSpec("keywords", "set", lambda d: _names(d["keywords"]["keywords"])),
    FeatureSpec("cast", "set", _cast),
    FeatureSpec("directors", "set", lambda d: _crew(d, lambda m: m["job"] == "Director")),
    FeatureSpec("writers", "set", lambda d: _crew(d, lambda m: m["department"] == "Writing")),
    FeatureSpec("composers", "set", lambda d: _crew(d, lambda m: m["job"] == "Original Music Composer")),
    FeatureSpec("companies", "set", lambda d: _names(d["production_companies"])),
    FeatureSpec("countries", "set", lambda d: _names(d["production_countries"], key="iso_3166_1")),
    FeatureSpec("language", "categorical", lambda d: d.get("original_language")),
    FeatureSpec("collection", "categorical", lambda d: (d.get("belongs_to_collection") or {}).get("name")),
    FeatureSpec("year", "numeric", _year),
    FeatureSpec("runtime", "numeric", _positive("runtime")),
    FeatureSpec("vote_average", "numeric", _positive("vote_average")),
    FeatureSpec("vote_count", "log_numeric", _positive("vote_count")),
    FeatureSpec("budget", "log_numeric", _positive("budget")),
    FeatureSpec("overview", "text", lambda d: " ".join(filter(None, [d.get("tagline"), d.get("overview")]))),
)


def extract_features(movies: pd.DataFrame, details: dict[int, dict]) -> pd.DataFrame:
    """Feature table (one row per movie_id) built from TMDB movie details."""
    rows = {
        movie_id: {spec.name: spec.extract(details[tmdb_id]) for spec in FEATURES}
        for movie_id, tmdb_id in movies["tmdb_id"].items()
    }
    return pd.DataFrame.from_dict(rows, orient="index")


@dataclass(frozen=True)
class SimilarityTensor:
    """Standardized per-feature similarity matrices, stacked as (n_features, n_movies, n_movies)."""

    feature_names: tuple[str, ...]
    movie_ids: np.ndarray
    values: np.ndarray

    @classmethod
    def from_features(cls, features: pd.DataFrame) -> "SimilarityTensor":
        kinds = {spec.name: spec.kind for spec in FEATURES}
        names = tuple(features.columns)
        matrices = [standardize(SIMILARITY_BY_TYPE[kinds[name]](features[name].tolist())) for name in names]
        return cls(names, features.index.to_numpy(), np.stack(matrices))

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def positions(self, movie_ids: np.ndarray) -> np.ndarray:
        lookup = pd.Series(np.arange(len(self.movie_ids)), index=self.movie_ids)
        return lookup.loc[movie_ids].to_numpy()
