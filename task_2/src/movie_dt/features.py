from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

TOP_CAST = 8
MIN_SET_DF = 3
MAX_SET_ITEMS = 40
MAX_TEXT_FEATURES = 20


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
    """Raw TMDB attributes, one row per movie_id. Sets stay as frozensets until they are binarized."""
    rows = {
        movie_id: {spec.name: spec.extract(details[tmdb_id]) for spec in FEATURES}
        for movie_id, tmdb_id in movies["tmdb_id"].items()
    }
    return pd.DataFrame.from_dict(rows, orient="index")


def _encode_numeric(values: pd.Series, log_scale: bool) -> np.ndarray:
    x = values.to_numpy(dtype=np.float64)
    if log_scale:
        x = np.log1p(x)
    median = np.nanmedian(x)
    if np.isnan(median):
        median = 0.0
    return np.where(np.isnan(x), median, x)


def _encode_set(values: pd.Series, prefix: str, min_df: int, max_items: int) -> tuple[np.ndarray, list[str]]:
    counts: Counter[str] = Counter()
    for items in values:
        if items:
            counts.update(items)
    vocab = [item for item, count in counts.most_common() if count >= min_df][:max_items]
    matrix = np.zeros((len(values), len(vocab)), dtype=np.float64)
    lookup = {item: j for j, item in enumerate(vocab)}
    for i, items in enumerate(values):
        if not items:
            continue
        for item in items:
            j = lookup.get(item)
            if j is not None:
                matrix[i, j] = 1.0
    return matrix, [f"{prefix}:{item}" for item in vocab]


def _encode_categorical(values: pd.Series, prefix: str, min_df: int) -> tuple[np.ndarray, list[str]]:
    counts = values.value_counts(dropna=True)
    vocab = [value for value, count in counts.items() if count >= min_df]
    matrix = np.zeros((len(values), len(vocab)), dtype=np.float64)
    lookup = {value: j for j, value in enumerate(vocab)}
    for i, value in enumerate(values):
        j = lookup.get(value)
        if j is not None:
            matrix[i, j] = 1.0
    return matrix, [f"{prefix}:{value}" for value in vocab]


def _encode_text(values: pd.Series, prefix: str, max_features: int) -> tuple[np.ndarray, list[str]]:
    documents = [text if isinstance(text, str) else "" for text in values]
    if max_features <= 0 or not any(documents):
        return np.zeros((len(values), 0)), []
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2, max_features=max_features)
    matrix = vectorizer.fit_transform(documents).toarray()
    names = [f"{prefix}:{token}" for token in vectorizer.get_feature_names_out()]
    return matrix, names


@dataclass(frozen=True)
class TabularFeatures:
    """Numeric matrix of TMDB attributes used as candidate split conditions."""

    feature_names: tuple[str, ...]
    movie_ids: np.ndarray
    values: np.ndarray

    @classmethod
    def from_raw(
        cls,
        raw: pd.DataFrame,
        min_df: int = MIN_SET_DF,
        max_set_items: int = MAX_SET_ITEMS,
        max_text_features: int = MAX_TEXT_FEATURES,
    ) -> "TabularFeatures":
        kinds = {spec.name: spec.kind for spec in FEATURES}
        blocks: list[np.ndarray] = []
        names: list[str] = []
        for column, kind in kinds.items():
            series = raw[column]
            if kind == "numeric":
                blocks.append(_encode_numeric(series, log_scale=False)[:, None])
                names.append(column)
            elif kind == "log_numeric":
                blocks.append(_encode_numeric(series, log_scale=True)[:, None])
                names.append(column)
            elif kind == "set":
                cap = None if column == "genres" else max_set_items
                matrix, column_names = _encode_set(series, column, min_df, cap or len(series))
                blocks.append(matrix)
                names.extend(column_names)
            elif kind == "categorical":
                matrix, column_names = _encode_categorical(series, column, min_df=2)
                blocks.append(matrix)
                names.extend(column_names)
            elif kind == "text":
                matrix, column_names = _encode_text(series, column, max_text_features)
                blocks.append(matrix)
                names.extend(column_names)
        values = np.hstack(blocks) if blocks else np.zeros((len(raw), 0))
        return cls(tuple(names), raw.index.to_numpy(), values)

    def rows(self, movie_ids: np.ndarray) -> np.ndarray:
        lookup = pd.Series(np.arange(len(self.movie_ids)), index=self.movie_ids)
        return self.values[lookup.loc[movie_ids].to_numpy()]
