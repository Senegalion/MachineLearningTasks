from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer


def set_similarity(values: Sequence[frozenset[str]]) -> np.ndarray:
    incidence = MultiLabelBinarizer(sparse_output=True).fit_transform(values).astype(np.float64)
    intersection = (incidence @ incidence.T).toarray()
    sizes = np.asarray(incidence.sum(axis=1)).ravel()
    union = sizes[:, None] + sizes[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def categorical_similarity(values: Sequence[str | None]) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    known = np.array([v is not None for v in values])
    return ((array[:, None] == array[None, :]) & known[:, None] & known[None, :]).astype(np.float64)


def numeric_similarity(values: Sequence[float], log_scale: bool = False) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if log_scale:
        x = np.log1p(x)
    low, high = np.nanpercentile(x, [5, 95])
    spread = high - low if high > low else 1.0
    similarity = 1.0 - np.clip(np.abs(x[:, None] - x[None, :]) / spread, 0.0, 1.0)
    missing = np.isnan(similarity)
    if missing.any():
        similarity[missing] = np.nanmedian(similarity)
    return similarity


def text_similarity(values: Sequence[str]) -> np.ndarray:
    tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2).fit_transform(values)
    return np.clip(cosine_similarity(tfidf), 0.0, 1.0)


SIMILARITY_BY_TYPE = {
    "set": set_similarity,
    "categorical": categorical_similarity,
    "numeric": numeric_similarity,
    "log_numeric": lambda values: numeric_similarity(values, log_scale=True),
    "text": text_similarity,
}


def standardize(similarity: np.ndarray) -> np.ndarray:
    off_diagonal = similarity[~np.eye(len(similarity), dtype=bool)]
    std = off_diagonal.std()
    if std == 0:
        return np.zeros_like(similarity)
    return (similarity - off_diagonal.mean()) / std
