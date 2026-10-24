import json
import logging
import os
import time
from collections.abc import Iterable
from pathlib import Path

import requests
from dotenv import load_dotenv

from movie_knn import config

logger = logging.getLogger(__name__)

API_URL = "https://api.themoviedb.org/3"
APPENDED_RESOURCES = "credits,keywords"
MAX_RETRIES = 5


class TMDBClient:
    """Minimal TMDB v3 client accepting either an API key or a v4 read access token."""

    def __init__(self, credential: str, timeout: float = 10.0) -> None:
        self._session = requests.Session()
        self._params: dict[str, str] = {}
        self._timeout = timeout
        if credential.startswith("eyJ"):
            self._session.headers["Authorization"] = f"Bearer {credential}"
        else:
            self._params["api_key"] = credential

    @classmethod
    def from_env(cls) -> "TMDBClient":
        load_dotenv(config.PROJECT_ROOT / ".env")
        credential = os.environ.get("TMDB_API_KEY", "").strip()
        if not credential:
            raise RuntimeError("TMDB_API_KEY is not set (see .env.example).")
        return cls(credential)

    def movie_details(self, tmdb_id: int) -> dict:
        params = {**self._params, "append_to_response": APPENDED_RESOURCES, "language": "en-US"}
        url = f"{API_URL}/movie/{tmdb_id}"
        for attempt in range(MAX_RETRIES):
            response = self._session.get(url, params=params, timeout=self._timeout)
            if response.status_code == 429 or response.status_code >= 500:
                delay = float(response.headers.get("Retry-After", 2**attempt))
                logger.warning("TMDB returned %s for %s, retrying in %.0fs", response.status_code, tmdb_id, delay)
                time.sleep(delay)
                continue
            response.raise_for_status()
            return response.json()
        raise RuntimeError(f"TMDB request for movie {tmdb_id} failed after {MAX_RETRIES} attempts.")


def cache_path(tmdb_id: int, cache_dir: Path = config.TMDB_CACHE_DIR) -> Path:
    return cache_dir / f"{tmdb_id}.json"


def fetch_movies(tmdb_ids: Iterable[int], cache_dir: Path = config.TMDB_CACHE_DIR, force: bool = False) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [i for i in tmdb_ids if force or not cache_path(i, cache_dir).exists()]
    if not missing:
        logger.info("All movie details already cached in %s", cache_dir)
        return

    client = TMDBClient.from_env()
    for n, tmdb_id in enumerate(missing, start=1):
        details = client.movie_details(tmdb_id)
        cache_path(tmdb_id, cache_dir).write_text(json.dumps(details, ensure_ascii=False, indent=1), encoding="utf-8")
        logger.info("[%d/%d] %s", n, len(missing), details.get("title", tmdb_id))


def load_cached(tmdb_ids: Iterable[int], cache_dir: Path = config.TMDB_CACHE_DIR) -> dict[int, dict]:
    details = {}
    for tmdb_id in tmdb_ids:
        path = cache_path(tmdb_id, cache_dir)
        if not path.exists():
            raise FileNotFoundError(f"Missing TMDB data for movie {tmdb_id}; run the 'fetch' command first.")
        details[tmdb_id] = json.loads(path.read_text(encoding="utf-8"))
    return details
