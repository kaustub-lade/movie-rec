"""Environment driven configuration for the Movie Recommender System.

All credentials are read from the environment (a local ``.env`` file is loaded
when present) so that nothing sensitive is hardcoded in the source tree.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Project root == parent directory of the ``app`` package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load a .env file from the project root if it exists (never overrides real env).
load_dotenv(PROJECT_ROOT / ".env", override=False)

DEFAULT_MONGODB_URI = "mongodb://localhost:27017/"
DEFAULT_DB_NAME = "movie_recommender"
DEFAULT_DATASET_URL = (
    "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
)
DEFAULT_TIMEOUT_MS = 5000


def _as_int(name: str, default: int) -> int:
    """Read an integer environment variable, falling back on invalid input."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _as_float(name: str, default: float) -> float:
    """Read a float environment variable, falling back on invalid input."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the application configuration."""

    mongodb_uri: str = field(
        default_factory=lambda: os.getenv("MONGODB_URI", DEFAULT_MONGODB_URI)
    )
    db_name: str = field(default_factory=lambda: os.getenv("DB_NAME", DEFAULT_DB_NAME))
    dataset_url: str = field(
        default_factory=lambda: os.getenv("DATASET_URL", DEFAULT_DATASET_URL)
    )
    data_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("DATA_DIR", str(PROJECT_ROOT / "data"))
        ).expanduser()
    )
    server_selection_timeout_ms: int = field(
        default_factory=lambda: _as_int("MONGODB_TIMEOUT_MS", DEFAULT_TIMEOUT_MS)
    )
    mongodb_required: bool = field(
        default_factory=lambda: os.getenv("MONGODB_REQUIRED", "0").strip()
        not in {"0", "false", "False", ""}
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper()
    )
    default_top_n: int = field(default_factory=lambda: _as_int("TOP_N", 10))
    default_top_k: int = field(default_factory=lambda: _as_int("TOP_K", 10))
    min_neighbors: int = field(default_factory=lambda: _as_int("MIN_NEIGHBORS", 1))

    @property
    def mongodb_is_configured(self) -> bool:
        """True when a MongoDB URI has been supplied (explicitly or default)."""
        return bool(self.mongodb_uri.strip())

    @property
    def ratings_collection_name(self) -> str:
        return "ratings"

    @property
    def movies_collection_name(self) -> str:
        return "movies"

    def ensure_data_dir(self) -> Path:
        """Create the local data directory when missing and return it."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()


def configure_logging(level: str | None = None) -> logging.Logger:
    """Configure root logging once and return the application logger."""
    resolved = (level or get_settings().log_level or "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, resolved, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("movie_recommender")


settings = get_settings()
logger = configure_logging()
