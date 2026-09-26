"""Initialise the MongoDB database from the MovieLens *Latest Small* dataset.

Run with::

    python -m scripts.load_data

The script is **idempotent**: re-running it will not create duplicate movies or
ratings.  It downloads ``ml-latest-small.zip`` when missing, extracts
``movies.csv`` and ``ratings.csv``, creates the indexes (including the unique
``(userId, movieId)`` index that guarantees uniqueness) and bulk-inserts the
documents.

Options::

    python -m scripts.load_data --drop        # wipe the collections first
    python -m scripts.load_data --force       # re-download even if cached
    python -m scripts.load_data --uri mongodb://localhost:27017/
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import ssl
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Iterator, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.config import get_settings
from app.database import (
    BATCH_SIZE,
    MOVIES_COLLECTION,
    RATINGS_COLLECTION,
    DatabaseError,
    DatabaseNotAvailable,
    MovieDatabase,
)

LOGGER = logging.getLogger("movie_recommender.load_data")

REQUIRED_FILES = ("movies.csv", "ratings.csv")
DOWNLOAD_CHUNK = 1 << 16  # 64 KiB
REQUEST_TIMEOUT = 60
USER_AGENT = "movie-recommender/1.0 (+https://files.grouplens.org/datasets/movielens/)"


class DatasetError(RuntimeError):
    """Raised when the dataset cannot be downloaded, extracted or parsed."""


def _mask_uri(uri: str) -> str:
    """Hide credentials before writing a MongoDB URI to logs."""
    if "@" not in uri:
        return uri
    scheme, separator, rest = uri.partition("://")
    _, _, host = rest.partition("@")
    return f"{scheme}{separator}***:***@{host}"


def _human_bytes(size: float) -> str:
    """Format a byte count for the progress log."""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GB"



def _build_ssl_contexts() -> list[tuple[str, ssl.SSLContext | None]]:
    """Return (label, context) pairs to try, most trusted first.

    Some Windows / corporate-proxy setups ship without a usable system CA
    store, which makes ``https://files.grouplens.org`` fail with
    ``CERTIFICATE_VERIFY_FAILED``.  When that happens we retry with the CA
    bundle that ``certifi`` ships, so the loader works out of the box.
    """
    contexts: list[tuple[str, ssl.SSLContext | None]] = [("system", None)]
    try:
        import certifi  # noqa: PLC0415 - optional dependency, probed at runtime
    except ImportError:
        LOGGER.debug("certifi is not installed; only the system CA store will be used")
    else:
        try:
            contexts.append(("certifi", ssl.create_default_context(cafile=certifi.where())))
        except OSError as exc:  # pragma: no cover - broken certifi install
            LOGGER.warning("Could not load the certifi CA bundle: %s", exc)
    return contexts


def _urlopen(request: Request) -> Any:
    """Open ``request``, retrying with the certifi CA bundle on SSL errors.

    ``urlopen`` wraps TLS failures in :class:`urllib.error.URLError`, so the
    underlying :class:`ssl.SSLError` is unwrapped here to decide whether a retry
    with a different CA store is worth attempting.
    """
    problems: list[str] = []
    for label, context in _build_ssl_contexts():
        try:
            return urlopen(request, timeout=REQUEST_TIMEOUT, context=context)  # noqa: S310
        except URLError as exc:
            reason = exc.reason
            if isinstance(reason, ssl.SSLError):
                problems.append(f"{label}: {reason}")
                LOGGER.warning(
                    "TLS verification failed using the %s CA store: %s", label, reason
                )
                continue
            raise
        except ssl.SSLError as exc:
            problems.append(f"{label}: {exc}")
            LOGGER.warning(
                "TLS verification failed using the %s CA store: %s", label, exc
            )
            continue
    raise ssl.SSLError("; ".join(problems) or "TLS verification failed")


# ----------------------------------------------------------------------
# Download / extraction
# ----------------------------------------------------------------------
def _archive_name(url: str) -> str:
    """Derive the zip filename from the dataset URL."""
    name = url.rstrip("/").rsplit("/", 1)[-1]
    if not name.lower().endswith(".zip"):
        name = "ml-latest-small.zip"
    return name


def download_dataset(
    url: str | None = None,
    data_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Download the MovieLens zip archive if it is not already present.

    Returns the path to the downloaded ``.zip`` file.
    """
    settings = get_settings()
    url = url or settings.dataset_url
    data_dir = Path(data_dir or settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / _archive_name(url)

    if archive_path.exists() and archive_path.stat().st_size > 0 and not force:
        LOGGER.info("Dataset archive already present: %s", archive_path)
        return archive_path

    partial = archive_path.with_suffix(archive_path.suffix + ".part")
    LOGGER.info("Downloading dataset from %s", url)
    LOGGER.info("Destination: %s", archive_path)

    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with _urlopen(request) as response:  # noqa: S310
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            last_report = 0.0
            with partial.open("wb") as handle:
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        now = time.monotonic()
                        if now - last_report >= 1.0:
                            LOGGER.info(
                                "  %.1f%% (%s / %s)",
                                downloaded * 100 / total,
                                _human_bytes(downloaded),
                                _human_bytes(total),
                            )
                            last_report = now
    except ssl.SSLError as exc:
        partial.unlink(missing_ok=True)
        raise DatasetError(
            f"TLS verification failed while downloading {url}: {exc}.\n"
            "Install a CA bundle (for example 'pip install certifi') or download "
            f"the archive manually into {data_dir}."
        ) from exc
    except (HTTPError, URLError, OSError, TimeoutError) as exc:
        partial.unlink(missing_ok=True)
        raise DatasetError(
            f"Failed to download the dataset from {url}: {exc}. "
            "Check your internet connection or download the archive manually "
            f"into {data_dir}."
        ) from exc

    if downloaded == 0:
        partial.unlink(missing_ok=True)
        raise DatasetError(f"The download from {url} returned an empty archive.")

    partial.replace(archive_path)
    LOGGER.info("Download complete: %s (%s)", archive_path, _human_bytes(downloaded))
    return archive_path


def extract_dataset(archive_path: Path, data_dir: Path | None = None) -> dict[str, Path]:
    """Extract ``movies.csv`` and ``ratings.csv`` from the archive.

    Returns a mapping of logical name to extracted path.  Only the required
    files are extracted, and a corrupt archive raises :class:`DatasetError`.
    """
    data_dir = Path(data_dir or archive_path.parent)
    if not zipfile.is_zipfile(archive_path):
        raise DatasetError(
            f"{archive_path} is not a valid zip archive. Delete it and re-run "
            "the script to download it again."
        )

    extracted: dict[str, Path] = {}
    try:
        with zipfile.ZipFile(archive_path) as bundle:
            members = {Path(name).name: name for name in bundle.namelist()}
            missing = [name for name in REQUIRED_FILES if name not in members]
            if missing:
                raise DatasetError(
                    f"The archive is missing required file(s): {', '.join(missing)}"
                )
            for name in REQUIRED_FILES:
                target = data_dir / name
                if target.exists() and target.stat().st_size > 0:
                    LOGGER.info("CSV already present, keeping it: %s", target)
                else:
                    with bundle.open(members[name]) as source, target.open("wb") as sink:
                        shutil.copyfileobj(source, sink)
                    LOGGER.info("Extracted %s", target)
                extracted[name] = target
    except zipfile.BadZipFile as exc:
        raise DatasetError(f"Could not read {archive_path}: {exc}") from exc
    return extracted


def ensure_csv_files(data_dir: Path | None = None, force_download: bool = False) -> dict[str, Path]:
    """Make sure both CSVs exist locally, downloading/extracting if needed."""
    settings = get_settings()
    data_dir = Path(data_dir or settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    present = {name: data_dir / name for name in REQUIRED_FILES}
    if all(path.exists() and path.stat().st_size > 0 for path in present.values()):
        LOGGER.info("Found both CSV files in %s - skipping download", data_dir)
        return present

    archive = download_dataset(settings.dataset_url, data_dir, force=force_download)
    return extract_dataset(archive, data_dir)


# ----------------------------------------------------------------------
# CSV parsing
# ----------------------------------------------------------------------
def parse_movies(csv_path: Path) -> Iterator[dict[str, Any]]:
    """Yield movie documents from ``movies.csv``, skipping malformed rows.

    MovieLens encodes the genres as a pipe separated string; it becomes a list
    of genre strings in MongoDB.
    """
    LOGGER.info("Parsing movies from %s", csv_path)
    count = 0
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_no, row in enumerate(reader, start=2):
            raw_id = (row.get("movieId") or "").strip()
            title = (row.get("title") or "").strip()
            if not raw_id or not title:
                LOGGER.warning("Skipping movies.csv line %d: missing movieId/title", line_no)
                continue
            try:
                movie_id = int(raw_id)
            except ValueError:
                LOGGER.warning("Skipping movies.csv line %d: bad movieId %r", line_no, raw_id)
                continue
            raw_genres = (row.get("genres") or "").strip()
            genres = [g.strip() for g in raw_genres.split("|") if g.strip()]
            count += 1
            yield {"movieId": movie_id, "title": title, "genres": genres}
    LOGGER.info("Parsed %d movie record(s)", count)


def parse_ratings(csv_path: Path) -> Iterator[dict[str, Any]]:
    """Yield rating documents from ``ratings.csv``, skipping malformed rows."""
    LOGGER.info("Parsing ratings from %s", csv_path)
    count = 0
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_no, row in enumerate(reader, start=2):
            raw_user = (row.get("userId") or "").strip()
            raw_movie = (row.get("movieId") or "").strip()
            raw_rating = (row.get("rating") or "").strip()
            if not raw_user or not raw_movie or not raw_rating:
                LOGGER.warning("Skipping ratings.csv line %d: incomplete row", line_no)
                continue
            try:
                document = {
                    "userId": int(raw_user),
                    "movieId": int(raw_movie),
                    "rating": float(raw_rating),
                }
            except ValueError:
                LOGGER.warning("Skipping ratings.csv line %d: non-numeric value", line_no)
                continue
            raw_ts = (row.get("timestamp") or "").strip()
            if raw_ts:
                try:
                    document["timestamp"] = int(raw_ts)
                except ValueError:
                    LOGGER.warning("Ignoring bad timestamp on line %d", line_no)
            count += 1
            yield document
    LOGGER.info("Parsed %d rating record(s)", count)


def _batched(iterable: Any, size: int) -> Iterator[list[Any]]:
    """Yield lists of at most ``size`` items from ``iterable``."""
    batch: list[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


# ----------------------------------------------------------------------
# Import
# ----------------------------------------------------------------------
def import_dataset(
    db: MovieDatabase,
    movies_csv: Path,
    ratings_csv: Path,
    drop_existing: bool = False,
) -> dict[str, int]:
    """Import both CSV files into MongoDB and return the record counts.

    The operation is idempotent: re-running it leaves the stored totals
    unchanged.  The returned ``*_inserted`` figures are derived from the
    collection counts before and after the import, so they report only genuinely
    new documents.
    """
    if drop_existing:
        LOGGER.warning("Dropping the whole database '%s' ...", db.db_name)
        db.drop_database()

    before_movies = db.count_documents(MOVIES_COLLECTION)
    before_ratings = db.count_documents(RATINGS_COLLECTION)

    db.ensure_indexes()

    movies_processed = 0
    for batch in _batched(parse_movies(movies_csv), BATCH_SIZE):
        movies_processed += db.upsert_movies(batch)
        LOGGER.info("  movies: %d processed", movies_processed)

    ratings_processed = 0
    for batch in _batched(parse_ratings(ratings_csv), BATCH_SIZE):
        ratings_processed += db.upsert_ratings(batch)
        LOGGER.info("  ratings: %d processed", ratings_processed)

    movies_total = db.count_documents(MOVIES_COLLECTION)
    ratings_total = db.count_documents(RATINGS_COLLECTION)

    return {
        "movies_processed": movies_processed,
        "ratings_processed": ratings_processed,
        "movies_total": movies_total,
        "ratings_total": ratings_total,
        "movies_inserted": max(0, movies_total - before_movies),
        "ratings_inserted": max(0, ratings_total - before_ratings),
    }


def load(uri: str | None = None, drop: bool = False, force_download: bool = False) -> dict[str, int]:
    """End-to-end initialisation: download -> extract -> import -> index."""
    settings = get_settings()
    data_dir = settings.ensure_data_dir()

    LOGGER.info("=" * 66)
    LOGGER.info("Movie Recommender System - database initialisation")
    LOGGER.info("=" * 66)

    files = ensure_csv_files(data_dir, force_download=force_download)

    db = MovieDatabase(uri=uri or settings.mongodb_uri)
    LOGGER.info("Target MongoDB: %s (database '%s')", _mask_uri(db.uri), db.db_name)
    try:
        db.connect()
        counts = import_dataset(db, files["movies.csv"], files["ratings.csv"], drop_existing=drop)
    finally:
        db.close()

    LOGGER.info("-" * 66)
    LOGGER.info("Import finished:")
    LOGGER.info(
        "  movies  : %d parsed / %d inserted new / %d stored in total",
        counts["movies_processed"],
        counts["movies_inserted"],
        counts["movies_total"],
    )
    LOGGER.info(
        "  ratings : %d parsed / %d inserted new / %d stored in total",
        counts["ratings_processed"],
        counts["ratings_inserted"],
        counts["ratings_total"],
    )
    if counts["movies_inserted"] == 0 and counts["ratings_inserted"] == 0:
        LOGGER.info("  (database already up to date - no duplicates were created)")
    LOGGER.info("-" * 66)
    return counts


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.load_data",
        description="Download MovieLens and load it into MongoDB (idempotent).",
    )
    parser.add_argument("--uri", help="MongoDB URI override (default: MONGODB_URI)")
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop the database (documents and indexes) before importing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download the dataset archive even if it is already cached.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        load(uri=args.uri, drop=args.drop, force_download=args.force)
    except DatabaseNotAvailable as exc:
        LOGGER.error("MongoDB is unavailable: %s", exc)
        LOGGER.error(
            "Start a local server (mongod) or set MONGODB_URI in your .env file "
            "to an Atlas connection string, then run this command again."
        )
        return 2
    except DatabaseError as exc:
        LOGGER.error("Database error: %s", exc)
        return 3
    except DatasetError as exc:
        LOGGER.error("Dataset error: %s", exc)
        return 4
    except KeyboardInterrupt:  # pragma: no cover
        LOGGER.warning("Interrupted by user.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
