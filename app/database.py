"""MongoDB access layer for the Movie Recommender System.

This module is the *only* place that talks to MongoDB.  The recommendation
engine and the UI both consume the data returned from here, so the CSV files
are never read directly by the application logic.

Collections
-----------
``movies``   : ``{movieId, title, genres[]}``
``ratings``  : ``{userId, movieId, rating, timestamp}``

A unique compound index on ``(userId, movieId)`` guarantees that a rating can
only exist once, which makes the import in ``scripts/load_data.py`` idempotent.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable, Sequence

import pandas as pd
from pymongo import ASCENDING, DESCENDING, MongoClient, ReplaceOne
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import (
    BulkWriteError,
    ConnectionFailure,
    OperationFailure,
    PyMongoError,
    ServerSelectionTimeoutError,
)

from app.config import Settings, get_settings

LOGGER = logging.getLogger("movie_recommender.database")

MOVIES_COLLECTION = "movies"
RATINGS_COLLECTION = "ratings"

#: Documents written by a previous (possibly broken) import are removed before
#: a re-run so the loader is fully idempotent.
BATCH_SIZE = 2000


class DatabaseError(RuntimeError):
    """Raised when the database cannot be reached or a query fails."""


class DatabaseNotAvailable(DatabaseError):
    """Raised when a MongoDB connection cannot be established."""


def _candidate_uris(uri: str | None = None) -> list[str]:
    """Return the list of connection strings to try, in order of preference.

    The configured URI is always tried first.  When it points at a local
    ``localhost`` deployment we additionally attempt the ``127.0.0.1`` loopback
    alias, because some Windows setups only bind one of the two.
    """
    settings = get_settings()
    primary = (uri or settings.mongodb_uri or "").strip()
    if not primary:
        raise DatabaseNotAvailable(
            "MONGODB_URI is not set. Define it in your .env file, for example "
            "MONGODB_URI=mongodb://localhost:27017/"
        )

    candidates = [primary]
    for host in ("localhost", "127.0.0.1"):
        if host in primary:
            other = primary.replace(host, "127.0.0.1" if host == "localhost" else "localhost", 1)
            if other not in candidates:
                candidates.append(other)
    return candidates


class MovieDatabase:
    """Thin, dependency-free wrapper around a MongoDB database.

    Parameters
    ----------
    uri:
        Optional connection string override (defaults to ``MONGODB_URI``).
    db_name:
        Optional database name override (defaults to ``DB_NAME``).
    connect_timeout_ms:
        Server selection timeout; kept short so the UI fails fast with a
        helpful message instead of hanging.
    """

    def __init__(
        self,
        uri: str | None = None,
        db_name: str | None = None,
        connect_timeout_ms: int | None = None,
        _client: MongoClient | None = None,
    ) -> None:
        settings: Settings = get_settings()
        self.settings = settings
        self.uri = (uri if uri is not None else settings.mongodb_uri or "").strip()
        # An explicitly empty ``db_name`` must be reported as missing rather than
        # silently falling back to the default, so ``or`` is not used here.
        self.db_name = (
            db_name.strip() if db_name is not None else (settings.db_name or "").strip()
        )
        self.connect_timeout_ms = (
            connect_timeout_ms
            if connect_timeout_ms is not None
            else settings.server_selection_timeout_ms
        )
        #: An already-open client can be injected (used by the test suite so
        #: several helpers can share one connection pool).
        self._client: MongoClient | None = _client

    # ------------------------------------------------------------------
    # Connection handling
    # ------------------------------------------------------------------
    @property
    def client(self) -> MongoClient:
        """Return a live client, connecting lazily on first access."""
        if self._client is None:
            self.connect()
        assert self._client is not None  # narrowed by connect()
        return self._client

    def connect(self) -> MongoClient:
        """Open a MongoDB client, trying every candidate URI in turn.

        Raises
        ------
        DatabaseNotAvailable
            If no candidate URI yields a reachable server.
        """
        if self._client is not None:
            return self._client

        errors: list[str] = []
        for candidate in _candidate_uris(self.uri):
            try:
                client: MongoClient = MongoClient(
                    candidate,
                    serverSelectionTimeoutMS=self.connect_timeout_ms,
                    connectTimeoutMS=self.connect_timeout_ms,
                    socketTimeoutMS=30000,
                    retryWrites=True,
                    tz_aware=False,
                )
                client.admin.command("ping")
            except (ServerSelectionTimeoutError, ConnectionFailure, OperationFailure) as exc:
                errors.append(f"{candidate} -> {type(exc).__name__}: {exc}")
                try:
                    client.close()  # type: ignore[union-attr]
                except Exception:  # pragma: no cover - best effort cleanup
                    pass
                continue
            except PyMongoError as exc:  # pragma: no cover - defensive
                errors.append(f"{candidate} -> {type(exc).__name__}: {exc}")
                continue

            self._client = client
            LOGGER.info("Connected to MongoDB at %s (db=%s)", candidate, self.db_name)
            return client

        raise DatabaseNotAvailable(
            "Could not connect to MongoDB. Tried:\n  - " + "\n  - ".join(errors)
        )

    def close(self) -> None:
        """Close the underlying client if one was opened."""
        if self._client is not None:
            self._client.close()
            self._client = None
            LOGGER.info("MongoDB connection closed")

    def __enter__(self) -> "MovieDatabase":
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def is_connected(self) -> bool:
        """Best-effort connectivity probe that never raises."""
        try:
            self.client.admin.command("ping")
            return True
        except DatabaseError:
            return False
        except Exception:  # pragma: no cover - defensive
            return False

    def ping(self) -> bool:
        """Ping the server; returns ``True`` when the server answers."""
        return self.is_connected

    # ------------------------------------------------------------------
    # Collections
    # ------------------------------------------------------------------
    @property
    def db(self) -> Database:
        """Return the configured :class:`~pymongo.database.Database`."""
        if not self.db_name:
            raise DatabaseError("DB_NAME is not set. Define it in your .env file.")
        return self.client[self.db_name]

    @property
    def movies(self) -> Collection:
        return self.db[MOVIES_COLLECTION]

    @property
    def ratings(self) -> Collection:
        return self.db[RATINGS_COLLECTION]

    # ------------------------------------------------------------------
    # Indexes
    # ------------------------------------------------------------------
    def ensure_indexes(self) -> dict[str, list[str]]:
        """Create the indexes used by the application and return their names.

        Only ``ratings.(userId, movieId)`` is unique, which is what guarantees a
        user can never rate the same movie twice.  Every other index is a plain
        lookup/sort aid.
        """
        # (collection, key specification, unique)
        index_specs: list[tuple[str, list[tuple[str, Any]], bool]] = [
            (MOVIES_COLLECTION, [("movieId", ASCENDING)], True),
            (MOVIES_COLLECTION, [("title", ASCENDING)], False),
            (RATINGS_COLLECTION, [("userId", ASCENDING)], False),
            (RATINGS_COLLECTION, [("movieId", ASCENDING)], False),
            (RATINGS_COLLECTION, [("userId", ASCENDING), ("movieId", ASCENDING)], True),
            (RATINGS_COLLECTION, [("movieId", ASCENDING), ("rating", DESCENDING)], False),
        ]

        created: dict[str, list[str]] = {MOVIES_COLLECTION: [], RATINGS_COLLECTION: []}
        for name, spec, unique in index_specs:
            created[name].append(self._create_index(name, spec, unique))

        LOGGER.info(
            "Indexes ensured: movies=%s ratings=%s",
            created[MOVIES_COLLECTION],
            created[RATINGS_COLLECTION],
        )
        return created

    def _create_index(self, collection: str, spec: list[tuple[str, Any]], unique: bool) -> str:
        """Create one index, replacing a same-named index that has other options.

        MongoDB derives the index name from the keys, so requesting
        ``unique=True`` for a key that already exists as a non-unique index
        raises ``IndexKeySpecsConflict`` (code 86).  Dropping and recreating the
        index makes the loader self-healing.
        """
        try:
            return self.db[collection].create_index(spec, unique=unique)
        except OperationFailure as exc:
            if exc.code != 86:  # IndexKeySpecsConflict
                raise DatabaseError(
                    f"Failed to create index {spec} on {collection}: {exc}"
                ) from exc
            name = "_".join(f"{field}_{direction}" for field, direction in spec)
            LOGGER.info("Replacing index %s on %s with the requested options", name, collection)
            self.db[collection].drop_index(name)
            return self.db[collection].create_index(spec, unique=unique)

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------
    def upsert_movies(self, movies: Iterable[dict[str, Any]]) -> int:
        """Insert or replace movie documents keyed on ``movieId``.

        Returns
        -------
        int
            Number of documents written (matched + modified/inserted).
        """
        operations = []
        for doc in movies:
            if not isinstance(doc, dict) or "movieId" not in doc or "title" not in doc:
                LOGGER.warning("Skipping malformed movie document: %r", doc)
                continue
            operations.append(
                ReplaceOne({"movieId": doc["movieId"]}, doc, upsert=True)
            )
        if not operations:
            return 0
        result = self.movies.bulk_write(operations, ordered=False)
        return result.upserted_count + result.modified_count

    def upsert_ratings(self, ratings: Iterable[dict[str, Any]]) -> int:
        """Insert rating documents, ignoring duplicates.

        The unique compound index on ``(userId, movieId)`` makes re-inserting an
        existing pair a no-op, so the loader is idempotent.  Should the batch
        still trip a duplicate-key error (for example when a pre-existing unique
        index on some other field conflicts), the offending writes are skipped
        rather than aborting the whole import.
        """
        operations = []
        for doc in ratings:
            if (
                not isinstance(doc, dict)
                or "userId" not in doc
                or "movieId" not in doc
                or "rating" not in doc
            ):
                LOGGER.warning("Skipping malformed rating document: %r", doc)
                continue
            operations.append(
                ReplaceOne(
                    {"userId": doc["userId"], "movieId": doc["movieId"]},
                    doc,
                    upsert=True,
                )
            )
        if not operations:
            return 0
        try:
            result = self.ratings.bulk_write(operations, ordered=False)
            return result.upserted_count + result.modified_count
        except BulkWriteError as exc:
            write_errors = exc.details.get("writeErrors", [])
            duplicates = [e for e in write_errors if e.get("code") == 11000]
            if not duplicates:
                raise DatabaseError(f"Bulk insert of ratings failed: {exc}") from exc
            offending = {e.get("keyValue") for e in duplicates}
            LOGGER.error(
                "%d of %d rating write(s) violated a unique index and were "
                "skipped. Offending (userId, movieId) values: %s",
                len(duplicates),
                len(operations),
                sorted(offending, key=str)[:10],
            )
            raise DatabaseError(
                f"{len(duplicates)} rating document(s) conflict with an existing "
                "unique index. Drop the collections with 'python -m scripts.load_data "
                "--drop' and re-run the import."
            ) from exc

    def clear_collection(self, name: str) -> int:
        """Delete every document in a collection and return the deleted count."""
        if name not in (MOVIES_COLLECTION, RATINGS_COLLECTION):
            raise DatabaseError(f"Unknown collection: {name}")
        result = self.db[name].delete_many({})
        return result.deleted_count

    def drop_database(self) -> None:
        """Drop the whole database, including every index.

        Used by ``python -m scripts.load_data --drop`` to guarantee a clean
        slate, so a previously misconfigured index cannot linger.
        """
        if not self.db_name:
            raise DatabaseError("DB_NAME is not set. Define it in your .env file.")
        self.client.drop_database(self.db_name)
        LOGGER.info("Dropped database '%s'", self.db_name)

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------
    def count_documents(self, collection: str = RATINGS_COLLECTION, query: dict | None = None) -> int:
        """Count documents in a collection using the count index where possible.

        ``limit`` is deliberately omitted: MongoDB rejects a limit of 0 with
        "the limit must be positive", and omitting it means "no limit".
        """
        if collection not in (MOVIES_COLLECTION, RATINGS_COLLECTION):
            raise DatabaseError(f"Unknown collection: {collection}")
        return int(self.db[collection].count_documents(query or {}))

    def is_empty(self) -> bool:
        """True when either collection is empty (i.e. data not loaded yet)."""
        try:
            return (
                self.count_documents(RATINGS_COLLECTION) == 0
                or self.count_documents(MOVIES_COLLECTION) == 0
            )
        except DatabaseError:
            return True

    def get_statistics(self) -> dict[str, Any]:
        """Return the dashboard statistics, all read from MongoDB.

        Uses aggregation pipelines so the server performs the work instead of
        streaming documents to the client.
        """
        try:
            ratings_stats = list(
                self.ratings.aggregate(
                    [
                        {
                            "$group": {
                                "_id": None,
                                "total_ratings": {"$sum": 1},
                                "total_users": {"$addToSet": "$userId"},
                                "average_rating": {"$avg": "$rating"},
                            }
                        },
                        {
                            "$project": {
                                "total_ratings": 1,
                                "total_users": {"$size": "$total_users"},
                                "average_rating": 1,
                            }
                        },
                    ]
                )
            )
            total_movies = self.movies.count_documents({})

            if ratings_stats:
                stats = ratings_stats[0]
                return {
                    "total_movies": int(total_movies),
                    "total_users": int(stats.get("total_users", 0)),
                    "total_ratings": int(stats.get("total_ratings", 0)),
                    "average_rating": float(stats.get("average_rating") or 0.0),
                }
            return {
                "total_movies": int(total_movies),
                "total_users": 0,
                "total_ratings": 0,
                "average_rating": 0.0,
            }
        except DatabaseError:
            raise
        except PyMongoError as exc:
            raise DatabaseError(f"Failed to read statistics: {exc}") from exc

    def get_all_movies(self) -> pd.DataFrame:
        """Return every movie as a DataFrame (movieId, title, genres)."""
        try:
            cursor = self.movies.find({}, {"_id": 0, "movieId": 1, "title": 1, "genres": 1})
            records = list(cursor)
        except PyMongoError as exc:
            raise DatabaseError(f"Failed to read movies: {exc}") from exc
        return pd.DataFrame(records, columns=["movieId", "title", "genres"])

    def get_all_ratings(self) -> pd.DataFrame:
        """Return every rating as a DataFrame.

        Columns: ``userId, movieId, rating, timestamp``.  The timestamp is
        projected out of the documents to keep the transfer small because the
        recommender does not need it.
        """
        try:
            cursor = self.ratings.find(
                {}, {"_id": 0, "userId": 1, "movieId": 1, "rating": 1}
            )
            records = list(cursor)
        except PyMongoError as exc:
            raise DatabaseError(f"Failed to read ratings: {exc}") from exc
        return pd.DataFrame(records, columns=["userId", "movieId", "rating"])

    def get_user_ids(self) -> list[int]:
        """Return every userId that has at least one rating, ascending."""
        try:
            return sorted(
                int(row["_id"])
                for row in self.ratings.aggregate(
                    [{"$group": {"_id": "$userId"}}]
                )
            )
        except PyMongoError as exc:
            raise DatabaseError(f"Failed to list users: {exc}") from exc

    def get_user_ratings(self, user_id: int) -> pd.DataFrame:
        """Return one user's ratings joined with the movie metadata."""
        try:
            cursor = self.ratings.find(
                {"userId": int(user_id)},
                {"_id": 0, "movieId": 1, "rating": 1, "timestamp": 1},
            ).sort("rating", DESCENDING)
            rows = list(cursor)
        except PyMongoError as exc:
            raise DatabaseError(f"Failed to read ratings for user {user_id}: {exc}") from exc
        return self._join_movies(pd.DataFrame(rows, columns=["movieId", "rating", "timestamp"]))

    def get_rating_count_per_user(self) -> pd.DataFrame:
        """Return a DataFrame of ``userId`` / ``rating_count`` for all users."""
        try:
            rows = list(
                self.ratings.aggregate(
                    [
                        {"$group": {"_id": "$userId", "count": {"$sum": 1}}},
                        {"$sort": {"_id": 1}},
                    ]
                )
            )
        except PyMongoError as exc:
            raise DatabaseError(f"Failed to count user ratings: {exc}") from exc
        return pd.DataFrame(rows, columns=["userId", "rating_count"])

    def get_movie_rating_stats(self) -> pd.DataFrame:
        """Return per-movie ``movieId, average_rating, rating_count``."""
        try:
            rows = list(
                self.ratings.aggregate(
                    [
                        {
                            "$group": {
                                "_id": "$movieId",
                                "average_rating": {"$avg": "$rating"},
                                "rating_count": {"$sum": 1},
                            }
                        }
                    ]
                )
            )
        except PyMongoError as exc:
            raise DatabaseError(f"Failed to aggregate movie ratings: {exc}") from exc
        # The pipeline groups by movieId, so ``_id`` holds the movie id.
        return pd.DataFrame(
            rows, columns=["_id", "average_rating", "rating_count"]
        ).rename(columns={"_id": "movieId"})

    def get_movie_genres_map(self, movie_ids: Sequence[int] | None = None) -> dict[int, list[str]]:
        """Return ``{movieId: [genre, ...]}`` for the given ids (or all)."""
        query: dict[str, Any] = {}
        if movie_ids is not None:
            ids = [int(m) for m in movie_ids]
            if not ids:
                return {}
            query["movieId"] = {"$in": ids}
        try:
            cursor = self.movies.find(query, {"_id": 0, "movieId": 1, "genres": 1})
            return {int(doc["movieId"]): list(doc.get("genres", [])) for doc in cursor}
        except PyMongoError as exc:
            raise DatabaseError(f"Failed to read genres: {exc}") from exc

    def search_movies(self, query: str, limit: int = 50) -> pd.DataFrame:
        """Case-insensitive prefix/substring search over movie titles.

        Uses a MongoDB regex query with an index-friendly ordering; the
        community statistics are joined from the ``ratings`` collection with a
        single aggregation.
        """
        text = (query or "").strip()
        if not text:
            return pd.DataFrame(columns=["movieId", "title", "genres", "average_rating", "rating_count"])

        pattern = re.escape(text)
        match = {"title": {"$regex": pattern, "$options": "i"}}
        try:
            rows = list(
                self.movies.aggregate(
                    [
                        {"$match": match},
                        {
                            "$lookup": {
                                "from": RATINGS_COLLECTION,
                                "localField": "movieId",
                                "foreignField": "movieId",
                                "as": "_rating_stats",
                            }
                        },
                        {
                            "$addFields": {
                                "average_rating": {
                                    "$cond": [
                                        {"$gt": [{"$size": "$_rating_stats"}, 0]},
                                        {
                                            "$avg": "$_rating_stats.rating"
                                        },
                                        0.0,
                                    ]
                                },
                                "rating_count": {"$size": "$_rating_stats"},
                            }
                        },
                        {"$project": {"_id": 0, "rating_stats": 0, "_rating_stats": 0}},
                        {"$sort": {"rating_count": -1, "title": 1}},
                        {"$limit": int(limit)},
                    ]
                )
            )
        except PyMongoError as exc:
            raise DatabaseError(f"Movie search failed: {exc}") from exc

        return pd.DataFrame(
            rows,
            columns=["movieId", "title", "genres", "average_rating", "rating_count"],
        )

    def get_rating_distribution(self) -> pd.DataFrame:
        """Return the global rating distribution from the ``ratings`` collection."""
        try:
            rows = list(
                self.ratings.aggregate(
                    [
                        {"$group": {"_id": "$rating", "count": {"$sum": 1}}},
                        {"$sort": {"_id": 1}},
                    ]
                )
            )
        except PyMongoError as exc:
            raise DatabaseError(f"Failed to build rating distribution: {exc}") from exc
        # The pipeline groups by rating, so ``_id`` holds the star value.
        return pd.DataFrame(rows, columns=["_id", "count"]).rename(
            columns={"_id": "rating"}
        )

    def get_top_movies(self, limit: int = 10, min_ratings: int = 1) -> pd.DataFrame:
        """Return the highest rated movies (optionally requiring a min count)."""
        try:
            rows = list(
                self.ratings.aggregate(
                    [
                        {
                            "$group": {
                                "_id": "$movieId",
                                "average_rating": {"$avg": "$rating"},
                                "rating_count": {"$sum": 1},
                            }
                        },
                        {"$match": {"rating_count": {"$gte": int(min_ratings)}}},
                        {"$sort": {"average_rating": -1, "rating_count": -1}},
                        {"$limit": int(limit)},
                        {
                            "$lookup": {
                                "from": MOVIES_COLLECTION,
                                "localField": "_id",
                                "foreignField": "movieId",
                                "as": "movie",
                            }
                        },
                        {"$unwind": "$movie"},
                        {
                            "$project": {
                                "_id": 0,
                                "movieId": "$_id",
                                "title": "$movie.title",
                                "genres": "$movie.genres",
                                "average_rating": 1,
                                "rating_count": 1,
                            }
                        },
                    ]
                )
            )
        except PyMongoError as exc:
            raise DatabaseError(f"Failed to fetch top movies: {exc}") from exc
        return pd.DataFrame(
            rows,
            columns=["movieId", "title", "genres", "average_rating", "rating_count"],
        )

    def _join_movies(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """Attach movie metadata to a ratings frame using a single query."""
        if ratings.empty:
            ratings = ratings.copy()
            ratings["title"] = pd.Series(dtype="object")
            ratings["genres"] = pd.Series(dtype="object")
            return ratings
        movie_ids = ratings["movieId"].astype(int).unique().tolist()
        genres_map = self.get_movie_genres_map(movie_ids)
        titles = {
            int(m["movieId"]): m["title"]
            for m in self.movies.find(
                {"movieId": {"$in": movie_ids}}, {"_id": 0, "movieId": 1, "title": 1}
            )
        }
        out = ratings.copy()
        out["title"] = out["movieId"].map(titles)
        out["genres"] = out["movieId"].map(
            lambda mid: genres_map.get(int(mid), [])
        )
        return out


def get_database() -> MovieDatabase:
    """Factory returning a :class:`MovieDatabase` built from the environment."""
    return MovieDatabase()


def load_env_files() -> None:
    """Re-exported for convenience so scripts can trigger .env loading early."""
    from app.config import PROJECT_ROOT  # noqa: F401
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)


__all__ = [
    "DatabaseError",
    "DatabaseNotAvailable",
    "MovieDatabase",
    "get_database",
    "load_env_files",
    "MOVIES_COLLECTION",
    "RATINGS_COLLECTION",
    "BATCH_SIZE",
]
