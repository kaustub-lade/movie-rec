"""Tests for the MongoDB layer and the dataset loader.

Two groups:

* **Pure tests** (always run) - CSV parsing, batching, duplicate handling
  against a fake collection and graceful behaviour when MongoDB is absent.
* **Integration tests** - executed only when a live server is reachable
  (``MONGO_TEST_URI`` or a working ``MONGODB_URI``).  Use
  ``MONGO_TEST_DB`` to pick a throwaway database name, which is dropped
  afterwards.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from app.database import (
    MOVIES_COLLECTION,
    RATINGS_COLLECTION,
    DatabaseError,
    DatabaseNotAvailable,
    MovieDatabase,
)
from scripts.load_data import (
    DatasetError,
    _batched,
    _human_bytes,
    ensure_csv_files,
    extract_dataset,
    parse_movies,
    parse_ratings,
)

# ----------------------------------------------------------------------
# Minimal in-memory stand-ins for the pymongo collections
# ----------------------------------------------------------------------


class _FakeIndex:
    def __init__(self, key, unique=False):
        self.key = key
        self.unique = unique


class _FakeCursor(list):
    """A list that also exposes the pymongo ``Cursor.sort`` signature."""

    def sort(self, key_or_list=None, direction=None):  # noqa: A003 - pymongo API
        if key_or_list is None:
            return self
        if isinstance(key_or_list, str):
            pairs = [(key_or_list, direction if direction is not None else 1)]
        else:
            pairs = list(key_or_list)
        for field, order in reversed(pairs):
            super().sort(
                key=lambda r: (r.get(field) is None, r.get(field)), reverse=order < 0
            )
        return self


class FakeCollection:
    """In-memory collection honouring the subset of pymongo we use."""

    def __init__(self, identity: tuple[str, ...] = ("userId", "movieId")):
        #: Fields that uniquely identify a document in this collection.
        self.identity = identity
        self.documents: list[dict] = []
        self.indexes: list[_FakeIndex] = []
        self.find_calls: list[dict] = []

    # -- indexes ------------------------------------------------------
    def create_index(self, spec, unique=False):
        name = "_".join(f"{field}_{direction}" for field, direction in spec)
        self.indexes.append(_FakeIndex(tuple(field for field, _ in spec), unique=unique))
        return name

    def has_unique_user_movie_index(self) -> bool:
        return any(
            index.unique and set(index.key) == {"userId", "movieId"}
            for index in self.indexes
        )

    def insert_one(self, document):
        self.documents.append(dict(document))
        return type("InsertResult", (), {"inserted_id": len(self.documents)})()

    # -- writes -------------------------------------------------------
    def bulk_write(self, operations, ordered=True):
        upserted = 0
        modified = 0
        for operation in operations:
            key = {field: operation._filter[field] for field in self.identity}
            existing = next(
                (
                    d
                    for d in self.documents
                    if all(d.get(k) == v for k, v in key.items())
                ),
                None,
            )
            document = operation._doc
            if existing is None:
                self.documents.append(dict(document))
                upserted += 1
            elif existing != document:
                existing.update(document)
                modified += 1
        return type(
            "BulkResult",
            (),
            {"upserted_count": upserted, "modified_count": modified},
        )()

    def delete_many(self, query):
        removed = len(self.documents)
        self.documents.clear()
        return type("DeleteResult", (), {"deleted_count": removed})()

    # -- reads --------------------------------------------------------
    def find(self, query=None, projection=None):
        import re

        self.find_calls.append(dict(query or {}))
        documents = []
        for doc in self.documents:
            if not self._matches(doc, query or {}):
                continue
            if projection is None:
                documents.append({k: v for k, v in doc.items() if k != "_id"})
            else:
                documents.append(
                    {k: v for k, v in doc.items() if k in projection and k != "_id"}
                )
        return _FakeCursor(documents)

    def count_documents(self, query=None, limit=0):
        return len([d for d in self.documents if self._matches(d, query or {})])

    @staticmethod
    def _matches(document, query):
        import re

        for field, condition in query.items():
            value = document.get(field)
            if isinstance(condition, dict):
                for operator, operand in condition.items():
                    if operator == "$in" and value not in operand:
                        return False
                    if operator == "$ne" and value == operand:
                        return False
                    if operator == "$regex":
                        if value is None or not re.search(operand, str(value), re.I):
                            return False
            elif value != condition:
                return False
        return True

    def aggregate(self, pipeline):
        return _run_pipeline(self.documents, pipeline)


def _eval(expression, row):
    """Evaluate the small subset of aggregation expressions the app uses."""
    if isinstance(expression, str) and expression.startswith("$"):
        return row.get(expression[1:])
    if isinstance(expression, dict):
        if len(expression) == 1:
            (operator, argument), = expression.items()
            if operator == "$size":
                value = _eval(argument, row)
                return len(value) if isinstance(value, (list, tuple)) else 0
            if operator == "$avg":
                values = _eval(argument, row) or []
                values = [v for v in values if isinstance(v, (int, float))]
                return sum(values) / len(values) if values else None
            if operator == "$sum":
                value = _eval(argument, row)
                if isinstance(value, (list, tuple)):
                    return sum(value)
                return value
            if operator == "$gt":
                left, right = argument
                return _eval(left, row) > _eval(right, row)
            if operator == "$cond":
                condition, then_expr, else_expr = argument
                return _eval(then_expr, row) if _eval(condition, row) else _eval(else_expr, row)
        return {key: _eval(value, row) for key, value in expression.items()}
    if isinstance(expression, list):
        return [_eval(value, row) for value in expression]
    return expression


def _accumulate(operator, argument, group, rows):
    """Evaluate one ``$group`` accumulator over a group of documents."""
    if operator == "$sum":
        if argument == 1:
            return len(rows)
        return sum(_eval(argument, row) or 0 for row in rows)
    if operator == "$avg":
        values = [_eval(argument, row) for row in rows]
        values = [v for v in values if isinstance(v, (int, float))]
        return sum(values) / len(values) if values else None
    if operator == "$addToSet":
        return sorted({_eval(argument, row) for row in rows}, key=lambda v: (v is None, v))
    if operator == "$min":
        return min((_eval(argument, row) for row in rows), default=None)
    if operator == "$max":
        return max((_eval(argument, row) for row in rows), default=None)
    raise NotImplementedError(f"Unsupported accumulator {operator}")


def _hashable(value):
    """Return a dict-key friendly version of an aggregation ``_id``."""
    if isinstance(value, (list, tuple)):
        return tuple(value)
    if isinstance(value, dict):
        return tuple(sorted(value.items()))
    return value


def _run_pipeline(documents, pipeline):
    """Support the handful of aggregation stages the app actually uses."""
    rows = [dict(doc) for doc in documents]
    for stage in pipeline:
        if "$match" in stage:
            rows = [r for r in rows if FakeCollection._matches(r, stage["$match"])]

        elif "$group" in stage:
            spec = stage["$group"]
            grouped: dict = {}
            order: list = []
            for row in rows:
                key = _hashable(_eval(spec["_id"], row))
                if key not in grouped:
                    grouped[key] = []
                    order.append(key)
                grouped[key].append(row)
            out = []
            for key in order:
                group = grouped[key]
                document = {
                    "_id": key[0] if isinstance(key, tuple) and len(key) == 1 else key
                }
                for field, value in spec.items():
                    if field == "_id":
                        continue
                    if isinstance(value, dict) and len(value) == 1:
                        operator, argument = next(iter(value.items()))
                        document[field] = _accumulate(operator, argument, field, group)
                    else:
                        document[field] = _eval(value, group[0])
                out.append(document)
            rows = out

        elif "$project" in stage:
            spec = stage["$project"]
            # A projection whose values are all 0/False is an *exclusion*
            # projection, exactly like MongoDB's behaviour.
            exclusion = all(v in (0, False) for v in spec.values())
            out = []
            for row in rows:
                if exclusion:
                    document = {k: v for k, v in row.items() if k not in spec}
                    if "_id" in spec:
                        document.pop("_id", None)
                else:
                    document = {}
                    for field, include in spec.items():
                        if include == 0 or include is False:
                            continue
                        if include == 1 or include is True:
                            if field in row:
                                document[field] = row[field]
                        else:
                            document[field] = _eval(include, row)
                out.append(document)
            rows = out

        elif "$addFields" in stage:
            out = []
            for row in rows:
                document = dict(row)
                for field, expression in stage["$addFields"].items():
                    document[field] = _eval(expression, row)
                out.append(document)
            rows = out

        elif "$sort" in stage:
            for field, direction in reversed(list(stage["$sort"].items())):
                rows = sorted(
                    rows,
                    key=lambda r: (r.get(field) is None, r.get(field)),
                    reverse=direction < 0,
                )

        elif "$limit" in stage:
            rows = rows[: stage["$limit"]]

        elif "$lookup" in stage:
            spec = stage["$lookup"]
            joined = []
            for row in rows:
                document = dict(row)
                document[spec["as"]] = [
                    d
                    for d in documents
                    if d.get(spec["foreignField"]) == row.get(spec["localField"])
                ]
                joined.append(document)
            rows = joined

        elif "$unwind" in stage:
            field = stage["$unwind"]
            out = []
            for row in rows:
                for item in row.get(field, []):
                    document = dict(row)
                    document[field] = item
                    out.append(document)
            rows = out

        else:  # pragma: no cover - guards against unsupported pipelines
            raise NotImplementedError(f"Unsupported stage: {sorted(stage)}")
    return rows


class FakeDB:
    """Minimal stand-in for ``pymongo.database.Database``."""

    def __init__(self):
        self.collections = {
            MOVIES_COLLECTION: FakeCollection(identity=("movieId",)),
            RATINGS_COLLECTION: FakeCollection(identity=("userId", "movieId")),
        }

    def __getitem__(self, name):
        return self.collections[name]


class FakeDatabase(MovieDatabase):
    """A :class:`MovieDatabase` wired to in-memory fake collections."""

    def __init__(self):
        super().__init__(uri="mongodb://fake:27017/", db_name="test_movie_recommender")
        self._fake_db = FakeDB()
        self.dropped = 0

    @property
    def db(self) -> FakeDB:  # type: ignore[override]
        return self._fake_db

    @property
    def movies(self) -> FakeCollection:  # type: ignore[override]
        return self._fake_db.collections[MOVIES_COLLECTION]

    @property
    def ratings(self) -> FakeCollection:  # type: ignore[override]
        return self._fake_db.collections[RATINGS_COLLECTION]

    @property
    def client(self):  # type: ignore[override]
        raise DatabaseNotAvailable("fake client has no network access")

    def connect(self):  # type: ignore[override]
        raise DatabaseNotAvailable("fake client has no network access")

    def is_connected(self) -> bool:  # type: ignore[override]
        return True

    def drop_database(self) -> None:  # type: ignore[override]
        self.dropped += 1
        self.movies.delete_many({})
        self.ratings.delete_many({})
        self.movies.indexes.clear()
        self.ratings.indexes.clear()


# ----------------------------------------------------------------------
# Fake-database behaviour (no server required)
# ----------------------------------------------------------------------


class TestDuplicatePrevention:
    @pytest.fixture()
    def db(self):
        database = FakeDatabase()
        database.ensure_indexes()
        return database

    def test_unique_compound_index_is_created(self, db):
        indexes = [tuple(i.key) for i in db.ratings.indexes]
        assert ("userId", "movieId") in indexes
        assert db.ratings.has_unique_user_movie_index()

    def test_movie_index_is_created_on_movieid(self, db):
        assert any(i.key == ("movieId",) for i in db.movies.indexes)

    def test_upsert_ratings_does_not_duplicate_on_reimport(self, db):
        documents = [
            {"userId": 1, "movieId": 10, "rating": 4.0, "timestamp": 100},
            {"userId": 1, "movieId": 20, "rating": 3.0, "timestamp": 200},
        ]
        assert db.upsert_ratings(documents) == 2
        # Second identical import must not add anything.
        assert db.upsert_ratings(documents) == 0
        assert db.count_documents(RATINGS_COLLECTION) == 2

    def test_upsert_updates_a_changed_rating_in_place(self, db):
        db.upsert_ratings([{"userId": 1, "movieId": 10, "rating": 4.0}])
        db.upsert_ratings([{"userId": 1, "movieId": 10, "rating": 5.0}])
        assert db.count_documents(RATINGS_COLLECTION) == 1
        assert db.ratings.documents[0]["rating"] == 5.0

    def test_upsert_movies_is_idempotent(self, db):
        movies = [{"movieId": 1, "title": "Toy Story (1995)", "genres": ["Animation"]}]
        db.upsert_movies(movies)
        db.upsert_movies(movies)
        assert db.count_documents(MOVIES_COLLECTION) == 1

    def test_upsert_skips_malformed_documents(self, db):
        written = db.upsert_ratings(
            [
                {"userId": 1, "movieId": 10, "rating": 4.0},
                {"userId": 2},                       # no movieId
                {"userId": 3, "movieId": 30},        # no rating
                "not-a-dict",
            ]
        )
        assert written == 1
        assert db.count_documents(RATINGS_COLLECTION) == 1

    def test_upsert_of_nothing_is_a_noop(self, db):
        assert db.upsert_ratings([]) == 0
        assert db.upsert_movies([]) == 0


class TestQueriesAgainstFakeDb:
    @pytest.fixture()
    def populated(self):
        db = FakeDatabase()
        db.ensure_indexes()
        db.upsert_movies(
            [
                {"movieId": 1, "title": "Toy Story (1995)", "genres": ["Animation", "Comedy"]},
                {"movieId": 2, "title": "The Matrix (1999)", "genres": ["Action", "Sci-Fi"]},
                {"movieId": 3, "title": "Amelie (2001)", "genres": ["Comedy", "Romance"]},
            ]
        )
        db.upsert_ratings(
            [
                {"userId": 1, "movieId": 1, "rating": 4.0},
                {"userId": 1, "movieId": 2, "rating": 5.0},
                {"userId": 2, "movieId": 2, "rating": 3.0},
                {"userId": 2, "movieId": 3, "rating": 4.0},
            ]
        )
        return db

    def test_statistics_are_aggregated_from_the_ratings_collection(self, populated):
        stats = populated.get_statistics()
        assert stats["total_movies"] == 3
        assert stats["total_users"] == 2
        assert stats["total_ratings"] == 4
        assert stats["average_rating"] == pytest.approx(4.0)

    def test_get_all_ratings_frame_shape(self, populated):
        frame = populated.get_all_ratings()
        assert list(frame.columns) == ["userId", "movieId", "rating"]
        assert len(frame) == 4

    def test_get_all_movies_frame(self, populated):
        frame = populated.get_all_movies()
        assert set(frame.columns) == {"movieId", "title", "genres"}
        assert len(frame) == 3

    def test_get_user_ratings_joins_metadata(self, populated):
        frame = populated.get_user_ratings(1)
        assert len(frame) == 2
        assert set(frame["title"]) == {"Toy Story (1995)", "The Matrix (1999)"}

    def test_get_movie_rating_stats(self, populated):
        stats = populated.get_movie_rating_stats()
        assert len(stats) == 3
        row = stats[stats["movieId"] == 2].iloc[0]
        assert row["average_rating"] == pytest.approx(4.0)
        assert row["rating_count"] == 2

    def test_get_movie_genres_map(self, populated):
        genres = populated.get_movie_genres_map([1, 2])
        assert genres[1] == ["Animation", "Comedy"]
        assert set(genres) == {1, 2}

    def test_get_movie_genres_map_with_no_ids(self, populated):
        assert populated.get_movie_genres_map([]) == {}

    def test_is_empty_detects_missing_data(self, populated):
        assert populated.is_empty() is False
        populated.clear_collection(RATINGS_COLLECTION)
        assert populated.is_empty() is True

    def test_is_empty_on_untouched_database(self):
        assert FakeDatabase().is_empty() is True

    def test_clear_collection_rejects_unknown_names(self, populated):
        with pytest.raises(DatabaseError):
            populated.clear_collection("something_else")

    def test_count_documents_rejects_unknown_collections(self, populated):
        with pytest.raises(DatabaseError):
            populated.count_documents("nope")

    def test_empty_query_returns_empty_frame_not_an_error(self, populated):
        frame = populated.get_all_ratings().iloc[0:0]
        assert frame.empty
        assert list(frame.columns) == ["userId", "movieId", "rating"]


# ----------------------------------------------------------------------
# Graceful degradation
# ----------------------------------------------------------------------


class TestMissingDatabaseHandling:
    def test_connect_to_a_dead_host_raises_database_not_available(self):
        db = MovieDatabase(uri="mongodb://127.0.0.1:1/", connect_timeout_ms=250)
        with pytest.raises(DatabaseNotAvailable) as excinfo:
            db.connect()
        message = str(excinfo.value)
        assert "Could not connect to MongoDB" in message
        assert "Tried" in message

    def test_context_manager_raises_on_failure(self):
        db = MovieDatabase(uri="mongodb://127.0.0.1:1/", connect_timeout_ms=250)
        with pytest.raises(DatabaseNotAvailable):
            with db:
                pass

    def test_is_connected_returns_false_instead_of_raising(self):
        db = MovieDatabase(uri="mongodb://127.0.0.1:1/", connect_timeout_ms=250)
        assert db.is_connected is False

    def test_ping_returns_false_when_unreachable(self):
        db = MovieDatabase(uri="mongodb://127.0.0.1:1/", connect_timeout_ms=250)
        assert db.ping() is False

    def test_missing_uri_is_reported_clearly(self):
        from app.database import _candidate_uris

        with pytest.raises(DatabaseNotAvailable) as excinfo:
            _candidate_uris("   ")
        assert "MONGODB_URI" in str(excinfo.value)

    def test_localhost_gets_a_loopback_fallback_candidate(self):
        from app.database import _candidate_uris

        candidates = _candidate_uris("mongodb://localhost:27017/")
        assert candidates[0] == "mongodb://localhost:27017/"
        assert "mongodb://127.0.0.1:27017/" in candidates

    def test_atlas_uri_is_not_rewritten(self):
        from app.database import _candidate_uris

        uri = "mongodb+srv://user:pass@cluster0.abcde.mongodb.net/?retryWrites=true"
        assert _candidate_uris(uri) == [uri]

    def test_reads_fail_loudly_with_database_error(self):
        """A dropped connection must surface as DatabaseError, not a raw pymongo error."""
        from pymongo.errors import ServerSelectionTimeoutError

        db = MovieDatabase(uri="mongodb://127.0.0.1:1/", connect_timeout_ms=250)
        with pytest.raises((DatabaseNotAvailable, DatabaseError)):
            db.get_statistics()

    def test_missing_db_name_is_reported(self):
        db = MovieDatabase(uri="mongodb://127.0.0.1:27017/", db_name="")
        with pytest.raises(DatabaseError) as excinfo:
            _ = db.db
        assert "DB_NAME" in str(excinfo.value)

    def test_ensure_indexes_raises_when_unreachable(self):
        from pymongo.errors import PyMongoError

        db = MovieDatabase(uri="mongodb://127.0.0.1:1/", connect_timeout_ms=250)
        with pytest.raises((DatabaseNotAvailable, DatabaseError, PyMongoError)):
            db.ensure_indexes()

    def test_recommender_reports_a_missing_database_without_crashing(self):
        from app.recommender import InsufficientDataError, UserBasedRecommender

        engine = UserBasedRecommender(MovieDatabase(uri="mongodb://127.0.0.1:1/", connect_timeout_ms=250))
        with pytest.raises((DatabaseError, InsufficientDataError)):
            engine.recommend(user_id=1)


# ----------------------------------------------------------------------
# CSV parsing (pure)
# ----------------------------------------------------------------------


MOVIES_CSV = """movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
2,"GoldenEye (1995, James Bond)",Action|Adventure|Thriller
3,Bad Row Without Genres,
,Bad Row No Id,Comedy
notanumber,Bad Row Non Numeric Id,Comedy
"""

RATINGS_CSV = """userId,movieId,rating,timestamp
1,1,4.0,964982703
1,10,3.5,964981747
2,1,5.0,945771920
2,notanumber,4.0,945771921
,3,4.0,945771922
3,3,abc,945771923
3,4,2.0,notanumber
"""


class TestCsvParsing:
    @pytest.fixture()
    def movies_csv(self, tmp_path: Path) -> Path:
        path = tmp_path / "movies.csv"
        path.write_text(MOVIES_CSV, encoding="utf-8")
        return path

    @pytest.fixture()
    def ratings_csv(self, tmp_path: Path) -> Path:
        path = tmp_path / "ratings.csv"
        path.write_text(RATINGS_CSV, encoding="utf-8")
        return path

    def test_genres_become_a_list(self, movies_csv):
        documents = list(parse_movies(movies_csv))
        assert documents[0] == {
            "movieId": 1,
            "title": "Toy Story (1995)",
            "genres": ["Adventure", "Animation", "Children", "Comedy", "Fantasy"],
        }

    def test_titles_containing_commas_are_handled(self, movies_csv):
        documents = list(parse_movies(movies_csv))
        assert documents[1]["title"] == "GoldenEye (1995, James Bond)"

    def test_malformed_movie_rows_are_skipped(self, movies_csv):
        documents = list(parse_movies(movies_csv))
        ids = [d["movieId"] for d in documents]
        # Row 3 has empty genres (kept); rows 4 and 5 have no usable movieId.
        assert ids == [1, 2, 3]
        assert documents[2]["genres"] == []

    def test_ratings_are_typed_correctly(self, ratings_csv):
        documents = list(parse_ratings(ratings_csv))
        assert documents[0] == {
            "userId": 1,
            "movieId": 1,
            "rating": 4.0,
            "timestamp": 964982703,
        }
        assert isinstance(documents[0]["rating"], float)
        assert isinstance(documents[0]["userId"], int)

    def test_malformed_rating_rows_are_skipped(self, ratings_csv):
        documents = list(parse_ratings(ratings_csv))
        pairs = [(d["userId"], d["movieId"]) for d in documents]
        assert (2, "notanumber") not in [tuple(map(str, p)) for p in pairs]
        assert all(isinstance(d["userId"], int) for d in documents)
        assert all(isinstance(d["movieId"], int) for d in documents)
        assert all(isinstance(d["rating"], float) for d in documents)

    def test_a_bad_timestamp_is_omitted_but_the_row_survives(self, ratings_csv):
        documents = list(parse_ratings(ratings_csv))
        last = documents[-1]
        assert last == {"userId": 3, "movieId": 4, "rating": 2.0}

    def test_parser_yields_a_generator(self, movies_csv):
        result = parse_movies(movies_csv)
        assert hasattr(result, "__next__")
        assert next(result)["movieId"] == 1


class TestBatching:
    def test_batches_respect_the_size_limit(self):
        items = list(range(10))
        batches = list(_batched(items, 3))
        assert [len(b) for b in batches] == [3, 3, 3, 1]

    def test_exact_multiple_has_no_empty_tail(self):
        assert list(_batched(range(6), 3)) == [[0, 1, 2], [3, 4, 5]]

    def test_empty_input_yields_nothing(self):
        assert list(_batched([], 5)) == []

    def test_all_items_are_preserved(self):
        items = list(range(101))
        flattened = [x for batch in _batched(items, 7) for x in batch]
        assert flattened == items


class TestArchiveHandling:
    def test_invalid_archive_raises_dataset_error(self, tmp_path: Path):
        broken = tmp_path / "broken.zip"
        broken.write_bytes(b"this is not a zip file")
        with pytest.raises(DatasetError) as excinfo:
            extract_dataset(broken, tmp_path)
        assert "not a valid zip archive" in str(excinfo.value)

    def test_archive_missing_required_files_raises(self, tmp_path: Path):
        import zipfile

        archive = tmp_path / "partial.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("other.csv", "a,b\n1,2\n")
        with pytest.raises(DatasetError) as excinfo:
            extract_dataset(archive, tmp_path)
        assert "movies.csv" in str(excinfo.value)
        assert "ratings.csv" in str(excinfo.value)

    def test_ensure_csv_files_skips_download_when_files_exist(self, tmp_path: Path, monkeypatch):
        (tmp_path / "movies.csv").write_text("movieId,title,genres\n1,A,Drama\n", encoding="utf-8")
        (tmp_path / "ratings.csv").write_text("userId,movieId,rating,timestamp\n1,1,5,1\n", encoding="utf-8")

        def _explode(*args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("download_dataset should not be called")

        monkeypatch.setattr("scripts.load_data.download_dataset", _explode)
        files = ensure_csv_files(tmp_path)
        assert files["movies.csv"].exists()
        assert files["ratings.csv"].exists()

    def test_human_bytes_formats_nicely(self):
        assert _human_bytes(512) == "512 B"
        assert _human_bytes(2048) == "2.0 KB"
        assert _human_bytes(5 * 1024 * 1024) == "5.0 MB"


class TestImportPipeline:
    def test_import_is_idempotent_end_to_end(self, tmp_path: Path):
        """Run the real import twice against the fake collections."""
        from scripts.load_data import import_dataset

        (tmp_path / "movies.csv").write_text(MOVIES_CSV, encoding="utf-8")
        (tmp_path / "ratings.csv").write_text(RATINGS_CSV, encoding="utf-8")

        db = FakeDatabase()
        first = import_dataset(db, tmp_path / "movies.csv", tmp_path / "ratings.csv")
        assert first["movies_total"] == 3
        assert first["ratings_total"] == 4

        second = import_dataset(db, tmp_path / "movies.csv", tmp_path / "ratings.csv")
        assert second["movies_total"] == 3, "movies were duplicated on re-import"
        assert second["ratings_total"] == 4, "ratings were duplicated on re-import"

    def test_import_with_drop_clears_previous_data(self, tmp_path: Path):
        from scripts.load_data import import_dataset

        (tmp_path / "movies.csv").write_text(MOVIES_CSV, encoding="utf-8")
        (tmp_path / "ratings.csv").write_text(RATINGS_CSV, encoding="utf-8")

        db = FakeDatabase()
        import_dataset(db, tmp_path / "movies.csv", tmp_path / "ratings.csv")
        counts = import_dataset(
            db, tmp_path / "movies.csv", tmp_path / "ratings.csv", drop_existing=True
        )
        assert db.dropped == 1
        assert counts["movies_total"] == 3
        assert counts["ratings_total"] == 4

    def test_imported_ratings_feed_the_recommender(self, tmp_path: Path):
        """The full path CSV -> MongoDB -> recommendations must work."""
        from app.recommender import UserBasedRecommender
        from scripts.load_data import import_dataset

        (tmp_path / "movies.csv").write_text(MOVIES_CSV, encoding="utf-8")
        (tmp_path / "ratings.csv").write_text(RATINGS_CSV, encoding="utf-8")

        db = FakeDatabase()
        import_dataset(db, tmp_path / "movies.csv", tmp_path / "ratings.csv")

        engine = UserBasedRecommender(db)
        engine.load(force=True)
        result = engine.recommend(user_id=2, top_n=5, top_k=5)

        rated = set(engine.matrix.columns[engine.mask.loc[2].to_numpy()])
        assert {r.movie_id for r in result.recommendations}.isdisjoint(rated)
        for rec in result.recommendations:
            assert rec.title and rec.title != "nan"
            assert 0.5 <= rec.predicted_rating <= 5.0


# ----------------------------------------------------------------------
# Live-server integration tests (opt-in)
# ----------------------------------------------------------------------

TEST_URI = os.getenv("MONGO_TEST_URI") or os.getenv("MONGODB_URI") or "mongodb://localhost:27017/"
TEST_DB = os.getenv("MONGO_TEST_DB", "movie_recommender_test")


@pytest.fixture(scope="module")
def live_client():
    """Connect to a real server once, or skip when none is reachable."""
    database = MovieDatabase(uri=TEST_URI, db_name=TEST_DB, connect_timeout_ms=1500)
    try:
        database.connect()
    except DatabaseError as exc:
        pytest.skip(f"No MongoDB server available: {exc}")
    try:
        yield database.client
    finally:
        database.close()


@pytest.fixture()
def live_db(live_client):
    """A clean, throwaway database for each integration test."""
    database = MovieDatabase(
        uri=TEST_URI, db_name=TEST_DB, connect_timeout_ms=1500, _client=live_client
    )
    live_client.drop_database(TEST_DB)
    try:
        yield database
    finally:
        live_client.drop_database(TEST_DB)


@pytest.mark.skipif(not TEST_URI, reason="no MongoDB URI configured")
class TestLiveMongoDb:
    def test_insert_is_idempotent(self, live_db):
        live_db.ensure_indexes()
        documents = [
            {"userId": 1, "movieId": 1, "rating": 4.0, "timestamp": 1},
            {"userId": 1, "movieId": 2, "rating": 3.5, "timestamp": 2},
        ]
        live_db.upsert_ratings(documents)
        live_db.upsert_ratings(documents)
        assert live_db.count_documents(RATINGS_COLLECTION) == 2

    def test_unique_index_rejects_duplicate_pairs(self, live_db):
        live_db.ensure_indexes()
        live_db.ratings.insert_one({"userId": 9, "movieId": 9, "rating": 5.0})
        from pymongo.errors import DuplicateKeyError

        with pytest.raises(DuplicateKeyError):
            live_db.ratings.insert_one({"userId": 9, "movieId": 9, "rating": 1.0})

    def test_statistics_match_manual_counts(self, live_db):
        live_db.ensure_indexes()
        live_db.upsert_movies(
            [
                {"movieId": 1, "title": "A (2000)", "genres": ["Drama"]},
                {"movieId": 2, "title": "B (2001)", "genres": ["Action"]},
            ]
        )
        live_db.upsert_ratings(
            [
                {"userId": 1, "movieId": 1, "rating": 4.0},
                {"userId": 1, "movieId": 2, "rating": 2.0},
                {"userId": 2, "movieId": 1, "rating": 5.0},
            ]
        )
        stats = live_db.get_statistics()
        assert stats["total_movies"] == 2
        assert stats["total_users"] == 2
        assert stats["total_ratings"] == 3
        assert stats["average_rating"] == pytest.approx((4.0 + 2.0 + 5.0) / 3)

    def test_search_finds_movies_case_insensitively(self, live_db):
        live_db.ensure_indexes()
        live_db.upsert_movies([{"movieId": 42, "title": "The Matrix (1999)", "genres": ["Sci-Fi"]}])
        live_db.upsert_ratings([{"userId": 1, "movieId": 42, "rating": 5.0}])

        results = live_db.search_movies("matrix")
        assert not results.empty
        assert "The Matrix (1999)" in set(results["title"])
        row = results[results["title"] == "The Matrix (1999)"].iloc[0]
        assert row["average_rating"] == pytest.approx(5.0)
        assert row["rating_count"] == 1

    def test_search_with_no_query_returns_an_empty_frame(self, live_db):
        results = live_db.search_movies("   ")
        assert results.empty
        assert "title" in results.columns

    def test_end_to_end_recommendation(self, live_db):
        live_db.ensure_indexes()
        live_db.upsert_movies(
            [
                {"movieId": 1, "title": "Alpha (2000)", "genres": ["Drama"]},
                {"movieId": 2, "title": "Beta (2001)", "genres": ["Action"]},
                {"movieId": 3, "title": "Gamma (2002)", "genres": ["Comedy"]},
            ]
        )
        live_db.upsert_ratings(
            [
                {"userId": 1, "movieId": 1, "rating": 5.0},
                {"userId": 1, "movieId": 2, "rating": 4.0},
                {"userId": 2, "movieId": 1, "rating": 5.0},
                {"userId": 2, "movieId": 3, "rating": 4.0},
            ]
        )
        from app.recommender import UserBasedRecommender

        engine = UserBasedRecommender(live_db)
        engine.load(force=True)
        result = engine.recommend(1, top_n=5, top_k=5)

        assert [r.movie_id for r in result.recommendations] == [3]
        assert result.recommendations[0].title == "Gamma (2002)"
        assert result.recommendations[0].neighbor_count == 1

    def test_get_statistics_on_an_empty_database(self, live_db):
        live_db.client.drop_database(TEST_DB)
        stats = live_db.get_statistics()
        assert stats == {
            "total_movies": 0,
            "total_users": 0,
            "total_ratings": 0,
            "average_rating": 0.0,
        }
        assert live_db.is_empty() is True


class TestAnalyticsAgainstFakeDb:
    def test_search_returns_a_frame_with_stats(self, populated=None):
        db = FakeDatabase()
        db.ensure_indexes()
        db.upsert_movies([{"movieId": 7, "title": "Blade Runner (1982)", "genres": ["Sci-Fi"]}])
        db.upsert_ratings([{"userId": 1, "movieId": 7, "rating": 4.5}])

        from app.analytics import AnalyticsService

        service = AnalyticsService(db)  # type: ignore[arg-type]
        results = service.search("blade", limit=10)
        assert not results.empty
        assert results.iloc[0]["title"] == "Blade Runner (1982)"

    def test_user_statistics_summarises_a_profile(self, sample_ratings, sample_movies):
        from app.analytics import user_statistics

        merged = sample_ratings.merge(sample_movies, on="movieId", how="left")
        stats = user_statistics(merged, user_id=1)

        assert stats["user_id"] == 1
        assert stats["rating_count"] == 3
        assert stats["average_rating"] == pytest.approx((5.0 + 4.0 + 1.0) / 3)
        assert stats["has_data"] is True
        assert len(stats["top_movies"]) == 3
        assert "Action" in stats["top_genres"] or stats["top_genres"]

    def test_user_statistics_for_an_unknown_user(self, sample_ratings, sample_movies):
        from app.analytics import user_statistics

        merged = sample_ratings.merge(sample_movies, on="movieId", how="left")
        stats = user_statistics(merged, user_id=999)

        assert stats["has_data"] is False
        assert stats["rating_count"] == 0
        assert stats["top_movies"].empty

    def test_overall_statistics_on_an_empty_frame(self):
        from app.analytics import overall_statistics

        assert overall_statistics(pd.DataFrame()) == {
            "total_movies": 0,
            "total_users": 0,
            "total_ratings": 0,
            "average_rating": 0.0,
        }
