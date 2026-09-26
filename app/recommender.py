"""User-based collaborative filtering using cosine similarity.

The pipeline is:

1. Fetch ``(userId, movieId, rating)`` documents from MongoDB.
2. Build a user x movie matrix.  Missing ratings are stored as ``0`` because
   the cosine formula divides by the vector norm, but a companion boolean
   matrix records *where* a real rating exists so an unrated movie is never
   mistaken for a 0-star rating.
3. Compute the full user-user cosine similarity matrix once with
   :func:`sklearn.metrics.pairwise.cosine_similarity`.
4. For a target user, rank all other users by similarity, drop zero-similarity
   rows and keep the Top-K neighbours.
5. For every movie the target user has *not* rated, compute the
   similarity-weighted average rating over the neighbours that *did* rate it,
   then return the best N.

Nothing in this module reads the CSV files: the frames always originate from
:mod:`app.database`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from app.database import DatabaseError, MovieDatabase

LOGGER = logging.getLogger("movie_recommender.recommender")

RATING_COLUMNS = ("userId", "movieId", "rating")
MIN_RATINGS_FOR_USER = 1


class InsufficientDataError(ValueError):
    """Raised when the ratings frame cannot support recommendations."""


@dataclass
class Neighbor:
    """A similar user together with their cosine similarity to the target."""

    user_id: int
    similarity: float
    rating_count: int


@dataclass(frozen=True)
class Recommendation:
    """A single movie recommendation for a target user."""

    movie_id: int
    title: str
    genres: tuple[str, ...]
    predicted_rating: float
    neighbor_count: int
    #: Similarity-weighted strength of evidence (sum of contributing sims).
    similarity_mass: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return a StreamLog/Plotly friendly representation."""
        return {
            "movieId": self.movie_id,
            "title": self.title,
            "genres": list(self.genres),
            "genres_display": ", ".join(self.genres) if self.genres else "Uncategorized",
            "predicted_rating": round(self.predicted_rating, 3),
            "neighbor_count": self.neighbor_count,
            "similarity_mass": round(self.similarity_mass, 4),
        }


@dataclass
class SimilarUserResult:
    """Bundle returned by :meth:`UserBasedRecommender.recommend`."""

    user_id: int
    neighbors: list[Neighbor] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    message: str = ""
    candidates_considered: int = 0

    @property
    def has_recommendations(self) -> bool:
        return bool(self.recommendations)


# ----------------------------------------------------------------------
# Pure functions (unit-testable without a database)
# ----------------------------------------------------------------------
def build_user_movie_matrix(
    ratings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the numeric rating matrix and the boolean "has rating" mask.

    Parameters
    ----------
    ratings:
        DataFrame with the columns ``userId``, ``movieId`` and ``rating``.

    Returns
    -------
    (matrix, mask):
        ``matrix`` is a dense ``users x movies`` DataFrame of float ratings
        where unrated entries are ``0``; ``mask`` has the same shape with
        ``True`` where a genuine rating exists.

    Raises
    ------
    InsufficientDataError
        If the required columns are missing or the frame is empty.
    """
    _validate_ratings_frame(ratings)

    if ratings.duplicated(subset=["userId", "movieId"]).any():
        LOGGER.warning(
            "Duplicate (userId, movieId) pairs found; keeping the last rating."
        )
        ratings = ratings.drop_duplicates(subset=["userId", "movieId"], keep="last")

    matrix = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        aggfunc="last",
        fill_value=0.0,
    )
    matrix = matrix.astype(float)
    # Sorted, unique axes make downstream positional indexing deterministic.
    matrix = matrix.reindex(
        index=sorted(matrix.index.tolist()),
        columns=sorted(matrix.columns.tolist()),
        fill_value=0.0,
    )
    matrix.index.name = "userId"
    matrix.columns.name = "movieId"

    mask = matrix.ne(0.0)
    return matrix, mask


def _validate_ratings_frame(ratings: pd.DataFrame) -> None:
    if not isinstance(ratings, pd.DataFrame):
        raise InsufficientDataError("ratings must be a pandas DataFrame")
    missing = [c for c in RATING_COLUMNS if c not in ratings.columns]
    if missing:
        raise InsufficientDataError(
            f"ratings is missing required column(s): {', '.join(missing)}"
        )
    if ratings.empty:
        raise InsufficientDataError(
            "ratings is empty - load the MovieLens data into MongoDB first "
            "(python -m scripts.load_data)."
        )


def calculate_similarity_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute the user-user cosine similarity matrix.

    ``cosine_similarity(A, B) = dot(A, B) / (norm(A) * norm(B))``

    Users whose rating vector is all zeros have a norm of 0, which yields a
    division by zero; those rows are forced to 0 instead of ``nan``.
    """
    values = matrix.to_numpy(dtype=float)
    if values.size == 0:
        return pd.DataFrame(index=matrix.index, columns=matrix.index, dtype=float)

    similarity = cosine_similarity(values)
    similarity = np.nan_to_num(similarity, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(similarity, 0.0)  # a user is never their own neighbour

    return pd.DataFrame(
        similarity, index=matrix.index, columns=matrix.index, dtype=float
    )


def get_similar_users(
    similarity: pd.DataFrame,
    user_id: int,
    top_k: int = 10,
    min_similarity: float = 1e-9,
) -> list[Neighbor]:
    """Return the Top-K most similar users for ``user_id``.

    The target user is always excluded, zero-similarity users are dropped and
    ties are broken by the smaller user id for deterministic output.
    """
    if user_id not in similarity.index:
        raise InsufficientDataError(f"User {user_id} is not present in the matrix.")

    k = int(max(1, top_k))
    row = similarity.loc[user_id].drop(index=user_id, errors="ignore")
    candidates = [
        Neighbor(user_id=int(other), similarity=float(score),
                 rating_count=0)
        for other, score in row.items()
        if float(score) > min_similarity
    ]
    candidates.sort(key=lambda n: (-n.similarity, n.user_id))
    return candidates[:k]


def weighted_predicted_rating(
    similarities: Sequence[float],
    ratings: Sequence[float],
) -> float:
    """Return the similarity-weighted mean of ``ratings``.

    ``Predicted = sum(similarity * rating) / sum(similarity)``.  Returns
    ``0.0`` when the total similarity mass is zero (no evidence).
    """
    sims = np.asarray(list(similarities), dtype=float)
    vals = np.asarray(list(ratings), dtype=float)
    if sims.size == 0 or sims.size != vals.size:
        return 0.0
    total = float(sims.sum())
    if total <= 0.0:
        return 0.0
    return float(np.dot(sims, vals) / total)


def _weighted_predictions(
    mask: np.ndarray,
    matrix: np.ndarray,
    neighbor_rows: np.ndarray,
    neighbor_sims: np.ndarray,
    target_row: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised similarity-weighted prediction for every candidate movie.

    Returns ``(predictions, neighbor_counts, similarity_mass)`` arrays aligned
    with the *columns* of the matrix.
    """
    n_movies = matrix.shape[1]
    predictions = np.zeros(n_movies, dtype=float)
    counts = np.zeros(n_movies, dtype=int)
    mass = np.zeros(n_movies, dtype=float)

    if neighbor_rows.size == 0:
        return predictions, counts, mass

    # Only movies the target user has actually rated are excluded downstream;
    # here we mark the already-rated ones with NaN to filter them out later.
    neighbor_ratings = matrix[neighbor_rows, :]          # (n_neighbors, n_movies)
    neighbor_mask = mask[neighbor_rows, :]                # (n_neighbors, n_movies)

    sim_matrix = np.outer(neighbor_sims, np.ones(n_movies))
    numerator = (neighbor_ratings * sim_matrix).sum(axis=0)
    denominator = (neighbor_mask * sim_matrix).sum(axis=0)
    counts_arr = neighbor_mask.sum(axis=0).astype(int)

    valid = denominator > 0
    predictions[valid] = numerator[valid] / denominator[valid]
    counts[:] = counts_arr
    mass[:] = denominator

    return predictions, counts, mass


def recommend_for_user(
    matrix: pd.DataFrame,
    mask: pd.DataFrame,
    similarity: pd.DataFrame,
    user_id: int,
    top_n: int = 10,
    top_k: int = 10,
    movie_titles: Mapping[int, str] | None = None,
    movie_genres: Mapping[int, Sequence[str]] | None = None,
    min_similarity: float = 1e-9,
    min_neighbors: int = 1,
) -> SimilarUserResult:
    """Generate Top-N movie recommendations for ``user_id``.

    Notes
    -----
    * Movies already rated by ``user_id`` are removed **before** ranking.
    * Only movies rated by at least ``min_neighbors`` neighbours are eligible,
      which filters out items supported by a single, weakly related user.
    * Ranking is by predicted rating, then by the number of contributing
      neighbours, then by descending similarity mass for stability.
    """
    result = SimilarUserResult(user_id=int(user_id))
    n = int(max(1, top_n))
    k = int(max(1, top_k))

    if user_id not in matrix.index:
        result.message = (
            f"User {user_id} has no ratings in the database, so no personal "
            "recommendations can be generated."
        )
        return result

    rated_by_user = int(mask.loc[user_id].sum())
    if rated_by_user < MIN_RATINGS_FOR_USER:
        result.message = f"User {user_id} has no ratings to base a profile on."
        return result

    neighbors = get_similar_users(similarity, user_id, top_k=k, min_similarity=min_similarity)
    if not neighbors:
        result.message = (
            f"No other user is similar enough to User {user_id}. Try lowering "
            "the similarity threshold or rating more movies."
        )
        return result

    neighbor_ids = [nb.user_id for nb in neighbors]
    rating_counts = mask.sum(axis=1)
    for nb in neighbors:
        nb.rating_count = int(rating_counts.get(nb.user_id, 0))
    result.neighbors = neighbors

    available = [uid for uid in neighbor_ids if uid in matrix.index]
    if not available:
        result.message = "Similar users have no ratings available in the matrix."
        return result

    neighbor_rows = np.array(
        [matrix.index.get_loc(uid) for uid in available], dtype=int
    )
    neighbor_sims = np.array(
        [similarity.loc[user_id, uid] for uid in available], dtype=float
    )

    target_row = matrix.index.get_loc(user_id)
    predictions, counts, mass = _weighted_predictions(
        mask.to_numpy(), matrix.to_numpy(), neighbor_rows, neighbor_sims, target_row
    )

    # --- Exclude everything the target user has already rated -------------
    already_rated = mask.iloc[target_row].to_numpy()
    eligible = (
        ~already_rated
        & (predictions > 0.0)
        & (counts >= max(1, int(min_neighbors)))
    )
    candidate_indices = np.flatnonzero(eligible)
    result.candidates_considered = int(candidate_indices.size)

    if candidate_indices.size == 0:
        result.message = (
            "Every movie your neighbours rated has already been seen by you. "
            "Try increasing the number of similar users."
        )
        return result

    # --- Rank: predicted rating desc, neighbour count desc, mass desc ------
    order = np.lexsort(
        (
            -mass[candidate_indices],          # 3rd key
            -counts[candidate_indices],        # 2nd key (reliability)
            -predictions[candidate_indices],   # 1st key (primary)
        )
    )
    ranked = candidate_indices[order][:n]

    titles = movie_titles or {}
    genres = movie_genres or {}
    movie_ids = list(matrix.columns)

    for idx in ranked:
        movie_id = int(movie_ids[idx])
        result.recommendations.append(
            Recommendation(
                movie_id=movie_id,
                title=str(titles.get(movie_id, f"Movie {movie_id}")),
                genres=tuple(genres.get(movie_id, ())),
                predicted_rating=float(predictions[idx]),
                neighbor_count=int(counts[idx]),
                similarity_mass=float(mass[idx]),
            )
        )

    if not result.message:
        result.message = (
            f"Found {result.candidates_considered} candidate movie(s) from "
            f"{len(neighbors)} similar user(s)."
        )
    return result


# ----------------------------------------------------------------------
# Database-backed orchestrator
# ----------------------------------------------------------------------
class UserBasedRecommender:
    """Loads data from MongoDB once and serves recommendations from memory.

    The similarity matrix is computed lazily and cached, so the (610x610)
    MovieLens matrix is only calculated on the first request.
    """

    def __init__(
        self,
        db: MovieDatabase,
        top_n: int = 10,
        top_k: int = 10,
        min_similarity: float = 1e-9,
        min_neighbors: int = 1,
    ) -> None:
        self.db = db
        self.top_n = int(top_n)
        self.top_k = int(top_k)
        self.min_similarity = float(min_similarity)
        self.min_neighbors = int(min_neighbors)

        self._matrix: pd.DataFrame | None = None
        self._mask: pd.DataFrame | None = None
        self._similarity: pd.DataFrame | None = None
        self._movies: pd.DataFrame | None = None
        self._titles: dict[int, str] = {}
        self._genres: dict[int, tuple[str, ...]] = {}

    # -- data loading ---------------------------------------------------
    def load(self, force: bool = False) -> None:
        """(Re)load ratings and movies from MongoDB into memory."""
        if self._matrix is not None and not force:
            return
        try:
            ratings = self.db.get_all_ratings()
            movies = self.db.get_all_movies()
        except DatabaseError:
            raise
        self.set_data(ratings, movies)

    def set_data(self, ratings: pd.DataFrame, movies: pd.DataFrame) -> None:
        """Inject data directly (used by tests and by :meth:`load`)."""
        self._matrix, self._mask = build_user_movie_matrix(ratings)
        self._similarity = None  # invalidate the cached similarity matrix
        self._movies = movies
        if movies is not None and not movies.empty:
            self._titles = {
                int(row.movieId): str(row.title)
                for row in movies.itertuples()
            }
            self._genres = {
                int(row.movieId): tuple(row.genres or ())
                for row in movies.itertuples()
            }

    @property
    def matrix(self) -> pd.DataFrame:
        self.load()
        assert self._matrix is not None
        return self._matrix

    @property
    def mask(self) -> pd.DataFrame:
        self.load()
        assert self._mask is not None
        return self._mask

    @property
    def similarity(self) -> pd.DataFrame:
        """Cosine similarity matrix, computed once and cached."""
        if self._similarity is None:
            LOGGER.info("Computing user-user cosine similarity matrix ...")
            self._similarity = calculate_similarity_matrix(self.matrix)
            LOGGER.info(
                "Similarity matrix ready: %s users", self._similarity.shape[0]
            )
        return self._similarity

    @property
    def user_ids(self) -> list[int]:
        """Sorted list of user ids present in the matrix."""
        return [int(uid) for uid in self.matrix.index.tolist()]

    @property
    def movie_ids(self) -> list[int]:
        """Sorted list of movie ids present in the matrix columns."""
        return [int(mid) for mid in self.matrix.columns.tolist()]

    def is_ready(self) -> bool:
        """True when data is loaded and non-empty."""
        return self._matrix is not None and not self._matrix.empty

    # -- recommendation -------------------------------------------------
    def get_similar_users(
        self, user_id: int, top_k: int | None = None
    ) -> list[Neighbor]:
        """Top-K neighbours for ``user_id`` (excluding the user itself).

        The rating count of each neighbour is filled in from the matrix so the
        UI can show how much evidence each neighbour brings.
        """
        neighbors = get_similar_users(
            self.similarity, int(user_id), top_k=top_k or self.top_k
        )
        counts = self.mask.sum(axis=1)
        for neighbor in neighbors:
            neighbor.rating_count = int(counts.get(neighbor.user_id, 0))
        return neighbors

    def recommend(
        self,
        user_id: int,
        top_n: int | None = None,
        top_k: int | None = None,
    ) -> SimilarUserResult:
        """Return the Top-N recommendations for ``user_id``."""
        return recommend_for_user(
            self._require_matrix(),
            self.mask,
            self.similarity,
            int(user_id),
            top_n=top_n or self.top_n,
            top_k=top_k or self.top_k,
            movie_titles=self._titles,
            movie_genres=self._genres,
            min_similarity=self.min_similarity,
            min_neighbors=self.min_neighbors,
        )

    def recommendations_frame(
        self,
        user_id: int,
        top_n: int | None = None,
        top_k: int | None = None,
    ) -> pd.DataFrame:
        """Recommendations as a DataFrame (empty when there are none)."""
        columns = [
            "movieId",
            "title",
            "genres_display",
            "predicted_rating",
            "neighbor_count",
        ]
        result = self.recommend(user_id, top_n=top_n, top_k=top_k)
        if not result.recommendations:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame([r.as_dict() for r in result.recommendations])[columns]

    def _require_matrix(self) -> pd.DataFrame:
        self.load()
        assert self._matrix is not None
        if self._matrix.empty:
            raise InsufficientDataError(
                "The user-movie matrix is empty. Load the dataset with "
                "'python -m scripts.load_data'."
            )
        return self._matrix


__all__ = [
    "InsufficientDataError",
    "Neighbor",
    "Recommendation",
    "SimilarUserResult",
    "UserBasedRecommender",
    "build_user_movie_matrix",
    "calculate_similarity_matrix",
    "get_similar_users",
    "recommend_for_user",
    "weighted_predicted_rating",
]
