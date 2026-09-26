"""Analytics helpers: user statistics, genre analysis and distributions.

Every function takes already-loaded DataFrames (sourced from MongoDB) or a
:class:`~app.database.MovieDatabase` and returns data ready for rendering in
Streamlit / Plotly.  No raw traceback ever reaches the UI: unrecoverable
conditions raise :class:`AnalyticsError` with a human readable message.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from app.database import DatabaseError, MovieDatabase

LOGGER = logging.getLogger("movie_recommender.analytics")

#: A movie is "liked" for genre analysis at or above this rating.
HIGH_RATING_THRESHOLD = 4.0
#: Star colour scale used consistently across every chart.
RATING_COLOR_SCALE = [
    [0.0, "#2b2d42"],
    [0.25, "#3d5a80"],
    [0.5, "#4f8ef7"],
    [0.75, "#98c1d9"],
    [1.0, "#ffd166"],
]
ACCENT = "#4f8ef7"
ACCENT_2 = "#ffd166"
TEMPLATE = "plotly_dark"


class AnalyticsError(RuntimeError):
    """Raised when analytics cannot be produced from the available data."""


# ----------------------------------------------------------------------
# Basic statistics
# ----------------------------------------------------------------------
def overall_statistics(
    ratings: pd.DataFrame,
    movie_count: int | None = None,
) -> dict[str, Any]:
    """Compute the dashboard statistics from a ratings frame.

    Parameters
    ----------
    ratings:
        Frame with ``userId``, ``movieId``, ``rating``.
    movie_count:
        Number of distinct movies; derived from ``ratings`` when omitted.
    """
    if ratings is None or ratings.empty:
        return {
            "total_movies": int(movie_count or 0),
            "total_users": 0,
            "total_ratings": 0,
            "average_rating": 0.0,
        }
    return {
        "total_movies": int(
            movie_count
            if movie_count is not None
            else ratings["movieId"].nunique()
        ),
        "total_users": int(ratings["userId"].nunique()),
        "total_ratings": int(len(ratings)),
        "average_rating": float(ratings["rating"].mean()),
    }


def rating_distribution(ratings: pd.DataFrame) -> pd.DataFrame:
    """Return the count of ratings per star value, ordered by rating."""
    if ratings is None or ratings.empty:
        return pd.DataFrame(columns=["rating", "count"])
    counts = ratings["rating"].value_counts().sort_index()
    return pd.DataFrame(
        {"rating": counts.index.astype(float), "count": counts.to_numpy()}
    ).reset_index(drop=True)


# ----------------------------------------------------------------------
# User level analytics
# ----------------------------------------------------------------------
def user_statistics(ratings: pd.DataFrame, user_id: int) -> dict[str, Any]:
    """Return profile statistics for a single user.

    Keys
    ----
    user_id, rating_count, average_rating, min_rating, max_rating,
    std_rating, top_movies (DataFrame), recently_rated (DataFrame),
    rating_distribution (DataFrame), genre_counts (DataFrame),
    top_genres (list[str]), has_data (bool)
    """
    empty_cols = ["movieId", "title", "genres", "rating", "timestamp"]
    if ratings is None or ratings.empty:
        return {
            "user_id": int(user_id),
            "rating_count": 0,
            "average_rating": 0.0,
            "min_rating": 0.0,
            "max_rating": 0.0,
            "std_rating": 0.0,
            "top_movies": pd.DataFrame(columns=empty_cols),
            "recently_rated": pd.DataFrame(columns=empty_cols),
            "rating_distribution": pd.DataFrame(columns=["rating", "count"]),
            "genre_counts": pd.DataFrame(columns=["genre", "count", "average_rating"]),
            "top_genres": [],
            "has_data": False,
        }

    user_rows = ratings[ratings["userId"] == int(user_id)].copy()
    if user_rows.empty:
        return user_statistics(pd.DataFrame(columns=ratings.columns), user_id)

    if "title" not in user_rows.columns:
        user_rows["title"] = user_rows["movieId"].map(lambda m: f"Movie {m}")
    if "genres" not in user_rows.columns:
        user_rows["genres"] = [[] for _ in range(len(user_rows))]

    sorted_rows = user_rows.sort_values(
        ["rating", "title"], ascending=[False, True]
    )
    genre_counts = genre_preferences(user_rows)

    return {
        "user_id": int(user_id),
        "rating_count": int(len(user_rows)),
        "average_rating": float(user_rows["rating"].mean()),
        "min_rating": float(user_rows["rating"].min()),
        "max_rating": float(user_rows["rating"].max()),
        "std_rating": float(user_rows["rating"].std() or 0.0),
        "top_movies": sorted_rows.head(10).reset_index(drop=True),
        "recently_rated": sorted_rows.head(10).reset_index(drop=True),
        "rating_distribution": rating_distribution(user_rows),
        "genre_counts": genre_counts,
        "top_genres": genre_counts["genre"].head(5).tolist()
        if not genre_counts.empty
        else [],
        "has_data": True,
    }


# ----------------------------------------------------------------------
# Genre analytics
# ----------------------------------------------------------------------
def _explode_genres(movies_with_ratings: pd.DataFrame) -> pd.DataFrame:
    """Explode the list-valued ``genres`` column into one row per genre."""
    if movies_with_ratings.empty or "genres" not in movies_with_ratings.columns:
        return pd.DataFrame(columns=["movieId", "genre", "rating"])
    working = movies_with_ratings[["movieId", "genres", "rating"]].copy()
    working["genres"] = working["genres"].apply(
        lambda g: list(g) if isinstance(g, (list, tuple, np.ndarray)) else []
    )
    exploded = working.explode("genres", ignore_index=True)
    exploded = exploded[exploded["genres"].notna() & (exploded["genres"] != "")]
    return exploded.rename(columns={"genres": "genre"})


def genre_preferences(
    movies_with_ratings: pd.DataFrame,
    threshold: float | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Count genres across a user's rated movies and compute their mean rating.

    Parameters
    ----------
    movies_with_ratings:
        Frame with ``movieId``, ``genres`` (list) and ``rating``.
    threshold:
        When given, only movies rated at or above this value are counted.
    limit:
        Optionally truncate to the top ``limit`` genres.
    """
    exploded = _explode_genres(movies_with_ratings)
    if exploded.empty:
        return pd.DataFrame(columns=["genre", "count", "average_rating"])
    if threshold is not None:
        exploded = exploded[exploded["rating"] >= threshold]
    if exploded.empty:
        return pd.DataFrame(columns=["genre", "count", "average_rating"])

    grouped = (
        exploded.groupby("genre")
        .agg(count=("movieId", "nunique"), average_rating=("rating", "mean"))
        .reset_index()
    )
    grouped["average_rating"] = grouped["average_rating"].round(2)
    grouped = grouped.sort_values(
        ["count", "average_rating"], ascending=[False, False]
    ).reset_index(drop=True)
    if limit:
        grouped = grouped.head(limit)
    return grouped


def preferred_genres(
    movies_with_ratings: pd.DataFrame,
    threshold: float = HIGH_RATING_THRESHOLD,
    top_n: int = 5,
) -> list[tuple[str, int]]:
    """Return ``(genre, movie_count)`` for the user's highest rated movies."""
    frame = genre_preferences(movies_with_ratings, threshold=threshold)
    if frame.empty:
        return []
    return [
        (str(row.genre), int(row.count))
        for row in frame.head(top_n).itertuples()
    ]


def global_genre_ranking(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    limit: int = 15,
) -> pd.DataFrame:
    """Most-rated genres across the whole dataset.

    ``ratings`` may already carry the ``title``/``genres`` columns (as returned by
    :meth:`AnalyticsService.load`); in that case the join is skipped so a
    pre-merged frame is never joined against itself.
    """
    if ratings is None or movies is None or ratings.empty or movies.empty:
        return pd.DataFrame(columns=["genre", "count", "average_rating"])
    if "genres" in ratings.columns:
        merged = ratings
    else:
        merged = ratings.merge(
            movies[["movieId", "title", "genres"]], on="movieId", how="left"
        )
    return genre_preferences(merged).head(limit)


# ----------------------------------------------------------------------
# Plotly figure builders
# ----------------------------------------------------------------------
def _base_layout(title: str, **kwargs: Any) -> dict[str, Any]:
    layout = {
        "title": {"text": title, "font": {"size": 17}},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#e8e8ef", "family": "Segoe UI, Inter, sans-serif"},
        "margin": {"l": 20, "r": 20, "t": 55, "b": 20},
        "hoverlabel": {"bgcolor": "#1a1b26", "font": {"color": "#ffffff"}},
    }
    layout.update(kwargs)
    return layout


def rating_distribution_figure(
    distribution: pd.DataFrame,
    title: str = "Rating Distribution (All Users)",
    color: str = ACCENT,
) -> Any:
    """Bar chart of how many ratings each star value received."""
    import plotly.graph_objects as go

    if distribution is None or distribution.empty:
        return _empty_figure(title, "No rating data available")

    labels = [f"{r:g} ★" for r in distribution["rating"]]
    colors = [ACCENT_2 if r >= HIGH_RATING_THRESHOLD else color for r in distribution["rating"]]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=distribution["count"],
            marker_color=colors,
            marker_line_color="rgba(255,255,255,0.15)",
            marker_line_width=1,
            hovertemplate="%{x}<br>Ratings: %{y:,}<extra></extra>",
            width=0.62,
        )
    )
    fig.update_layout(**_base_layout(title, showlegend=False))
    fig.update_xaxes(title="Rating", gridcolor="rgba(255,255,255,0.07)")
    fig.update_yaxes(title="Number of Ratings", gridcolor="rgba(255,255,255,0.07)")
    return fig


def user_rating_distribution_figure(
    stats: dict[str, Any], title: str = "Your Rating Distribution"
) -> Any:
    """Bar chart comparing the selected user's distribution to the dataset."""
    import plotly.graph_objects as go

    user_dist = stats.get("rating_distribution")
    if user_dist is None or user_dist.empty:
        return _empty_figure(title, "This user has not rated any movies yet")

    labels = [f"{r:g} ★" for r in user_dist["rating"]]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=user_dist["count"],
            marker=dict(
                color=user_dist["rating"],
                colorscale=RATING_COLOR_SCALE,
                line=dict(color="rgba(255,255,255,0.2)", width=1),
            ),
            name="Selected user",
            hovertemplate="%{x}<br>Ratings: %{y}<extra></extra>",
        )
    )
    avg = stats.get("average_rating", 0.0)
    fig.add_vline(
        x=len(labels) / 2 - 0.5,
        line=dict(color="rgba(255,255,255,0.18)", dash="dot"),
    )
    fig.update_layout(**_base_layout(f"{title} · avg {avg:.2f} ★", showlegend=False))
    fig.update_xaxes(title="Rating", gridcolor="rgba(255,255,255,0.07)")
    fig.update_yaxes(title="Movies Rated", gridcolor="rgba(255,255,255,0.07)")
    return fig


def user_genre_figure(
    stats: dict[str, Any],
    threshold: float = HIGH_RATING_THRESHOLD,
    limit: int = 10,
) -> Any:
    """Horizontal bar chart of the genres the user rates most highly."""
    import plotly.graph_objects as go

    frame = stats.get("genre_counts")
    if frame is None or frame.empty:
        return _empty_figure("Top Genres You Love", "Not enough rated movies yet")

    liked = frame[frame["average_rating"] >= threshold].head(limit)
    if liked.empty:
        liked = frame.head(limit)

    liked = liked.sort_values("count", ascending=True)
    colors = [
        ACCENT_2 if row.average_rating >= threshold else ACCENT
        for row in liked.itertuples()
    ]
    fig = go.Figure(
        go.Bar(
            x=liked["count"],
            y=liked["genre"],
            orientation="h",
            marker_color=colors,
            hovertemplate=(
                "<b>%{y}</b><br>Movies rated ≥ "
                f"{threshold:g}★: %{{x}}<br>Avg rating: %{{customdata:.2f}} ★<extra></extra>"
            ),
            customdata=liked["average_rating"],
        )
    )
    subtitle = (
        f"genres from movies rated {threshold:g}★ and above"
        if not frame[frame["average_rating"] >= threshold].empty
        else "genres from all rated movies"
    )
    fig.update_layout(**_base_layout(f"Top Genres You Love · {subtitle}", showlegend=False))
    fig.update_xaxes(title="Movies", gridcolor="rgba(255,255,255,0.07)")
    fig.update_yaxes(title="", gridcolor="rgba(255,255,255,0.05)")
    return fig


def similarity_figure(neighbors: Sequence[Any], top: int = 10) -> Any:
    """Bar chart of the cosine similarity of the Top-10 similar users."""
    import plotly.graph_objects as go

    if not neighbors:
        return _empty_figure("Top 10 Similar Users", "No similar users found")

    top_neighbors = list(neighbors)[:top]
    labels = [f"User {n.user_id}" for n in top_neighbors]
    scores = [round(float(n.similarity), 4) for n in top_neighbors]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=scores,
            marker=dict(
                color=scores,
                colorscale=[[0.0, "#3d5a80"], [1.0, "#ffd166"]],
                line=dict(color="rgba(255,255,255,0.2)", width=1),
            ),
            hovertemplate=(
                "<b>%{x}</b><br>Cosine similarity: %{y:.4f}"
                "<br>Movies rated: %{customdata}<extra></extra>"
            ),
            customdata=[int(n.rating_count) for n in top_neighbors],
            width=0.6,
        )
    )
    fig.update_layout(**_base_layout("Top 10 Similar Users · Cosine Similarity", showlegend=False))
    fig.update_xaxes(title="User", tickangle=-45, gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(
        title="Cosine similarity", range=[0, 1.05], gridcolor="rgba(255,255,255,0.07)"
    )
    return fig


def similarity_scatter_figure(
    neighbors: Sequence[Any],
    target_rating_count: int,
    title: str = "Similarity vs. Activity",
) -> Any:
    """Scatter plot of neighbour similarity against the number of movies rated."""
    import plotly.graph_objects as go

    if not neighbors:
        return _empty_figure(title, "No similar users found")

    fig = go.Figure(
        go.Scatter(
            x=[int(n.rating_count) for n in neighbors],
            y=[float(n.similarity) for n in neighbors],
            mode="markers+text",
            text=[f"U{n.user_id}" for n in neighbors],
            textposition="top center",
            textfont=dict(size=9, color="#9aa0c0"),
            marker=dict(
                size=[10 + 60 * float(n.similarity) for n in neighbors],
                color=[float(n.similarity) for n in neighbors],
                colorscale=[[0.0, "#3d5a80"], [1.0, "#ffd166"]],
                line=dict(color="rgba(255,255,255,0.3)", width=1),
                showscale=False,
            ),
            hovertemplate=(
                "<b>User %{text}</b><br>Movies rated: %{x}"
                "<br>Cosine similarity: %{y:.4f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        **_base_layout(
            f"{title} · you rated {target_rating_count} movies", showlegend=False
        )
    )
    fig.update_xaxes(title="Movies Rated by Neighbour", gridcolor="rgba(255,255,255,0.07)")
    fig.update_yaxes(
        title="Cosine Similarity", range=[0, 1.05], gridcolor="rgba(255,255,255,0.07)"
    )
    return fig


def genre_bar_figure(
    frame: pd.DataFrame, title: str = "Most Rated Genres", limit: int = 12
) -> Any:
    """Horizontal bar chart for a pre-computed genre ranking frame."""
    import plotly.graph_objects as go

    if frame is None or frame.empty:
        return _empty_figure(title, "No genre data available")

    data = frame.head(limit).sort_values("count")
    fig = go.Figure(
        go.Bar(
            x=data["count"],
            y=data["genre"],
            orientation="h",
            marker_color=ACCENT,
            hovertemplate=(
                "<b>%{y}</b><br>Movies rated: %{x}<br>Avg rating: %{customdata:.2f} ★<extra></extra>"
            ),
            customdata=data["average_rating"],
        )
    )
    fig.update_layout(**_base_layout(title, showlegend=False))
    fig.update_xaxes(title="Movies", gridcolor="rgba(255,255,255,0.07)")
    fig.update_yaxes(title="", gridcolor="rgba(255,255,255,0.05)")
    return fig


def _empty_figure(title: str, message: str) -> Any:
    """Placeholder figure so charts never crash on empty data."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=13, color="#8a90b0"),
    )
    fig.update_layout(**_base_layout(title))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


# ----------------------------------------------------------------------
# Convenience wrapper used by the UI
# ----------------------------------------------------------------------
class AnalyticsService:
    """Bundle of database reads plus the derived analytics for one page load."""

    def __init__(self, db: MovieDatabase) -> None:
        self.db = db

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return ``(merged_ratings, movies)`` read from MongoDB.

        The returned ratings frame is **already joined** with the movie metadata,
        because the ``ratings`` collection has no genre information while the
        ``movies`` collection has no ratings.  Pass it straight into
        :func:`user_statistics` / :func:`genre_preferences` to avoid loading the
        dataset a second time.
        """
        ratings = self.db.get_all_ratings()
        movies = self.db.get_all_movies()
        if not ratings.empty:
            ratings = ratings.merge(
                movies[["movieId", "title", "genres"]], on="movieId", how="left"
            )
        return ratings, movies

    def statistics(self) -> dict[str, Any]:
        """Dashboard statistics computed from MongoDB data."""
        return self.db.get_statistics()

    def user_profile(
        self,
        user_id: int,
        ratings: pd.DataFrame | None = None,
        movies: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Full analytics bundle for one user.

        Pass the frames from :meth:`load` to reuse them; when omitted the
        dataset is fetched from MongoDB.
        """
        if ratings is None:
            ratings, movies = self.load()
        return user_statistics(ratings, user_id)

    def user_ratings(self, user_id: int) -> pd.DataFrame:
        """One user's ratings joined with movie metadata, best rating first."""
        return self.db.get_user_ratings(user_id)

    def global_distribution(self) -> pd.DataFrame:
        """Global rating distribution straight from an aggregation pipeline."""
        return self.db.get_rating_distribution()

    def global_genres(
        self,
        limit: int = 15,
        ratings: pd.DataFrame | None = None,
        movies: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Most-rated genres across the dataset.

        Pass the frames from :meth:`load` to reuse them.  ``movies`` is only
        fetched when ``ratings`` still needs to be joined against it.
        """
        if ratings is None:
            ratings, movies = self.load()
        elif movies is None and "genres" not in ratings.columns:
            movies = self.db.get_all_movies()
        return global_genre_ranking(ratings, movies, limit=limit)

    def search(self, query: str, limit: int = 50) -> pd.DataFrame:
        """Title search executed by MongoDB."""
        try:
            return self.db.search_movies(query, limit=limit)
        except DatabaseError as exc:
            LOGGER.error("Search failed: %s", exc)
            return pd.DataFrame(
                columns=[
                    "movieId",
                    "title",
                    "genres",
                    "average_rating",
                    "rating_count",
                ]
            )


def top_genre_counter(ratings: pd.DataFrame, limit: int = 5) -> list[tuple[str, int]]:
    """Fast genre tally used for compact profile chips."""
    exploded = _explode_genres(ratings)
    if exploded.empty:
        return []
    counter = Counter(exploded["genre"].tolist())
    return [(genre, count) for genre, count in counter.most_common(limit)]


__all__ = [
    "ACCENT",
    "ACCENT_2",
    "AnalyticsError",
    "AnalyticsService",
    "HIGH_RATING_THRESHOLD",
    "global_genre_ranking",
    "genre_bar_figure",
    "genre_preferences",
    "overall_statistics",
    "preferred_genres",
    "rating_distribution",
    "rating_distribution_figure",
    "similarity_figure",
    "similarity_scatter_figure",
    "top_genre_counter",
    "user_genre_figure",
    "user_rating_distribution_figure",
    "user_statistics",
]
