"""Streamlit presentation layer for the Movie Recommender System.

Layout
------
* Sidebar        : user selection, Top-N / Top-K controls, generate button.
* Dashboard      : four MongoDB-backed statistic tiles.
* User profile   : rating behaviour, top rated movies, preferred genres.
* Recommendations: movie cards + the full result table.
* Similar users  : neighbour table and cosine-similarity charts.
* Visualisations : four Plotly charts driven by real database data.
* Search         : MongoDB title search with community ratings.

Database failures are surfaced as friendly ``st.error`` panels, never as
tracebacks.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import streamlit as st
from pymongo.errors import PyMongoError

from app.analytics import (
    ACCENT,
    ACCENT_2,
    HIGH_RATING_THRESHOLD,
    AnalyticsService,
    genre_bar_figure,
    rating_distribution_figure,
    similarity_figure,
    similarity_scatter_figure,
    user_genre_figure,
    user_rating_distribution_figure,
)
from app.config import Settings, get_settings
from app.database import DatabaseError, DatabaseNotAvailable, MovieDatabase
from app.recommender import (
    InsufficientDataError,
    Neighbor,
    UserBasedRecommender,
)

LOGGER = logging.getLogger("movie_recommender.ui")

# ----------------------------------------------------------------------
# Theme
# ----------------------------------------------------------------------
APP_TITLE = "🎬 Movie Recommender"
APP_SUBTITLE = "User-Based Collaborative Filtering · MongoDB · Cosine Similarity"

PRIMARY = "#4f8ef7"
ACCENT_COLOR = "#ffd166"
BG_DEEP = "#0b0d17"
BG_CARD = "#151829"
BG_CARD_ALT = "#1c2036"
BORDER = "rgba(255,255,255,0.10)"
TEXT = "#e8e8ef"
MUTED = "#98a0c0"

INJECTED_CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  .stApp {{
    background:
      radial-gradient(1100px 620px at 12% -8%, rgba(79,142,247,0.20), transparent 60%),
      radial-gradient(900px 520px at 92% 4%, rgba(255,209,102,0.13), transparent 58%),
      {BG_DEEP};
    color: {TEXT};
    font-family: 'Inter', 'Segoe UI', sans-serif;
  }}
  .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{ letter-spacing: -0.02em; }}

  /* Header -------------------------------------------------------- */
  .app-header {{
    display: flex; align-items: center; justify-content: space-between;
    gap: 1rem; flex-wrap: wrap;
    padding: 1.35rem 1.6rem; margin-bottom: 1.35rem;
    background: linear-gradient(120deg, rgba(79,142,247,0.20), rgba(28,32,54,0.92) 55%);
    border: 1px solid {BORDER}; border-radius: 20px;
    box-shadow: 0 18px 46px rgba(0,0,0,0.45);
  }}
  .app-header h1 {{
    margin: 0; font-size: 2.05rem; font-weight: 700; color: #fff;
    display: flex; align-items: center; gap: 0.6rem;
  }}
  .app-header p {{ margin: 0.35rem 0 0; color: {MUTED}; font-size: 0.92rem; }}

  .tech-pill {{
    display: inline-block; padding: 0.35rem 0.8rem; margin: 0.2rem;
    background: rgba(79,142,247,0.16); border: 1px solid rgba(79,142,247,0.42);
    border-radius: 999px; font-size: 0.75rem; color: #cfe0ff; font-weight: 500;
  }}

  /* Stat tiles ---------------------------------------------------- */
  .stat-card {{
    background: linear-gradient(160deg, {BG_CARD_ALT}, {BG_CARD});
    border: 1px solid {BORDER}; border-radius: 16px;
    padding: 1.05rem 1.2rem; text-align: left; height: 100%;
    box-shadow: 0 10px 26px rgba(0,0,0,0.32);
  }}
  .stat-card .label {{
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.10em;
    color: {MUTED}; font-weight: 600;
  }}
  .stat-card .value {{
    font-size: 1.85rem; font-weight: 700; color: #fff; margin-top: 0.3rem;
    line-height: 1.1;
  }}
  .stat-card .hint {{ font-size: 0.74rem; color: #7f88a8; margin-top: 0.25rem; }}

  /* Section headers ----------------------------------------------- */
  .section-title {{
    font-size: 1.32rem; font-weight: 650; color: #fff; margin: 1.5rem 0 0.2rem;
    display: flex; align-items: center; gap: 0.5rem;
  }}
  .section-sub {{ color: {MUTED}; font-size: 0.88rem; margin-bottom: 0.85rem; }}

  /* Profile summary ----------------------------------------------- */
  .profile-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 0.85rem; margin-bottom: 0.4rem;
  }}
  .profile-item {{
    background: {BG_CARD}; border: 1px solid {BORDER};
    border-left: 3px solid {PRIMARY};
    border-radius: 12px; padding: 0.75rem 0.9rem;
  }}
  .profile-item .k {{
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em;
    color: {MUTED}; font-weight: 600;
  }}
  .profile-item .v {{ font-size: 1.25rem; font-weight: 650; color: #fff; margin-top: 0.15rem; }}

  .chip {{
    display: inline-block; padding: 0.3rem 0.72rem; margin: 0.2rem 0.25rem 0.2rem 0;
    background: rgba(79,142,247,0.14); border: 1px solid rgba(79,142,247,0.38);
    border-radius: 999px; font-size: 0.8rem; color: #d6e4ff; font-weight: 500;
  }}
  .chip-accent {{
    background: rgba(255,209,102,0.13); border-color: rgba(255,209,102,0.40);
    color: #ffe6ab;
  }}

  /* Recommendation cards ------------------------------------------ */
  .rec-grid {{ display: grid; gap: 0.85rem; }}
  .rec-card {{
    background: linear-gradient(150deg, {BG_CARD_ALT} 0%, {BG_CARD} 100%);
    border: 1px solid {BORDER}; border-radius: 16px;
    padding: 1rem 1.1rem; position: relative; overflow: hidden;
    box-shadow: 0 10px 24px rgba(0,0,0,0.30);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
  }}
  .rec-card::before {{
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
    background: linear-gradient(180deg, {PRIMARY}, {ACCENT_COLOR});
  }}
  .rec-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 16px 34px rgba(0,0,0,0.45);
  }}
  .rec-title {{
    font-size: 1.03rem; font-weight: 650; color: #fff; margin: 0 0 0.45rem 0;
    line-height: 1.32;
  }}
  .rec-genres {{ margin: 0 0 0.6rem 0; }}
  .rec-genre {{
    display: inline-block; padding: 0.16rem 0.55rem; margin: 0 0.28rem 0.28rem 0;
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.14);
    border-radius: 6px; font-size: 0.71rem; color: #c3c9e0; font-weight: 500;
  }}
  .rec-meta {{ display: flex; gap: 1.1rem; align-items: center; flex-wrap: wrap; }}
  .rec-score {{
    display: inline-flex; align-items: center; gap: 0.35rem;
    background: rgba(255,209,102,0.15); border: 1px solid rgba(255,209,102,0.42);
    color: {ACCENT_COLOR}; font-weight: 700; font-size: 0.92rem;
    padding: 0.22rem 0.62rem; border-radius: 999px;
  }}
  .rec-neighbours {{ font-size: 0.78rem; color: {MUTED}; }}
  .rec-neighbours b {{ color: #cdd4f0; font-weight: 600; }}
  .rec-bar {{
    height: 5px; background: rgba(255,255,255,0.08); border-radius: 999px;
    margin-top: 0.7rem; overflow: hidden;
  }}
  .rec-bar span {{
    display: block; height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, {PRIMARY}, {ACCENT_COLOR});
  }}

  /* Sidebar ------------------------------------------------------- */
  section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #12152780, #0d0f1c);
    border-right: 1px solid {BORDER};
  }}
  section[data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}
  .brand {{
    display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.15rem;
  }}
  .brand h2 {{ margin: 0; font-size: 1.28rem; color: #fff; font-weight: 700; }}
  .brand-sub {{ color: {MUTED}; font-size: 0.76rem; margin-bottom: 1rem; }}

  .algo-note {{
    background: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 12px;
    padding: 0.8rem 0.9rem; font-size: 0.78rem; color: #b9c0da; line-height: 1.55;
  }}
  .algo-note code {{
    background: rgba(79,142,247,0.16); color: #cfe0ff; padding: 0.05rem 0.32rem;
    border-radius: 5px; font-size: 0.74rem;
  }}

  .footnote {{ color: #6f7899; font-size: 0.74rem; text-align: center; padding-top: 0.8rem; }}

  /* Streamlit widget polish ---------------------------------------- */
  .stButton > button {{
    background: linear-gradient(135deg, {PRIMARY}, #6a5cf0);
    color: #fff; border: none; border-radius: 11px; font-weight: 650;
    font-size: 0.95rem; padding: 0.6rem 1rem; width: 100%;
    box-shadow: 0 8px 20px rgba(79,142,247,0.30);
    transition: transform 0.12s ease, box-shadow 0.12s ease;
  }}
  .stButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 12px 26px rgba(79,142,247,0.42);
  }}
  div[data-testid="stMetricValue"] {{ font-size: 1.5rem; }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 0.4rem; border-bottom: 1px solid {BORDER}; }}
  .stTabs [data-baseweb="tab"] {{
    background: transparent; color: {MUTED}; border-radius: 9px 9px 0 0;
    padding: 0.5rem 0.95rem; font-weight: 550;
  }}
  .stTabs [aria-selected="true"] {{ color: #fff !important; background: rgba(79,142,247,0.16); }}
  .stDataFrame {{ border: 1px solid {BORDER}; border-radius: 12px; overflow: hidden; }}
</style>
"""


# ----------------------------------------------------------------------
# Cached resources
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Connecting to MongoDB ...")
def get_database(uri: str, db_name: str) -> MovieDatabase:
    """Return a process-wide MongoDB client (Streamlit keeps it alive)."""
    return MovieDatabase(uri=uri, db_name=db_name)


@st.cache_data(show_spinner=False)
def load_user_ids(uri: str, db_name: str) -> list[int]:
    """All user ids that have at least one rating."""
    return get_database(uri, db_name).get_user_ids()


@st.cache_data(show_spinner=False)
def load_statistics(uri: str, db_name: str) -> dict[str, Any]:
    """Dashboard statistics, aggregated by MongoDB."""
    return get_database(uri, db_name).get_statistics()


@st.cache_data(show_spinner="Building user-movie matrix ...")
def load_frames(uri: str, db_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(ratings, movies)`` joined frames used across the app."""
    ratings = get_database(uri, db_name).get_all_ratings()
    movies = get_database(uri, db_name).get_all_movies()
    if not ratings.empty:
        ratings = ratings.merge(
            movies[["movieId", "title", "genres"]], on="movieId", how="left"
        )
    return ratings, movies


@st.cache_resource(show_spinner=False)
def get_recommender(uri: str, db_name: str) -> UserBasedRecommender:
    """Reuse one recommender instance (and its cached similarity matrix)."""
    return UserBasedRecommender(get_database(uri, db_name))


@st.cache_data(show_spinner=False)
def cached_recommendation(
    uri: str, db_name: str, user_id: int, top_n: int, top_k: int
) -> list[dict[str, Any]]:
    """Recommendations for one user, cached per (user, top_n, top_k)."""
    recommender = get_recommender(uri, db_name)
    result = recommender.recommend(user_id, top_n=top_n, top_k=top_k)
    return [rec.as_dict() for rec in result.recommendations]


@st.cache_data(show_spinner=False)
def cached_neighbors(uri: str, db_name: str, user_id: int, top_k: int) -> list[dict[str, Any]]:
    """Similar users for one user, cached per (user, top_k)."""
    recommender = get_recommender(uri, db_name)
    return [
        {
            "user_id": n.user_id,
            "similarity": round(float(n.similarity), 4),
            "rating_count": int(n.rating_count),
        }
        for n in recommender.get_similar_users(user_id, top_k=top_k)
    ]


@st.cache_data(show_spinner=False)
def search_movies_cached(uri: str, db_name: str, query: str, limit: int) -> pd.DataFrame:
    """MongoDB title search, cached per query string."""
    return get_database(uri, db_name).search_movies(query, limit=limit)


def _as_neighbors(records: list[dict[str, Any]]) -> list[Neighbor]:
    """Rebuild :class:`Neighbor` objects from the cached dict representation."""
    return [
        Neighbor(
            user_id=int(record["user_id"]),
            similarity=float(record["similarity"]),
            rating_count=int(record["rating_count"]),
        )
        for record in records
    ]


# ----------------------------------------------------------------------
# Small rendering helpers
# ----------------------------------------------------------------------
def section_title(icon: str, title: str, subtitle: str = "") -> None:
    """Render a consistent section header."""
    st.markdown(f'<div class="section-title">{icon} {title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


def render_header() -> None:
    """Render the top application banner."""
    st.markdown(
        f"""
        <div class="app-header">
          <div>
            <h1>{APP_TITLE}</h1>
            <p>{APP_SUBTITLE}</p>
          </div>
          <div>
            <span class="tech-pill">MongoDB</span>
            <span class="tech-pill">PyMongo</span>
            <span class="tech-pill">User-Based CF</span>
            <span class="tech-pill">Cosine Similarity</span>
            <span class="tech-pill">Streamlit</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_cards(stats: dict[str, Any]) -> None:
    """Render the four MongoDB-backed dashboard statistics."""
    cards = [
        ("🎞️ Total Movies", f"{stats.get('total_movies', 0):,}", "distinct movieIds in `movies`"),
        ("👥 Total Users", f"{stats.get('total_users', 0):,}", "distinct userIds in `ratings`"),
        ("⭐ Total Ratings", f"{stats.get('total_ratings', 0):,}", "rating documents stored"),
        (
            "💫 Average Rating",
            f"{stats.get('average_rating', 0.0):.2f} ★",
            "server-side $avg aggregation",
        ),
    ]
    columns = st.columns(4)
    for column, (label, value, hint) in zip(columns, cards):
        with column:
            st.markdown(
                f"""
                <div class="stat-card">
                  <div class="label">{label}</div>
                  <div class="value">{value}</div>
                  <div class="hint">{hint}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _star_bar(value: float, total: float = 5.0) -> str:
    """Render a thin progress bar for a predicted rating."""
    pct = max(0.0, min(1.0, value / total)) * 100.0
    return (
        '<div class="rec-bar"><span style="width: '
        f"{pct:.1f}%; background: linear-gradient(90deg, {PRIMARY}, {ACCENT_COLOR});"
        '"></span></div>'
    )


def render_recommendation_cards(recommendations: list[dict[str, Any]]) -> None:
    """Render recommendation cards in a responsive two-column grid."""
    if not recommendations:
        return

    rows = (len(recommendations) + 1) // 2
    for row in range(rows):
        pair = recommendations[row * 2 : row * 2 + 2]
        for column, rec in zip(st.columns(2), pair):
            genres = rec.get("genres") or []
            genre_html = "".join(f'<span class="rec-genre">{g}</span>' for g in genres[:4])
            if len(genres) > 4:
                genre_html += f'<span class="rec-genre">+{len(genres) - 4}</span>'
            if not genre_html:
                genre_html = '<span class="rec-genre">Uncategorized</span>'
            similarity_strength = rec.get("similarity_mass", 0.0)
            similarity_pct = max(0.0, min(1.0, similarity_strength / 3.0)) * 100.0
            with column:
                st.markdown(
                    f"""
                    <div class="rec-card">
                      <div class="rec-title">{rec['title']}</div>
                      <div class="rec-genres">{genre_html}</div>
                      <div class="rec-meta">
                        <span class="rec-score">★ {rec['predicted_rating']:.2f}</span>
                        <span class="rec-neighbours">
                          <b>{rec['neighbor_count']}</b> similar user(s) rated this
                        </span>
                      </div>
                      <div class="rec-bar">
                        <span style="width:{similarity_pct:.1f}%"></span>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.write("")


def render_user_profile(profile: dict[str, Any]) -> None:
    """Render the user profile section: metrics, chips and top movies."""
    st.markdown(
        f"""
        <div class="profile-grid">
          <div class="profile-item"><div class="k">User ID</div><div class="v">{profile['user_id']}</div></div>
          <div class="profile-item"><div class="k">Movies Rated</div><div class="v">{profile['rating_count']:,}</div></div>
          <div class="profile-item"><div class="k">Average Given</div><div class="v">{profile['average_rating']:.2f} ★</div></div>
          <div class="profile-item"><div class="k">Rating Range</div><div class="v">{profile['min_rating']:.0f} – {profile['max_rating']:.0f} ★</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**🏅 Top-Rated Movies by This User**")
        top_movies = profile.get("top_movies")
        if top_movies is None or top_movies.empty:
            st.info("No rated movies found for this user.")
        else:
            display = top_movies[["title", "rating", "genres"]].copy()
            display["genres"] = display["genres"].apply(
                lambda g: ", ".join(list(g)) if isinstance(g, (list, tuple)) else ""
            )
            display.columns = ["Movie", "Rating", "Genres"]
            st.dataframe(
                display,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Rating": st.column_config.ProgressColumn(
                        "Rating",
                        min_value=0.0,
                        max_value=5.0,
                        format="%.1f ★",
                    )
                },
            )

    with right:
        st.markdown("**🎭 Preferred Genres**")
        top_genres = profile.get("top_genres") or []
        if not top_genres:
            st.info("Not enough rated movies to infer a genre profile.")
        else:
            chips = "".join(
                f'<span class="chip chip-accent">{g}</span>' for g in top_genres[:3]
            )
            chips += "".join(
                f'<span class="chip">{g}</span>' for g in top_genres[3:]
            )
            st.markdown(chips, unsafe_allow_html=True)

            st.markdown(
                f"<div class='section-sub' style='margin-top:0.7rem'>"
                f"Genres counted from movies rated {HIGH_RATING_THRESHOLD:g}★ and above."
                f"</div>",
                unsafe_allow_html=True,
            )
            counts = profile.get("genre_counts")
            if counts is not None and not counts.empty:
                st.dataframe(
                    counts.rename(
                        columns={
                            "genre": "Genre",
                            "count": "Movies",
                            "average_rating": "Avg ★",
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                    height=240,
                )


def render_recommendations(recommendations: list[dict[str, Any]], message: str) -> None:
    """Render the recommendation section: cards plus a sortable table."""
    if not recommendations:
        st.info(f"No recommendations available. {message}".strip())
        return

    st.success(
        f"Generated **{len(recommendations)}** recommendation(s) from real MovieLens "
        f"ratings. {message}"
    )
    render_recommendation_cards(recommendations)

    with st.expander("📋 View as table", expanded=False):
        frame = pd.DataFrame(recommendations)
        frame = frame.rename(
            columns={
                "title": "Movie",
                "genres_display": "Genres",
                "predicted_rating": "Predicted ★",
                "neighbor_count": "Neighbors",
                "similarity_mass": "Similarity Mass",
            }
        )
        st.dataframe(
            frame[["Movie", "Genres", "Predicted ★", "Neighbors", "Similarity Mass"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "Predicted ★": st.column_config.ProgressColumn(
                    "Predicted ★", min_value=0.0, max_value=5.0, format="%.2f ★"
                )
            },
        )


def render_similar_users(neighbors: list[dict[str, Any]]) -> None:
    """Render the Top-K similar users table."""
    if not neighbors:
        st.warning(
            "No similar users were found. This happens when the selected user has "
            "rated movies that nobody else has rated."
        )
        return
    frame = pd.DataFrame(neighbors).rename(
        columns={
            "user_id": "User ID",
            "similarity": "Cosine Similarity",
            "rating_count": "Movies Rated",
        }
    )
    st.dataframe(
        frame,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Cosine Similarity": st.column_config.ProgressColumn(
                "Cosine Similarity",
                min_value=0.0,
                max_value=1.0,
                format="%.4f",
            )
        },
    )


def render_search(uri: str, db_name: str) -> None:
    """Render the MongoDB-backed movie search section."""
    section_title(
        "🔍",
        "Movie Search",
        "Search runs as a MongoDB query against the `movies` collection, "
        "joined with the community rating from `ratings`.",
    )
    column, spacer = st.columns([3, 1])
    with column:
        query = st.text_input(
            "Search by title",
            placeholder="e.g. Matrix, Star Wars, Godfather ...",
            label_visibility="collapsed",
        )
    with spacer:
        limit = st.select_slider("Max results", options=[10, 25, 50, 100], value=25)

    if not query.strip():
        st.markdown(
            '<div class="algo-note">Type at least one character to search. '
            "The lookup uses a case-insensitive regex on <code>title</code> and an "
            "<code>$lookup</code> to the <code>ratings</code> collection for the "
            "community average.</div>",
            unsafe_allow_html=True,
        )
        return

    try:
        with st.spinner("Searching MongoDB ..."):
            results = search_movies_cached(uri, db_name, query.strip(), int(limit))
    except DatabaseError as exc:
        st.error(f"Search failed: {exc}")
        return

    if results.empty:
        st.info(f'No movies matched "{query.strip()}".')
        return

    st.caption(f"{len(results)} match(es) found.")
    display = results.copy()
    display["genres"] = display["genres"].apply(
        lambda g: ", ".join(list(g)) if isinstance(g, (list, tuple)) else "Uncategorized"
    )
    display["average_rating"] = display["average_rating"].round(2)
    display = display.rename(
        columns={
            "title": "Movie",
            "genres": "Genres",
            "average_rating": "Community ★",
            "rating_count": "Ratings",
        }
    )
    st.dataframe(
        display[["Movie", "Genres", "Community ★", "Ratings"]],
        hide_index=True,
        use_container_width=True,
        column_config={
            "Community ★": st.column_config.ProgressColumn(
                "Community ★", min_value=0.0, max_value=5.0, format="%.2f ★"
            )
        },
    )


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
def render_sidebar(settings: Settings) -> dict[str, Any]:
    """Render the sidebar and return the current control values."""
    st.sidebar.markdown(
        f"""
        <div class="brand">
          <span style="font-size:1.7rem">🎬</span>
          <h2>Movie Recommender</h2>
        </div>
        <div class="brand-sub">User-based collaborative filtering</div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        f"""
        <div class="algo-note">
          <b style="color:#fff">How it works</b><br>
          1 · Ratings are read from MongoDB<br>
          2 · A user × movie matrix is built<br>
          3 · Cosine similarity finds your neighbours<br>
          4 · Neighbours vote on unseen movies<br><br>
          <code>pred = Σ(sim × rating) / Σ(sim)</code>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.write("")

    controls: dict[str, Any] = {"settings": settings}

    st.sidebar.markdown("**⚙️ Controls**")
    try:
        user_ids = load_user_ids(settings.mongodb_uri, settings.db_name)
    except (DatabaseError, InsufficientDataError) as exc:
        st.sidebar.error(str(exc))
        user_ids = []

    if not user_ids:
        st.sidebar.warning(
            "No users found. Run `python -m scripts.load_data` to import the "
            "MovieLens dataset into MongoDB."
        )
        return controls

    default_index = min(30, len(user_ids) - 1)
    user_id = st.sidebar.selectbox(
        "Select a user",
        options=user_ids,
        index=default_index,
        format_func=lambda uid: f"User {uid}",
        help="Pick the user whose taste profile you want to explore.",
    )
    top_n = st.sidebar.slider(
        "Number of recommendations",
        min_value=5,
        max_value=20,
        value=min(10, len(range(5, 21))),
        step=1,
        help="How many unseen movies to recommend (Top-N).",
    )
    top_k = st.sidebar.slider(
        "Number of similar users",
        min_value=5,
        max_value=50,
        value=10,
        step=1,
        help="How many nearest neighbours to consider (Top-K).",
    )

    generate = st.sidebar.button("✨ Generate Recommendations", use_container_width=True)

    st.sidebar.write("")
    st.sidebar.markdown(
        f'<div class="footnote">DB: <code>{settings.db_name}</code><br>'
        f"{len(user_ids)} users loaded</div>",
        unsafe_allow_html=True,
    )

    controls.update(
        {
            "user_id": user_id,
            "top_n": int(top_n),
            "top_k": int(top_k),
            "generate": generate,
            "user_count": len(user_ids),
        }
    )
    return controls


# ----------------------------------------------------------------------
# Main application
# ----------------------------------------------------------------------
def main() -> None:
    """Application entry point used by ``main.py`` and ``streamlit run``."""
    st.set_page_config(
        page_title="Movie Recommender System",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(INJECTED_CSS, unsafe_allow_html=True)

    settings = get_settings()
    render_header()

    if not settings.mongodb_is_configured:
        st.error(
            "**MONGODB_URI is not configured.** Add it to your `.env` file, e.g. "
            "`MONGODB_URI=mongodb://localhost:27017/`, then restart the app."
        )
        st.stop()

    controls = render_sidebar(settings)

    if "user_id" not in controls:
        st.info(
            "Set up your database first, then come back:\n\n"
            "```bash\npython -m scripts.load_data\n```"
        )
        st.stop()

    uri, db_name = settings.mongodb_uri, settings.db_name
    user_id = controls["user_id"]

    # ---- connection banner + data availability check -----------------
    try:
        db = get_database(uri, db_name)
        if not db.is_connected:
            raise DatabaseNotAvailable("The MongoDB server did not answer a ping.")
        stats = load_statistics(uri, db_name)
    except (DatabaseError, PyMongoError) as exc:
        st.error(
            f"**Could not reach MongoDB** — {exc}\n\n"
            "Start a local server (`mongod`) or set `MONGODB_URI` in your `.env` "
            "file to an Atlas connection string, then restart the app."
        )
        st.stop()

    if stats.get("total_ratings", 0) == 0:
        st.warning(
            "MongoDB is connected but the collections are empty. Initialise the "
            "database with `python -m scripts.load_data` and refresh this page."
        )
        st.stop()

    # ---- dashboard ----------------------------------------------------
    section_title(
        "📊",
        "Dataset Dashboard",
        "All four figures are aggregated by MongoDB on every refresh.",
    )
    render_stat_cards(stats)
    st.write("")

    try:
        with st.spinner("Loading ratings and movies ..."):
            ratings, movies = load_frames(uri, db_name)
    except (DatabaseError, InsufficientDataError) as exc:
        st.error(f"Could not load data: {exc}")
        st.stop()

    if ratings.empty:
        st.error("The `ratings` collection is empty. Run `python -m scripts.load_data`.")
        st.stop()

    service = AnalyticsService(db)

    # ---- user profile + recommendations ------------------------------
    profile_tab, rec_tab, users_tab, charts_tab, search_tab = st.tabs(
        [
            "👤 User Profile",
            "✨ Recommendations",
            "🧑‍🤝‍🧑 Similar Users",
            "📈 Visualisations",
            "🔍 Movie Search",
        ]
    )

    with profile_tab:
        section_title(
            "👤",
            f"User {user_id} Profile",
            "Computed from this user's ratings in the `ratings` collection, "
            "joined with `movies` metadata.",
        )
        try:
            profile = service.user_profile(user_id, ratings=ratings, movies=movies)
        except DatabaseError as exc:
            st.error(f"Could not build the user profile: {exc}")
            st.stop()
        render_user_profile(profile)

    with rec_tab:
        section_title(
            "✨",
            "Recommended Movies",
            "Movies you have not rated, ranked by a similarity-weighted average "
            "of your neighbours' ratings.",
        )
        if not controls.get("generate"):
            st.info(
                "Press **✨ Generate Recommendations** in the sidebar to compute "
                f"the Top-{controls['top_n']} movies for User {user_id} using "
                f"{controls['top_k']} most similar users."
            )
        else:
            try:
                with st.spinner("Running collaborative filtering ..."):
                    recommendations = cached_recommendation(
                        uri, db_name, user_id, controls["top_n"], controls["top_k"]
                    )
                    neighbors = cached_neighbors(uri, db_name, user_id, controls["top_k"])
            except InsufficientDataError as exc:
                st.warning(f"Not enough data: {exc}")
                recommendations, neighbors = [], []
            except DatabaseError as exc:
                st.error(f"Recommendation failed: {exc}")
                st.stop()

            message = (
                f"Evaluated from {len(neighbors)} similar user(s); already-rated "
                "movies are always excluded."
            )
            render_recommendations(recommendations, message)

            if recommendations and neighbors:
                best = neighbors[0]
                st.markdown(
                    f"""
                    <div class="algo-note" style="margin-top:0.9rem">
                      Your closest match is <b>User {best['user_id']}</b> with a cosine
                      similarity of <b>{best['similarity']:.4f}</b> across
                      <b>{best['rating_count']}</b> rated movies.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with users_tab:
        section_title(
            "🧑‍🤝‍🧑",
            f"Top {controls['top_k']} Similar Users to User {user_id}",
            "Cosine similarity over the user × movie rating vectors. "
            "The selected user is always excluded.",
        )
        if not controls.get("generate"):
            st.info("Press **✨ Generate Recommendations** to find your neighbours.")
        else:
            try:
                neighbors = cached_neighbors(uri, db_name, user_id, controls["top_k"])
            except (DatabaseError, InsufficientDataError) as exc:
                st.error(f"Could not compute similar users: {exc}")
                neighbors = []
            render_similar_users(neighbors)
            if neighbors:
                st.plotly_chart(
                    similarity_scatter_figure(
                        _as_neighbors(neighbors),
                        target_rating_count=int(profile["rating_count"]),
                    ),
                    use_container_width=True,
                )

    with charts_tab:
        section_title(
            "📈",
            "Visualisations",
            "Every chart is built from data currently stored in MongoDB.",
        )
        chart_row1 = st.columns(2)
        with chart_row1[0]:
            st.plotly_chart(
                rating_distribution_figure(
                    service.global_distribution(), "Rating Distribution · All Users"
                ),
                use_container_width=True,
            )
        with chart_row1[1]:
            st.plotly_chart(
                user_rating_distribution_figure(profile, f"User {user_id} Rating Spread"),
                use_container_width=True,
            )

        chart_row2 = st.columns(2)
        with chart_row2[0]:
            st.plotly_chart(
                user_genre_figure(profile), use_container_width=True
            )
        with chart_row2[1]:
            neighbors: list[dict[str, Any]] = []
            if controls.get("generate"):
                try:
                    neighbors = cached_neighbors(uri, db_name, user_id, controls["top_k"])
                except (DatabaseError, InsufficientDataError):
                    neighbors = []
            st.plotly_chart(
                similarity_figure(_as_neighbors(neighbors), top=10),
                use_container_width=True,
            )

        st.plotly_chart(
            genre_bar_figure(
                service.global_genres(limit=14, ratings=ratings, movies=movies),
                "Most Rated Genres Across the Dataset",
            ),
            use_container_width=True,
        )

    with search_tab:
        render_search(uri, db_name)

    st.markdown(
        '<div class="footnote">MovieLens Latest Small · 100k ratings · '
        "recommendations computed with user-based collaborative filtering and "
        "cosine similarity.</div>",
        unsafe_allow_html=True,
    )


__all__ = ["main", "INJECTED_CSS", "render_header", "render_sidebar"]
