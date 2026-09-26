"""Movie Recommender System application package.

Modules
-------
database    : MongoDB connectivity, queries and statistics.
recommender : User-based collaborative filtering with cosine similarity.
analytics   : User statistics, genre analysis and distribution data.
ui          : Streamlit presentation layer.
config      : Environment-driven configuration and logging setup.
"""

from app.config import settings

__all__ = ["settings", "__version__"]

__version__ = "1.0.0"
