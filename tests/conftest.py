"""Pytest configuration for the Movie Recommender System.

The recommendation-algorithm tests use small synthetic DataFrames and never
open a socket, so ``pytest`` works without a running MongoDB instance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture()
def sample_ratings():
    """A tiny ratings frame with hand-verified cosine similarities.

    Expected similarities to User 1 (ratings ``[5, 4, 1, 0, 0]``)::

        User 2 -> 42 / (sqrt(42) * sqrt(45)) = 0.96610
        User 3 -> 41 / (sqrt(42) * sqrt(66)) = 0.77873
        User 4 -> 14 / (sqrt(42) * sqrt(27)) = 0.41577
        User 5 -> 0.0  (disjoint movies, therefore dropped)

    User 1 has not rated movie 40, which neighbours 2-4 (only User 3) have
    rated, so movie 40 is the single expected recommendation.
    """
    import pandas as pd

    return pd.DataFrame(
        [
            # User 1 and User 2 have near-identical taste.
            {"userId": 1, "movieId": 10, "rating": 5.0},
            {"userId": 1, "movieId": 20, "rating": 4.0},
            {"userId": 1, "movieId": 30, "rating": 1.0},
            {"userId": 2, "movieId": 10, "rating": 4.0},
            {"userId": 2, "movieId": 20, "rating": 5.0},
            {"userId": 2, "movieId": 30, "rating": 2.0},
            # User 3 agrees with User 1 and additionally rated movie 40.
            {"userId": 3, "movieId": 10, "rating": 5.0},
            {"userId": 3, "movieId": 20, "rating": 4.0},
            {"userId": 3, "movieId": 40, "rating": 5.0},
            # User 4 is a low-rating user but still a (weak) positive match.
            {"userId": 4, "movieId": 10, "rating": 1.0},
            {"userId": 4, "movieId": 20, "rating": 1.0},
            {"userId": 4, "movieId": 30, "rating": 5.0},
            # User 5 rates movies 40 and 50, which nobody else shares with
            # User 1, so its similarity to User 1 is exactly 0.
            {"userId": 5, "movieId": 40, "rating": 4.0},
            {"userId": 5, "movieId": 50, "rating": 5.0},
        ]
    )


@pytest.fixture()
def sample_movies():
    """Movie metadata matching :func:`sample_ratings`."""
    import pandas as pd

    return pd.DataFrame(
        [
            {"movieId": 10, "title": "Action Hour (2001)", "genres": ["Action"]},
            {"movieId": 20, "title": "Comic Relief (2005)", "genres": ["Comedy"]},
            {"movieId": 30, "title": "Dark Depths (2011)", "genres": ["Horror", "Thriller"]},
            {"movieId": 40, "title": "Solo Film (2016)", "genres": ["Drama"]},
            {"movieId": 50, "title": "Untouched Classic (1999)", "genres": ["Classic"]},
        ]
    )
