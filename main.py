"""Entry point for the Movie Recommender System.

Run the web application with::

    streamlit run main.py

or, to verify the environment and the database without a browser::

    python main.py            # CLI self-check
    python main.py --check    # same as above
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from app.config import get_settings


def run_self_check(verbose: bool = True) -> int:
    """Verify the configuration, the MongoDB connection and the algorithm.

    Returns a process exit code: ``0`` success, ``1`` failure.
    """
    from app.database import DatabaseError, MovieDatabase
    from app.recommender import InsufficientDataError, UserBasedRecommender

    settings = get_settings()

    def say(message: str) -> None:
        if verbose:
            print(message)

    say("=" * 66)
    say("Movie Recommender System - environment self-check")
    say("=" * 66)
    say(f"Database name : {settings.db_name}")
    say(f"MongoDB URI   : {_mask_uri(settings.mongodb_uri)}")

    db = MovieDatabase()
    try:
        db.connect()
    except DatabaseError as exc:
        print(f"\n[FAIL] Could not connect to MongoDB:\n{exc}")
        print(
            "\nNext steps:\n"
            "  1. Start MongoDB locally (mongod) or create a free Atlas cluster.\n"
            "  2. Copy .env.example to .env and set MONGODB_URI / DB_NAME.\n"
            "  3. Run 'python -m scripts.load_data' to import the dataset."
        )
        return 1

    try:
        stats = db.get_statistics()
        say(
            f"Movies        : {stats['total_movies']:,}\n"
            f"Users         : {stats['total_users']:,}\n"
            f"Ratings       : {stats['total_ratings']:,}\n"
            f"Average rating: {stats['average_rating']:.3f} *"
        )
        if stats["total_ratings"] == 0:
            print("\n[WARN] The database is reachable but empty.")
            print("       Run 'python -m scripts.load_data' to import MovieLens.")
            return 1

        recommender = UserBasedRecommender(db)
        target = recommender.user_ids[0]
        result = recommender.recommend(target, top_n=5, top_k=10)

        say("-" * 66)
        say(f"Similar users for User {target} (cosine similarity):")
        for neighbor in result.neighbors[:5]:
            say(
                f"  User {neighbor.user_id:<5} sim={neighbor.similarity:.4f} "
                f"rated={neighbor.rating_count}"
            )
        say("-" * 66)
        say(f"Top-{len(result.recommendations)} recommendations for User {target}:")
        if result.recommendations:
            for rank, rec in enumerate(result.recommendations, start=1):
                genres = ", ".join(rec.genres) or "Uncategorized"
                say(
                    f"  {rank}. {rec.title}  ->  {rec.predicted_rating:.2f} * "
                    f"({rec.neighbor_count} neighbours)  [{genres}]"
                )
        else:
            say(f"  none ({result.message})")

        rated_ids = set(
            recommender.matrix.columns[recommender.mask.loc[target].to_numpy()]
        )
        leaked = [r.movie_id for r in result.recommendations if r.movie_id in rated_ids]
        if leaked:
            print(f"\n[FAIL] Already-rated movies leaked into results: {leaked}")
            return 1

        say("-" * 66)
        say("[OK] Recommendations generated from real ratings; "
            "no already-rated movie was returned.")
        return 0
    except (DatabaseError, InsufficientDataError) as exc:
        print(f"\n[FAIL] {exc}")
        return 1
    finally:
        db.close()


def _mask_uri(uri: str) -> str:
    """Hide credentials before printing a connection string."""
    if not uri:
        return "<not set>"
    if "@" in uri:
        scheme, _, rest = uri.partition("://")
        _, _, host = rest.partition("@")
        return f"{scheme}://***:***@{host}"
    return uri


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Launch the Streamlit app (no arguments) or run the self-check. "
            "The web UI is started with: streamlit run main.py"
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify MongoDB and the recommender instead of starting Streamlit.",
    )
    args = parser.parse_args(argv)

    if args.check:
        return run_self_check()

    try:
        from app.ui import main as ui_main
    except ImportError as exc:  # pragma: no cover - dependency problem
        print(f"[FAIL] Streamlit or a required dependency is missing: {exc}")
        print("       Run 'pip install -r requirements.txt' first.")
        return 1

    ui_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
