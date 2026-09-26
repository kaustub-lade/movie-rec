"""Unit tests for the user-based collaborative filtering engine.

Every test uses a small synthetic DataFrame, so no MongoDB instance is needed.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.recommender import (
    InsufficientDataError,
    UserBasedRecommender,
    build_user_movie_matrix,
    calculate_similarity_matrix,
    get_similar_users,
    recommend_for_user,
    weighted_predicted_rating,
)


# ----------------------------------------------------------------------
# 1. Matrix construction
# ----------------------------------------------------------------------
class TestUserMovieMatrix:
    def test_matrix_shape_and_axes_are_sorted(self, sample_ratings):
        matrix, mask = build_user_movie_matrix(sample_ratings)

        assert list(matrix.index) == [1, 2, 3, 4, 5]
        assert list(matrix.columns) == [10, 20, 30, 40, 50]
        assert matrix.shape == (5, 5)
        assert mask.shape == matrix.shape

    def test_ratings_land_in_the_correct_cells(self, sample_ratings):
        matrix, _ = build_user_movie_matrix(sample_ratings)

        assert matrix.loc[1, 10] == 5.0
        assert matrix.loc[1, 20] == 4.0
        assert matrix.loc[1, 30] == 1.0
        # Movie 40 and 50 were not rated by User 1 -> filled with zero.
        assert matrix.loc[1, 40] == 0.0
        assert matrix.loc[1, 50] == 0.0

    def test_missing_ratings_are_zero_in_matrix(self, sample_ratings):
        matrix, _ = build_user_movie_matrix(sample_ratings)
        missing = [
            (1, 40),
            (1, 50),
            (2, 40),
            (2, 50),
            (3, 30),
            (3, 50),
            (5, 10),
            (5, 20),
            (5, 30),
        ]
        for user_id, movie_id in missing:
            assert matrix.loc[user_id, movie_id] == 0.0

    def test_mask_distinguishes_missing_from_zero(self, sample_ratings):
        """A 0 in the matrix means 'not rated', which the mask must record."""
        matrix, mask = build_user_movie_matrix(sample_ratings)

        assert bool(mask.loc[1, 10]) is True       # genuinely rated
        assert bool(mask.loc[1, 40]) is False      # not rated -> zero in matrix
        # MovieLens ratings are 0.5..5.0, so a rated cell is never 0.0.
        assert (matrix.to_numpy()[mask.to_numpy()] != 0.0).all()
        assert (matrix.to_numpy()[~mask.to_numpy()] == 0.0).all()

    def test_matrix_is_dense_float(self, sample_ratings):
        matrix, mask = build_user_movie_matrix(sample_ratings)
        assert matrix.dtypes.apply(lambda d: d == float).all()
        assert mask.dtypes.apply(lambda d: d == bool).all()

    def test_duplicate_pairs_keep_the_last_rating(self):
        ratings = pd.DataFrame(
            [
                {"userId": 1, "movieId": 10, "rating": 2.0},
                {"userId": 1, "movieId": 10, "rating": 5.0},
            ]
        )
        matrix, mask = build_user_movie_matrix(ratings)
        assert matrix.loc[1, 10] == 5.0
        assert int(mask.sum().sum()) == 1

    def test_empty_frame_raises(self):
        with pytest.raises(InsufficientDataError):
            build_user_movie_matrix(pd.DataFrame(columns=["userId", "movieId", "rating"]))

    def test_missing_columns_raise(self):
        with pytest.raises(InsufficientDataError) as excinfo:
            build_user_movie_matrix(pd.DataFrame([{"userId": 1, "rating": 4.0}]))
        assert "movieId" in str(excinfo.value)

    def test_non_dataframe_raises(self):
        with pytest.raises(InsufficientDataError):
            build_user_movie_matrix([{"userId": 1, "movieId": 2, "rating": 3}])


# ----------------------------------------------------------------------
# 2. Cosine similarity
# ----------------------------------------------------------------------
class TestCosineSimilarity:
    def test_identical_vectors_are_perfectly_similar(self):
        matrix = pd.DataFrame(
            [[5.0, 4.0, 1.0], [5.0, 4.0, 1.0]], index=[1, 2], columns=[10, 20, 30]
        )
        similarity = calculate_similarity_matrix(matrix)
        assert similarity.loc[1, 2] == pytest.approx(1.0)

    def test_matches_the_manual_dot_product_formula(self):
        # docstring formula: dot(A, B) / (norm(A) * norm(B))
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        matrix = pd.DataFrame([a, b], index=[1, 2], columns=[10, 20, 30])

        expected = float(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        )
        similarity = calculate_similarity_matrix(matrix)
        assert similarity.loc[1, 2] == pytest.approx(expected, rel=1e-12)

    def test_orthogonal_vectors_score_zero(self):
        matrix = pd.DataFrame(
            [[1.0, 0.0], [0.0, 1.0]], index=[1, 2], columns=[10, 20]
        )
        similarity = calculate_similarity_matrix(matrix)
        assert similarity.loc[1, 2] == pytest.approx(0.0, abs=1e-12)

    def test_similarity_matrix_is_symmetric(self, sample_ratings):
        matrix, _ = build_user_movie_matrix(sample_ratings)
        similarity = calculate_similarity_matrix(matrix)
        assert np.allclose(similarity.to_numpy(), similarity.to_numpy().T)

    def test_diagonal_is_zeroed_so_a_user_is_not_its_own_neighbour(self, sample_ratings):
        matrix, _ = build_user_movie_matrix(sample_ratings)
        similarity = calculate_similarity_matrix(matrix)
        assert (np.diag(similarity.to_numpy()) == 0.0).all()

    def test_all_zero_row_yields_zero_similarity_not_nan(self):
        """A user with no ratings must not produce nan (division by zero)."""
        matrix = pd.DataFrame(
            [[5.0, 4.0], [0.0, 0.0]], index=[1, 2], columns=[10, 20]
        )
        similarity = calculate_similarity_matrix(matrix)
        assert not similarity.isna().any().any()
        assert similarity.loc[2, 1] == 0.0

    def test_similarity_is_bounded_by_one(self, sample_ratings):
        matrix, _ = build_user_movie_matrix(sample_ratings)
        similarity = calculate_similarity_matrix(matrix)
        assert similarity.to_numpy().max() <= 1.0 + 1e-12
        assert similarity.to_numpy().min() >= -1.0 - 1e-12

    def test_raw_ratings_never_yield_negative_similarity(self):
        """Every MovieLens rating is non-negative, so dot(A, B) >= 0 and the
        cosine can never be negative.  The filter still guards against it."""
        rng = np.random.default_rng(7)
        values = rng.integers(1, 6, size=(12, 20)).astype(float)
        matrix = pd.DataFrame(values, index=range(1, 13), columns=range(1, 21))
        similarity = calculate_similarity_matrix(matrix)
        assert similarity.to_numpy().min() >= 0.0

    def test_conflicting_taste_scores_lower_than_shared_taste(self):
        matrix = pd.DataFrame(
            [[5.0, 4.0, 3.0], [3.0, 4.0, 5.0], [5.0, 4.0, 3.0]],
            index=[1, 2, 3],
            columns=[10, 20, 30],
        )
        similarity = calculate_similarity_matrix(matrix)
        assert similarity.loc[1, 3] == pytest.approx(1.0)
        assert similarity.loc[1, 2] < similarity.loc[1, 3]


# ----------------------------------------------------------------------
# 3. Similar user identification
# ----------------------------------------------------------------------
class TestSimilarUsers:
    def test_selected_user_is_excluded(self, sample_ratings):
        matrix, _ = build_user_movie_matrix(sample_ratings)
        similarity = calculate_similarity_matrix(matrix)
        neighbors = get_similar_users(similarity, user_id=1, top_k=10)

        assert 1 not in [n.user_id for n in neighbors]

    def test_neighbors_are_ranked_by_descending_similarity(self, sample_ratings):
        matrix, _ = build_user_movie_matrix(sample_ratings)
        similarity = calculate_similarity_matrix(matrix)
        neighbors = get_similar_users(similarity, user_id=1, top_k=10)

        scores = [n.similarity for n in neighbors]
        assert scores == sorted(scores, reverse=True)

    def test_zero_similarity_users_are_dropped(self, sample_ratings):
        """User 5 rated only movies 40/50, which overlap with no neighbour of 1."""
        matrix, _ = build_user_movie_matrix(sample_ratings)
        similarity = calculate_similarity_matrix(matrix)

        assert similarity.loc[1, 5] == pytest.approx(0.0, abs=1e-12)
        neighbors = get_similar_users(similarity, user_id=1, top_k=10)
        assert 5 not in [n.user_id for n in neighbors]

    def test_top_k_limits_the_result_size(self, sample_ratings):
        matrix, _ = build_user_movie_matrix(sample_ratings)
        similarity = calculate_similarity_matrix(matrix)
        # Users 2, 3 and 4 have a positive similarity to User 1.
        for k in (1, 2, 3, 50):
            assert len(get_similar_users(similarity, user_id=1, top_k=k)) == min(k, 3)

    def test_negative_similarity_users_are_excluded(self):
        """An anti-correlated row must never become a neighbour."""
        similarity = pd.DataFrame(
            [
                [0.0, 0.8, -0.9, 0.0],
                [0.8, 0.0, 0.3, 0.0],
                [-0.9, 0.3, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            index=[1, 2, 3, 4],
            columns=[1, 2, 3, 4],
        )
        neighbors = get_similar_users(similarity, user_id=1, top_k=10)

        assert [n.user_id for n in neighbors] == [2]
        assert all(n.similarity > 0 for n in neighbors)

    def test_closest_neighbour_of_user_1_is_user_2(self, sample_ratings):
        matrix, _ = build_user_movie_matrix(sample_ratings)
        similarity = calculate_similarity_matrix(matrix)
        assert get_similar_users(similarity, user_id=1, top_k=1)[0].user_id == 2

    def test_unknown_user_raises(self, sample_ratings):
        matrix, _ = build_user_movie_matrix(sample_ratings)
        similarity = calculate_similarity_matrix(matrix)
        with pytest.raises(InsufficientDataError):
            get_similar_users(similarity, user_id=999, top_k=5)

    def test_min_similarity_threshold_is_respected(self, sample_ratings):
        matrix, _ = build_user_movie_matrix(sample_ratings)
        similarity = calculate_similarity_matrix(matrix)
        strict = get_similar_users(similarity, user_id=1, top_k=10, min_similarity=0.99)
        loose = get_similar_users(similarity, user_id=1, top_k=10, min_similarity=1e-9)
        assert len(strict) < len(loose)


# ----------------------------------------------------------------------
# 4/5. Weighted prediction
# ----------------------------------------------------------------------
class TestWeightedPrediction:
    def test_matches_the_weighted_mean_formula(self):
        sims = [0.8, 0.5, 0.2]
        ratings = [5.0, 3.0, 4.0]
        expected = (0.8 * 5.0 + 0.5 * 3.0 + 0.2 * 4.0) / (0.8 + 0.5 + 0.2)
        assert weighted_predicted_rating(sims, ratings) == pytest.approx(expected)

    def test_uniform_similarities_reduce_to_the_plain_average(self):
        assert weighted_predicted_rating([1.0, 1.0], [2.0, 4.0]) == pytest.approx(3.0)

    def test_closer_neighbour_dominates_the_result(self):
        """A high-similarity 5* should outweigh a low-similarity 1*."""
        value = weighted_predicted_rating([0.99, 0.01], [5.0, 1.0])
        assert value == pytest.approx((0.99 * 5.0 + 0.01 * 1.0) / 1.0)
        assert value > 4.0

    def test_zero_total_similarity_returns_zero(self):
        assert weighted_predicted_rating([0.0, 0.0], [5.0, 4.0]) == 0.0

    def test_empty_inputs_return_zero(self):
        assert weighted_predicted_rating([], []) == 0.0

    def test_mismatched_lengths_return_zero(self):
        assert weighted_predicted_rating([0.5, 0.5], [4.0]) == 0.0

    def test_prediction_stays_inside_the_rating_scale(self, sample_ratings):
        matrix, mask = build_user_movie_matrix(sample_ratings)
        similarity = calculate_similarity_matrix(matrix)
        result = recommend_for_user(matrix, mask, similarity, user_id=1, top_n=10, top_k=4)
        for rec in result.recommendations:
            assert 0.5 <= rec.predicted_rating <= 5.0

    def test_only_neighbours_that_rated_the_movie_contribute(self, sample_ratings):
        """Movie 50 is rated by User 5 only, and User 5's similarity to User 1
        is 0, so it must never surface as a recommendation."""
        matrix, mask = build_user_movie_matrix(sample_ratings)
        similarity = calculate_similarity_matrix(matrix)
        result = recommend_for_user(matrix, mask, similarity, user_id=1, top_n=10, top_k=3)
        recommended = {r.movie_id for r in result.recommendations}
        assert 50 not in recommended

    def test_a_movie_rated_by_a_neighbour_is_recommended(self, sample_ratings):
        """Movie 40 is unseen by User 1 and was rated by neighbour User 3."""
        matrix, mask = build_user_movie_matrix(sample_ratings)
        similarity = calculate_similarity_matrix(matrix)
        result = recommend_for_user(matrix, mask, similarity, user_id=1, top_n=10, top_k=3)
        assert [r.movie_id for r in result.recommendations] == [40]
        assert result.recommendations[0].predicted_rating == pytest.approx(5.0)
        assert result.recommendations[0].neighbor_count == 1


# ----------------------------------------------------------------------
# 6. Recommendation generation
# ----------------------------------------------------------------------
@pytest.fixture()
def prepared(sample_ratings):
    matrix, mask = build_user_movie_matrix(sample_ratings)
    similarity = calculate_similarity_matrix(matrix)
    return matrix, mask, similarity


class TestRecommendations:
    def test_already_rated_movies_are_never_recommended(self, prepared):
        matrix, mask, similarity = prepared
        result = recommend_for_user(matrix, mask, similarity, user_id=1, top_n=10, top_k=4)

        rated = set(matrix.columns[mask.loc[1].to_numpy()])
        recommended = {r.movie_id for r in result.recommendations}
        assert rated == {10, 20, 30}
        assert recommended.isdisjoint(rated)
        assert result.recommendations  # sanity: the filter removed everything

    def test_recommendations_are_sorted_by_predicted_rating_desc(self, prepared):
        matrix, mask, similarity = prepared
        result = recommend_for_user(matrix, mask, similarity, user_id=1, top_n=10, top_k=4)

        scores = [r.predicted_rating for r in result.recommendations]
        assert scores == sorted(scores, reverse=True)

    def test_neighbour_count_is_the_secondary_sort_key(self):
        """Ties on predicted rating must be broken by more supporting neighbours."""
        ratings = pd.DataFrame(
            [
                # User 1 is the target.
                {"userId": 1, "movieId": 1, "rating": 4.0},
                {"userId": 1, "movieId": 9, "rating": 4.0},
                # Movies 100/200/300 all receive a predicted 5.0, but they are
                # supported by 3, 2 and 1 neighbours respectively.
                {"userId": 2, "movieId": 1, "rating": 4.0},
                {"userId": 2, "movieId": 100, "rating": 5.0},
                {"userId": 2, "movieId": 9, "rating": 4.0},
                {"userId": 3, "movieId": 1, "rating": 4.0},
                {"userId": 3, "movieId": 100, "rating": 5.0},
                {"userId": 3, "movieId": 200, "rating": 5.0},
                {"userId": 3, "movieId": 9, "rating": 4.0},
                {"userId": 4, "movieId": 1, "rating": 4.0},
                {"userId": 4, "movieId": 100, "rating": 5.0},
                {"userId": 4, "movieId": 200, "rating": 5.0},
                {"userId": 4, "movieId": 300, "rating": 5.0},
                {"userId": 4, "movieId": 9, "rating": 4.0},
            ]
        )
        matrix, mask = build_user_movie_matrix(ratings)
        similarity = calculate_similarity_matrix(matrix)
        result = recommend_for_user(matrix, mask, similarity, user_id=1, top_n=3, top_k=3)

        # All three predictions are 5.0, so the tie-break decides the order.
        assert [r.predicted_rating for r in result.recommendations] == [5.0, 5.0, 5.0]
        assert [r.movie_id for r in result.recommendations] == [100, 200, 300]
        assert [r.neighbor_count for r in result.recommendations] == [3, 2, 1]

    def test_top_n_is_respected(self, prepared):
        matrix, mask, similarity = prepared
        for n in (1, 2, 3):
            result = recommend_for_user(matrix, mask, similarity, user_id=1, top_n=n, top_k=4)
            assert len(result.recommendations) <= n

    def test_recommendation_payload_contains_every_required_field(self, sample_ratings, sample_movies):
        matrix, mask = build_user_movie_matrix(sample_ratings)
        similarity = calculate_similarity_matrix(matrix)
        titles = {int(r.movieId): r.title for r in sample_movies.itertuples()}
        genres = {int(r.movieId): tuple(r.genres) for r in sample_movies.itertuples()}

        result = recommend_for_user(
            matrix,
            mask,
            similarity,
            user_id=1,
            top_n=5,
            top_k=4,
            movie_titles=titles,
            movie_genres=genres,
        )
        rec = result.recommendations[0]
        assert isinstance(rec.title, str) and rec.title
        assert isinstance(rec.genres, tuple)
        assert isinstance(rec.predicted_rating, float)
        assert isinstance(rec.neighbor_count, int) and rec.neighbor_count >= 1

        payload = rec.as_dict()
        assert {"title", "genres", "predicted_rating", "neighbor_count"} <= set(payload)

    def test_neighbor_count_reflects_the_actual_voters(self, prepared):
        matrix, mask, similarity = prepared
        result = recommend_for_user(matrix, mask, similarity, user_id=1, top_n=10, top_k=4)
        for rec in result.recommendations:
            voters = int(
                ((mask[rec.movie_id]) & (mask.index.isin([2, 3, 4]))).sum()
            )
            assert rec.neighbor_count == voters

    def test_min_neighbors_filters_thin_evidence(self):
        """A movie supported by a single neighbour can be filtered out."""
        ratings = pd.DataFrame(
            [
                {"userId": 1, "movieId": 1, "rating": 4.0},
                {"userId": 2, "movieId": 1, "rating": 4.0},
                {"userId": 2, "movieId": 2, "rating": 5.0},  # 1 supporter
                {"userId": 3, "movieId": 1, "rating": 4.0},
                {"userId": 3, "movieId": 3, "rating": 5.0},  # 1 supporter
            ]
        )
        matrix, mask = build_user_movie_matrix(ratings)
        similarity = calculate_similarity_matrix(matrix)

        relaxed = recommend_for_user(matrix, mask, similarity, 1, top_n=5, top_k=2, min_neighbors=1)
        strict = recommend_for_user(matrix, mask, similarity, 1, top_n=5, top_k=2, min_neighbors=2)

        assert {r.movie_id for r in relaxed.recommendations} == {2, 3}
        assert strict.recommendations == []
        assert strict.message


# ----------------------------------------------------------------------
# 7. Graceful handling of edge cases
# ----------------------------------------------------------------------
class TestEdgeCases:
    def test_user_with_no_eligible_neighbours(self):
        """Two users with disjoint movie sets have a cosine similarity of 0."""
        ratings = pd.DataFrame(
            [
                {"userId": 1, "movieId": 10, "rating": 5.0},
                {"userId": 1, "movieId": 20, "rating": 4.0},
                {"userId": 2, "movieId": 30, "rating": 3.0},
                {"userId": 2, "movieId": 40, "rating": 2.0},
            ]
        )
        matrix, mask = build_user_movie_matrix(ratings)
        similarity = calculate_similarity_matrix(matrix)
        result = recommend_for_user(matrix, mask, similarity, user_id=1, top_n=10, top_k=10)

        assert result.recommendations == []
        assert result.message
        assert "similar" in result.message.lower()

    def test_unknown_user_returns_an_explanatory_result(self, prepared):
        matrix, mask, similarity = prepared
        result = recommend_for_user(matrix, mask, similarity, user_id=4242, top_n=10, top_k=10)

        assert result.recommendations == []
        assert "no ratings" in result.message.lower()

    def test_user_who_rated_everything_reports_no_candidates(self):
        """When every neighbour-rated movie is already seen, say so clearly."""
        ratings = pd.DataFrame(
            [
                {"userId": 1, "movieId": 10, "rating": 5.0},
                {"userId": 1, "movieId": 20, "rating": 4.0},
                {"userId": 2, "movieId": 10, "rating": 5.0},
                {"userId": 2, "movieId": 20, "rating": 4.0},
            ]
        )
        matrix, mask = build_user_movie_matrix(ratings)
        similarity = calculate_similarity_matrix(matrix)
        result = recommend_for_user(matrix, mask, similarity, user_id=1, top_n=10, top_k=10)

        assert result.neighbors, "User 2 should still be a neighbour"
        assert result.candidates_considered == 0
        assert result.recommendations == []
        assert "already been seen" in result.message

    def test_strict_threshold_can_eliminate_every_candidate(self, prepared):
        matrix, mask, similarity = prepared
        result = recommend_for_user(
            matrix, mask, similarity, user_id=1, top_n=5, top_k=3, min_neighbors=99
        )
        assert result.recommendations == []
        assert result.message

    def test_single_user_dataset_produces_no_recommendations(self):
        ratings = pd.DataFrame(
            [
                {"userId": 1, "movieId": 1, "rating": 5.0},
                {"userId": 1, "movieId": 2, "rating": 4.0},
            ]
        )
        matrix, mask = build_user_movie_matrix(ratings)
        similarity = calculate_similarity_matrix(matrix)
        result = recommend_for_user(matrix, mask, similarity, user_id=1)

        assert result.recommendations == []
        assert result.neighbors == []

    def test_missing_titles_fall_back_to_a_readable_placeholder(self, prepared):
        matrix, mask, similarity = prepared
        result = recommend_for_user(matrix, mask, similarity, user_id=1, top_n=5, top_k=3)
        assert result.recommendations
        assert all(r.title.startswith("Movie ") for r in result.recommendations)

    def test_top_n_and_top_k_are_clamped_to_at_least_one(self, prepared):
        matrix, mask, similarity = prepared
        result = recommend_for_user(matrix, mask, similarity, user_id=1, top_n=0, top_k=2)
        assert len(result.recommendations) == 1, "top_n=0 must behave like top_n=1"
        assert len(result.neighbors) == 2, "top_k=0 must behave like top_k=1"

    def test_a_single_neighbour_can_yield_no_candidates(self, prepared):
        """User 2 (the closest match of User 1) rated only movies User 1 has
        already seen, so Top-K = 1 must report zero candidates gracefully."""
        matrix, mask, similarity = prepared
        result = recommend_for_user(matrix, mask, similarity, user_id=1, top_n=10, top_k=1)
        assert result.candidates_considered == 0
        assert result.recommendations == []
        assert "already been seen" in result.message


# ----------------------------------------------------------------------
# 8. The UserBasedRecommender orchestrator (no MongoDB required)
# ----------------------------------------------------------------------
class TestRecommenderOrchestrator:
    @pytest.fixture()
    def recommender(self, sample_ratings, sample_movies):
        engine = UserBasedRecommender(db=None)  # type: ignore[arg-type]
        engine.set_data(sample_ratings, sample_movies)
        return engine

    def test_set_data_prepares_the_engine(self, recommender):
        assert recommender.is_ready()
        assert recommender.user_ids == [1, 2, 3, 4, 5]
        assert recommender.movie_ids == [10, 20, 30, 40, 50]
        assert recommender.similarity.shape == (5, 5)

    def test_similarity_matrix_is_computed_once_and_cached(self, recommender):
        first = recommender.similarity
        second = recommender.similarity
        assert first is second

    def test_titles_and_genres_are_attached(self, recommender):
        result = recommender.recommend(user_id=1, top_n=5, top_k=4)
        titles = {r.title for r in result.recommendations}
        assert titles <= {m.title for m in recommender._movies.itertuples()}

    def test_recommendations_frame_columns(self, recommender):
        frame = recommender.recommendations_frame(user_id=1, top_n=5, top_k=4)
        assert list(frame.columns) == [
            "movieId",
            "title",
            "genres_display",
            "predicted_rating",
            "neighbor_count",
        ]
        assert not frame.empty

    def test_recommendations_frame_is_empty_when_no_result(self, recommender):
        """Top-K = 1 leaves only User 2, whose movies User 1 has all seen."""
        frame = recommender.recommendations_frame(user_id=1, top_n=5, top_k=1)
        assert frame.empty
        assert list(frame.columns) == [
            "movieId",
            "title",
            "genres_display",
            "predicted_rating",
            "neighbor_count",
        ]

    def test_get_similar_users_excludes_self(self, recommender):
        neighbors = recommender.get_similar_users(1, top_k=10)
        assert 1 not in [n.user_id for n in neighbors]
        assert all(isinstance(n.rating_count, int) for n in neighbors)

    def test_neighbor_rating_counts_match_the_matrix(self, recommender):
        neighbors = recommender.get_similar_users(1, top_k=4)
        expected = recommender.mask.sum(axis=1)
        for neighbor in neighbors:
            assert neighbor.rating_count == int(expected.loc[neighbor.user_id])

    def test_recommend_never_returns_rated_movies(self, recommender):
        for user_id in recommender.user_ids:
            result = recommender.recommend(user_id, top_n=20, top_k=10)
            rated = set(
                recommender.matrix.columns[recommender.mask.loc[user_id].to_numpy()]
            )
            assert {r.movie_id for r in result.recommendations}.isdisjoint(rated)

    def test_every_user_gets_recommendations_when_data_allows(self, recommender):
        """Every user in the fixture has at least one eligible neighbour and one
        unseen movie that a neighbour has rated."""
        got = [
            u
            for u in recommender.user_ids
            if recommender.recommend(u, top_n=10, top_k=10).recommendations
        ]
        assert got == [1, 2, 3, 4, 5]

    def test_setting_new_data_invalidates_the_similarity_cache(self, recommender):
        first = recommender.similarity
        recommender.set_data(
            pd.DataFrame([{"userId": 1, "movieId": 10, "rating": 5.0}]),
            pd.DataFrame([{"movieId": 10, "title": "Only One", "genres": ["Drama"]}]),
        )
        assert recommender.similarity is not first
        assert recommender.similarity.shape == (1, 1)


# ----------------------------------------------------------------------
# 9. Numerical sanity on a larger synthetic set
# ----------------------------------------------------------------------
class TestLargerSyntheticMatrix:
    def test_identical_taste_produces_similarity_of_one(self):
        """Users 1-3 have byte-identical rating vectors, so cosine == 1.0."""
        rng = np.random.default_rng(42)
        shared = {m: float(rng.integers(3, 6)) for m in range(1, 21)}

        rows = [
            {"userId": user_id, "movieId": movie_id, "rating": rating}
            for user_id in (1, 2, 3)
            for movie_id, rating in shared.items()
        ]
        # User 4 shares the profile of movies 1-10 and additionally rates 21-25.
        rows += [
            {"userId": 4, "movieId": movie_id, "rating": rating}
            for movie_id, rating in shared.items()
            if movie_id <= 10
        ]
        rows += [
            {"userId": 4, "movieId": movie_id, "rating": 2.0}
            for movie_id in range(21, 26)
        ]

        ratings = pd.DataFrame(rows)
        matrix, mask = build_user_movie_matrix(ratings)
        similarity = calculate_similarity_matrix(matrix)

        assert similarity.loc[1, 2] == pytest.approx(1.0)
        assert similarity.loc[1, 3] == pytest.approx(1.0)
        assert 0.0 < similarity.loc[1, 4] < 1.0

        # Movies 21-25 are unseen by User 1 and were rated by neighbour User 4.
        result = recommend_for_user(matrix, mask, similarity, user_id=1, top_n=5, top_k=4)
        assert len(result.recommendations) == 5
        assert {r.movie_id for r in result.recommendations} == {21, 22, 23, 24, 25}
        for rec in result.recommendations:
            assert rec.neighbor_count == 1
            assert rec.predicted_rating == pytest.approx(2.0)
            assert not math.isnan(rec.predicted_rating)

    def test_recommendation_matches_the_hand_computed_weighted_mean(self):
        """One neighbour, so the prediction must equal that user's rating."""
        ratings = pd.DataFrame(
            [
                {"userId": 1, "movieId": 1, "rating": 5.0},
                {"userId": 1, "movieId": 2, "rating": 4.0},
                {"userId": 2, "movieId": 1, "rating": 5.0},
                {"userId": 2, "movieId": 3, "rating": 2.5},
            ]
        )
        matrix, mask = build_user_movie_matrix(ratings)
        similarity = calculate_similarity_matrix(matrix)
        result = recommend_for_user(matrix, mask, similarity, user_id=1, top_n=5, top_k=1)

        assert len(result.recommendations) == 1
        rec = result.recommendations[0]
        assert rec.movie_id == 3
        assert rec.predicted_rating == pytest.approx(2.5)
        assert rec.neighbor_count == 1
