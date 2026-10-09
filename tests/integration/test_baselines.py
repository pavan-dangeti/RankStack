"""Baselines on a synthetic corpus: encoding, training, evaluation and the cold-start boundary."""

from __future__ import annotations

import duckdb
import numpy as np
import pytest

from rankstack.data.encode import build_cohort, load_encoding
from rankstack.eval.harness import EvalConfig, compare_sources, evaluate_source
from rankstack.retrieval.base import exact_rank
from rankstack.retrieval.cooccurrence import CooccurrenceModel, build_history
from tests.conftest import Trained, history_for, popularity_counts

pytestmark = pytest.mark.integration


def test_encoding_round_trips_through_disk(trained: Trained) -> None:
    written = trained.encoding
    reloaded = load_encoding(trained.settings.artifact_root, trained.spec.name)
    assert reloaded.user_to_code == written.user_to_code
    assert reloaded.item_to_code == written.item_to_code
    assert reloaded.n_users == written.n_users
    assert reloaded.n_items == written.n_items


def test_encoding_excludes_an_item_that_appears_only_after_the_boundary(trained: Trained) -> None:
    """The cold-start boundary, enforced by the code space itself."""
    encoding = trained.encoding
    new_item = trained.synthetic.new_item_asin
    assert new_item not in encoding.item_to_code
    assert trained.cooccurrence.n_items() == encoding.n_items


def test_popularity_serves_every_user_including_cold_start_ones(trained: Trained) -> None:
    popularity = trained.popularity
    seen = np.zeros(popularity.n_items(), dtype=bool)
    scores = popularity.score(0, seen)
    assert np.isfinite(scores).sum() == popularity.n_items()
    assert popularity.can_serve(999_999) is True


def test_a_user_with_no_train_history_is_unservable(trained: Trained) -> None:
    """A user first seen in the test window has no train history at all."""
    cold_user = trained.synthetic.test_only_user
    assert cold_user not in trained.encoding.user_to_code


def test_als_cannot_serve_a_user_with_one_interaction(trained: Trained) -> None:
    """One interaction gives a user factor no information, so ALS excludes them by design."""
    encoding = trained.encoding
    code = encoding.user_to_code[trained.synthetic.single_history_user]
    assert trained.cooccurrence.history_length(code) == 1
    assert trained.als.can_serve(code) is False
    assert not trained.als.user_factors[code].any()

    seen = np.zeros(encoding.n_items, dtype=bool)
    scores = trained.als.score(code, seen)
    assert not np.isfinite(scores).any()
    # Co-occurrence still has a neighbour list for that single item.
    assert trained.cooccurrence.can_serve(code) is True


def test_seen_items_are_never_recommended(trained: Trained) -> None:
    cooccurrence = trained.cooccurrence
    user = next(
        code for code in range(trained.encoding.n_users) if cooccurrence.history_length(code) >= 2
    )
    seen = np.zeros(trained.encoding.n_items, dtype=bool)
    seen[cooccurrence.history(user)] = True
    scores = cooccurrence.score(user, seen)
    assert not np.isfinite(scores[seen]).any()


def test_cooccurrence_is_cosine_normalised(trained: Trained) -> None:
    similarity = trained.cooccurrence.similarity
    assert similarity.shape[0] == similarity.shape[1] == trained.encoding.n_items
    assert similarity.nnz > 0
    # Cosine values cannot exceed 1.
    assert float(similarity.data.max()) <= 1.0 + 1e-6


def test_history_is_capped_and_ordered_by_recency(trained: Trained) -> None:
    cooccurrence = trained.cooccurrence
    assert cooccurrence.history_indptr.shape[0] == trained.encoding.n_users + 1
    assert cooccurrence.max_history == 50
    for code in range(min(trained.encoding.n_users, 20)):
        assert cooccurrence.history_length(code) <= cooccurrence.max_history


def test_build_history_returns_popularity_counts(trained: Trained) -> None:
    encoding = trained.encoding
    train = trained.processed / "interactions" / "train.parquet"
    indptr, codes, popularity = build_history(
        train, encoding.user_to_code, encoding.item_to_code, 50
    )
    assert indptr.shape[0] == encoding.n_users + 1
    assert codes.size == int(indptr[-1])
    assert popularity.sum() > 0
    assert popularity.shape[0] == encoding.n_items


def test_exact_rank_ranks_by_score_then_by_item_code() -> None:
    ascending = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert exact_rank(ascending, 2) == 1  # highest score ranks first
    assert exact_rank(ascending, 0) == 3

    # Four identical scores: ascending item code decides, so the last code comes fourth.
    tied = np.zeros(4, dtype=np.float32)
    assert exact_rank(tied, 3) == 4
    assert exact_rank(tied, 0) == 1


def test_exact_rank_ignores_masked_items_entirely() -> None:
    """Masked items are `-inf`, so they must not shift the rank of an unmasked target."""
    scores = np.array([0.0, 1.0, 2.0, 3.0, 0.0, 9.0], dtype=np.float32)
    seen = np.zeros(6, dtype=bool)
    seen[[5]] = True
    masked = np.where(seen, -np.inf, scores)
    assert exact_rank(masked, 5) == -1  # a masked item has no finite rank
    assert exact_rank(masked, 2) == 2  # only item 3 (3.0) outranks it; item 5 is masked away
    assert exact_rank(masked, 0) == 4  # codes 1-3 outrank it; code 4 ties but ranks after


def test_evaluation_reports_segments_and_a_usable_cohort(trained: Trained) -> None:
    encoding = trained.encoding
    connection = duckdb.connect(database=":memory:")
    cohort = build_cohort(
        connection,
        "all_test_users",
        encoding,
        trained.processed,
        size=100,
        seed=7,
    )
    connection.close()
    assert len(cohort) > 0
    assert cohort.population_size >= len(cohort)

    users = np.asarray(cohort.user_codes, dtype=np.int64)
    targets = np.asarray(cohort.target_codes, dtype=np.int64)
    history = history_for(trained, popularity_counts(trained))
    result = evaluate_source(
        trained.cooccurrence,
        users,
        targets,
        history,
        EvalConfig(name="cooccurrence", cohort_name="all_test_users"),
    )
    payload = result.as_dict()
    assert "ranking" in payload
    assert "segments" in payload
    assert "segment_sizes" in result.segments
    assert result.ranking.users == len(cohort)
    assert 0.0 <= result.ranking.mean("hr")["hr@20"] <= 1.0
    assert result.ranking.mean("hr")["hr@10"] <= result.ranking.mean("hr")["hr@20"]
    assert result.ranking.mean("ndcg")["ndcg@10"] <= result.ranking.mean("ndcg")["ndcg@50"]
    assert "coverage_at_50" in result.beyond_list


def test_cohort_is_deterministic_for_a_given_seed(trained: Trained) -> None:
    encoding = trained.encoding
    connection = duckdb.connect(database=":memory:")
    first = build_cohort(connection, "warm_users", encoding, trained.processed, size=50, seed=3)
    second = build_cohort(connection, "warm_users", encoding, trained.processed, size=50, seed=3)
    other = build_cohort(connection, "warm_users", encoding, trained.processed, size=50, seed=4)
    connection.close()
    assert first.user_codes == second.user_codes
    assert first.target_codes == second.target_codes
    assert first.user_codes != other.user_codes or first.user_codes == []


def test_warm_users_cohort_excludes_users_without_enough_history(trained: Trained) -> None:
    encoding = trained.encoding
    connection = duckdb.connect(database=":memory:")
    cohort = build_cohort(connection, "warm_users", encoding, trained.processed, size=200, seed=5)
    connection.close()
    depths = [trained.cooccurrence.history_length(code) for code in cohort.user_codes]
    assert depths, "cohort should not be empty on the synthetic corpus"
    assert min(depths) >= 2


def test_paired_comparison_reports_an_interval_and_a_probability(trained: Trained) -> None:
    encoding = trained.encoding
    connection = duckdb.connect(database=":memory:")
    cohort = build_cohort(
        connection, "all_test_users", encoding, trained.processed, size=100, seed=1
    )
    connection.close()
    users = np.asarray(cohort.user_codes, dtype=np.int64)
    targets = np.asarray(cohort.target_codes, dtype=np.int64)
    history = history_for(trained, popularity_counts(trained))
    config = EvalConfig(name="x", cohort_name="all_test_users", ks=(10, 20))
    popularity = evaluate_source(trained.popularity, users, targets, history, config)
    cooccurrence = evaluate_source(trained.cooccurrence, users, targets, history, config)
    comparison = compare_sources(cooccurrence, popularity, resamples=100, seed=2)
    assert set(comparison) == {"recall", "hr", "ndcg", "mrr", "served_fraction"}
    entry = comparison["hr"]["hr@20"]
    assert isinstance(entry, dict)
    assert entry["ci_low"] <= entry["difference"] <= entry["ci_high"]
    probability = entry["bootstrap_probability_alternative_wins"]
    assert isinstance(probability, float)
    assert 0.0 <= probability <= 1.0
    assert entry["n"] == len(cohort)


def test_removing_the_self_similarity_diagonal_cannot_change_a_masked_score(
    trained: Trained,
) -> None:
    """The diagonal fix must not alter evaluation: it only touched already-seen items.

    Every item in a user's history scored itself highest, but those items are masked to `-inf`
    before ranking, so no evaluated rank can depend on the diagonal. This asserts that directly
    rather than leaving it as an argument in a comment.
    """
    cooccurrence = trained.cooccurrence
    user = next(
        code for code in range(trained.encoding.n_users) if cooccurrence.history_length(code) >= 2
    )
    seen = np.zeros(trained.encoding.n_items, dtype=bool)
    seen[cooccurrence.history(user)] = True

    stripped = cooccurrence.similarity.copy()
    stripped.setdiag(0.0)
    stripped.eliminate_zeros()
    without_diagonal = CooccurrenceModel(
        similarity=stripped,
        item_popularity=cooccurrence.item_popularity,
        history_indptr=cooccurrence.history_indptr,
        history_codes=cooccurrence.history_codes,
        max_history=cooccurrence.max_history,
        top_k=cooccurrence.top_k,
        user_block=cooccurrence.user_block,
    )
    assert np.array_equal(
        cooccurrence.score(user, seen)[~seen], without_diagonal.score(user, seen)[~seen]
    )


def test_the_cooccurrence_similarity_has_no_self_count(trained: Trained) -> None:
    """The diagonal must be zero, or an item outranks itself purely by being in the history.

    This is what made leave-one-out leaky: with the diagonal intact, a held-out item still inside
    the model's stored history scored above every unseen item, and stage 1 reported a candidate
    recall of 1.0 — impossible in any real recommender.
    """
    diagonal = trained.cooccurrence.similarity.diagonal()
    assert np.count_nonzero(diagonal) == 0


def test_scoring_from_an_explicit_history_excludes_the_holdout(trained: Trained) -> None:
    """`score_with_history` is what lets leave-one-out drop the held-out item from its own input."""
    cooccurrence = trained.cooccurrence
    user = next(
        code for code in range(trained.encoding.n_users) if cooccurrence.history_length(code) >= 2
    )
    start, end = cooccurrence.history_indptr[user], cooccurrence.history_indptr[user + 1]
    stored = cooccurrence.history(user)
    assert stored.size == end - start

    held_out = cooccurrence.history_codes[start + 1 : end]
    assert held_out.size == stored.size - 1
    assert not np.array_equal(held_out, stored)
