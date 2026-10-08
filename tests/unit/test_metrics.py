"""Metric definitions, checked against hand-computed values."""

from __future__ import annotations

import numpy as np
import pytest

from rankstack.eval.metrics import (
    coverage,
    gini,
    hit_rate_at_k,
    intra_list_diversity,
    ndcg_at_k,
    new_item_share,
    novelty,
    ranking_metrics,
    reciprocal_rank,
    slice_metrics,
)


def test_hit_rate_uses_a_one_based_rank() -> None:
    assert hit_rate_at_k(1, 10) == 1.0
    assert hit_rate_at_k(10, 10) == 1.0
    assert hit_rate_at_k(11, 10) == 0.0
    assert hit_rate_at_k(0, 10) == 0.0  # 0 means "not served"
    assert hit_rate_at_k(-1, 10) == 0.0


def test_ndcg_matches_the_binary_relevance_formula() -> None:
    assert ndcg_at_k(1, 10) == 1.0
    assert ndcg_at_k(2, 10) == 1 / np.log2(3)
    assert ndcg_at_k(5, 10) == 1 / np.log2(6)
    assert ndcg_at_k(11, 10) == 0.0


def test_reciprocal_rank_is_zero_beyond_k() -> None:
    assert reciprocal_rank(3, 10) == 1 / 3
    assert reciprocal_rank(10, 10) == 0.1
    assert reciprocal_rank(11, 10) == 0.0


def test_ranking_metrics_report_the_served_fraction() -> None:
    ranks = np.array([1, 5, -1, 100])
    served = np.array([True, True, False, True])
    metrics = ranking_metrics(ranks, served, ks=(10, 20))
    payload = metrics.as_dict()
    assert payload["users"] == 4
    assert payload["served"] == 3
    assert payload["served_fraction"] == 0.75
    assert payload["hr@10"] == 0.5
    assert payload["hr@20"] == 0.5  # rank 100 is still outside
    assert payload["ndcg@10"] == pytest.approx((1.0 + 1 / np.log2(6)) / 4, abs=1e-6)


def test_unserved_users_count_as_zeros_not_as_missing() -> None:
    ranks = np.array([-1, -1])
    metrics = ranking_metrics(ranks, np.array([False, False]), ks=(10,))
    assert metrics.mean("hr")["hr@10"] == 0.0
    assert metrics.served_fraction == 0.0


def test_slice_metrics_carry_their_own_size() -> None:
    ranks = np.array([1, 5, 100, -1])
    metrics = ranking_metrics(ranks, np.array([True] * 4), ks=(10,))
    both_hits = slice_metrics(metrics, np.array([True, True, False, False]))
    assert both_hits["n"] == 2
    assert both_hits["hr@10"] == 1.0

    one_hit = slice_metrics(metrics, np.array([False, True, True, False]))
    assert one_hit["n"] == 2
    assert one_hit["hr@10"] == 0.5
    assert slice_metrics(metrics, np.array([False] * 4))["n"] == 0


def test_gini_is_zero_when_uniform_and_maximal_when_one_item_holds_everything() -> None:
    assert gini(np.array([10.0, 10.0, 10.0])) == 0.0
    # (n - 1) / n is the maximum reachable value: one item holding all the mass.
    assert gini(np.array([0.0, 0.0, 100.0])) == pytest.approx(2 / 3, abs=1e-9)
    assert gini(np.array([0.0, 0.0, 0.0])) == 0.0
    assert gini(np.array([])) == 0.0


def test_coverage_counts_distinct_items_across_users() -> None:
    recommendations = {1: np.array([0, 1]), 2: np.array([1, 2]), 3: np.array([], dtype=np.int64)}
    assert coverage(recommendations, 10) == 0.3
    assert coverage({}, 10) == 0.0


def test_novelty_is_higher_for_rare_items() -> None:
    popularity = np.array([100.0, 1.0])
    common_only = {1: np.array([0])}
    rare_only = {1: np.array([1])}
    assert novelty(popularity, rare_only, 101) > novelty(popularity, common_only, 101)


def test_new_item_share_counts_only_unseen_items() -> None:
    popularity = np.array([0.0, 5.0])
    recommendations = {1: np.array([0, 1])}
    assert new_item_share(recommendations, (popularity <= 0).astype(float)) == 0.5


def test_intra_list_diversity_is_zero_for_identical_vectors() -> None:
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    assert intra_list_diversity({1: np.array([0, 1])}, embeddings) == 0.0


def test_intra_list_diversity_is_one_for_orthogonal_vectors() -> None:
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert intra_list_diversity({1: np.array([0, 1])}, embeddings) == 1.0


def test_intra_list_diversity_skips_lists_that_are_too_short() -> None:
    embeddings = np.eye(3, dtype=np.float32)
    assert intra_list_diversity({1: np.array([0])}, embeddings) == 0.0
