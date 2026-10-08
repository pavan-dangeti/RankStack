"""Invariants that must hold for any ranking implementation."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from rankstack.eval.metrics import hit_rate_at_k, ndcg_at_k, ranking_metrics, reciprocal_rank
from rankstack.retrieval.base import exact_rank, mask_seen, top_k

finite_scores = st.lists(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=60,
)


@given(scores=finite_scores, target=st.integers(min_value=0, max_value=59))
@settings(max_examples=200, deadline=None)
def test_exact_rank_matches_a_brute_force_count(scores: list[float], target: int) -> None:
    array = np.asarray(scores, dtype=np.float32)
    if target >= array.size:
        return
    higher = int(np.count_nonzero(array > array[target]))
    ties_before = int(np.count_nonzero((array == array[target]) & (np.arange(array.size) < target)))
    assert exact_rank(array, target) == 1 + higher + ties_before


@given(scores=finite_scores, target=st.integers(min_value=0, max_value=59), k=st.integers(1, 20))
@settings(max_examples=200, deadline=None)
def test_rank_order_agrees_with_hit_rate(scores: list[float], target: int, k: int) -> None:
    """Check that a rank within K and a top-K membership agree, in both directions."""
    array = np.asarray(scores, dtype=np.float32)
    if target >= array.size:
        return
    rank = exact_rank(array, target)
    codes, _ = top_k(array, k)
    contained = target in set(codes.tolist())
    assert contained == (0 < rank <= k)
    assert hit_rate_at_k(rank, k) == float(contained)


@given(scores=finite_scores, k=st.integers(1, 30))
@settings(max_examples=150, deadline=None)
def test_top_k_is_sorted_and_holds_the_largest_finite_scores(scores: list[float], k: int) -> None:
    array = np.asarray(scores, dtype=np.float32)
    codes, values = top_k(array, k)
    assert len(codes) == min(k, array.size)
    assert np.all(np.diff(values) <= 0)
    assert len(set(codes.tolist())) == len(codes)
    best = np.sort(array)[::-1][: len(codes)]
    assert np.allclose(values, best, rtol=1e-5, atol=1e-5)


@given(scores=finite_scores, mask=st.lists(st.booleans(), min_size=1, max_size=60))
@settings(max_examples=100, deadline=None)
def test_masked_items_are_never_returned(scores: list[float], mask: list[bool]) -> None:
    array = np.asarray(scores, dtype=np.float32)
    if len(mask) != array.size:
        return
    masked = mask_seen(array.copy(), np.asarray(mask))
    codes, _ = top_k(masked, 10)
    assert not any(mask[code] for code in codes.tolist())


@given(ranks=st.lists(st.integers(min_value=1, max_value=10_000), min_size=1, max_size=200))
@settings(max_examples=100, deadline=None)
def test_metrics_are_monotone_in_the_rank(ranks: list[int]) -> None:
    """A worse rank can never score better on any metric."""
    ranks_array = np.asarray(ranks, dtype=np.int64)
    served = np.ones(ranks_array.size, dtype=bool)
    for k in (10, 50):
        best, worst = float(ranks_array.min()), float(ranks_array.max())
        assert hit_rate_at_k(best, k) >= hit_rate_at_k(worst, k)
        assert ndcg_at_k(best, k) >= ndcg_at_k(worst, k)
        assert reciprocal_rank(best, k) >= reciprocal_rank(worst, k)
        assert ranking_metrics(ranks_array, served, ks=(k,)).mean("hr")[f"hr@{k}"] <= 1.0


@given(scores=finite_scores)
@settings(max_examples=100, deadline=None)
def test_ranking_metrics_stay_in_range(scores: list[float]) -> None:
    array = np.asarray(scores, dtype=np.float32)
    ranks = np.array([exact_rank(array, index) for index in range(array.size)], dtype=np.int64)
    metrics = ranking_metrics(ranks, np.ones(array.size, dtype=bool), ks=(10, 20))
    for metric in ("recall", "hr", "ndcg", "mrr"):
        for value in metrics.mean(metric).values():
            assert 0.0 <= value <= 1.0
