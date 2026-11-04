"""Unit tests for the ANN benchmark's measurement and selection logic.

The sweep itself is exercised by running it; what is not exercised by any run is the code that
*chooses* the operating point and scores recall, because a wrong rule produces a confidently wrong
answer rather than an error. These tests pin those rules.
"""

from __future__ import annotations

import sys

import faiss
import numpy as np
import pytest
from scripts.bench_ann_index import (
    TIE_TOLERANCE,
    GroundTruth,
    Measurement,
    Recalls,
    _best,
    _choose,
    _index_bytes,
    _measure_recalls,
    _neighbour_contrast,
    _percentile,
)


def _point(recall: float, p99: float) -> Measurement:
    return Measurement(
        family="hnsw",
        params={},
        recalls=Recalls(at_top_k=recall, at_10=recall, tie_aware_at_10=recall),
        p50_ms=p99 / 2,
        p95_ms=p99 * 0.9,
        p99_ms=p99,
        index_bytes=1000,
        build_seconds=1.0,
    )


def test_choose_picks_the_fastest_configuration_that_reaches_the_target() -> None:
    """Latency decides among configurations that qualify; recall does not."""
    points = [_point(0.96, p99=9.0), _point(0.99, p99=12.0), _point(0.91, p99=1.0)]

    assert _choose(points, 0.95) is points[0]


def test_choose_returns_nothing_when_no_configuration_reaches_the_target() -> None:
    """A failing set has no operating point; the best of it must not be offered as one."""
    points = [_point(0.80, p99=9.0), _point(0.70, p99=1.0)]

    assert _choose(points, 0.95) is None
    assert _best(points) is points[0]


def test_choose_and_best_on_an_empty_curve_return_nothing() -> None:
    """No candidates means no claim, which is different from a default."""
    assert _choose([], 0.95) is None
    assert _best([]) is None


def test_percentile_uses_nearest_rank() -> None:
    """Latency percentiles match the convention the results files claim to use."""
    samples = [float(n) for n in range(1, 101)]

    assert _percentile(samples, 50) == 50.0
    assert _percentile(samples, 95) == 95.0
    assert _percentile(samples, 99) == 99.0
    assert _percentile(samples, 100) == 100.0


def test_percentile_on_one_sample_returns_that_sample() -> None:
    """A single measurement must not index past the end of the list."""
    assert _percentile([4.2], 99) == 4.2


def _one_hot_items(n: int) -> np.ndarray:
    """Item i scores `query[i]` against any query, so scores can be set per item directly."""
    return np.eye(n, dtype=np.float32)


def _truth_for(scores: np.ndarray, k: int) -> GroundTruth:
    order = np.argsort(-scores, kind="stable")[:k]
    return GroundTruth(ids=order[None, :].astype(np.int64), scores=scores[order][None, :])


def test_recall_counts_the_overlap_with_the_exact_result() -> None:
    """Strict recall is the fraction of exact ids recovered, at top-k and at top-10."""
    scores = np.linspace(1.0, 0.5, 30, dtype=np.float32)
    items = _one_hot_items(30)
    truth = _truth_for(scores, 20)

    perfect = _measure_recalls(truth.ids.copy(), truth, items, scores[None, :])
    assert perfect == Recalls(at_top_k=1.0, at_10=1.0, tie_aware_at_10=1.0)

    found = truth.ids.copy()
    found[0, 15:] = np.arange(25, 30)
    tail_missed = _measure_recalls(found, truth, items, scores[None, :])
    assert tail_missed.at_top_k == 15 / 20
    assert tail_missed.at_10 == 1.0


def test_tie_aware_recall_credits_an_equally_good_item_strict_recall_misses() -> None:
    """An item scoring within the tolerance of the 10th-best is a tie, not a miss."""
    scores = np.linspace(1.0, 0.5, 30, dtype=np.float32)
    scores[20] = scores[9] - TIE_TOLERANCE / 2
    truth = _truth_for(scores, 10)

    found = truth.ids.copy()
    found[0, 9] = 20
    recalls = _measure_recalls(found, truth, _one_hot_items(30), scores[None, :])
    assert recalls.at_10 == 0.9
    assert recalls.tie_aware_at_10 == 1.0


def test_tie_aware_recall_rejects_an_item_outside_the_tolerance() -> None:
    """A clearly worse item is a miss under both definitions."""
    scores = np.linspace(1.0, 0.5, 30, dtype=np.float32)
    truth = _truth_for(scores, 10)

    found = truth.ids.copy()
    found[0, 9] = 29
    recalls = _measure_recalls(found, truth, _one_hot_items(30), scores[None, :])
    assert recalls.at_10 == 0.9
    assert recalls.tie_aware_at_10 == 0.9


def test_recall_treats_faiss_padding_as_a_miss() -> None:
    """FAISS returns -1 when it finds fewer than k; that must never count as a tie hit."""
    scores = np.ones(30, dtype=np.float32)
    truth = _truth_for(scores, 10)

    found = truth.ids.copy()
    found[0, 5:] = -1
    recalls = _measure_recalls(found, truth, _one_hot_items(30), scores[None, :])
    assert recalls.at_10 == 0.5
    assert recalls.tie_aware_at_10 == 0.5


def test_neighbour_contrast_reads_ranks_one_ten_and_k() -> None:
    """The contrast summary must read the columns it names, not neighbouring ones."""
    scores = np.tile(np.linspace(1.0, 0.0, 20, dtype=np.float32), (3, 1))

    contrast = _neighbour_contrast(scores)
    assert contrast["median_score_rank_1"] == 1.0
    assert contrast["median_score_rank_10"] == round(float(scores[0, 9]), 4)
    assert contrast["median_score_rank_k"] == 0.0
    assert contrast["median_gap_rank_10_to_k"] == round(float(scores[0, 9]), 4)
    assert contrast["median_relative_gap_rank_10_to_k"] == 1.0


def test_index_bytes_counts_the_hnsw_graph() -> None:
    """A denser graph must cost more bytes; the serialised size once reported the vectors alone."""
    if sys.platform == "darwin" and "torch" in sys.modules:
        # torch and faiss bundle separate libomp copies on macOS and the process aborts when both
        # initialise; Linux CI runs this, and so does this file on its own.
        pytest.skip("faiss cannot run alongside torch's libomp on macOS")
    vectors = np.random.default_rng(0).standard_normal((2000, 16)).astype(np.float32)
    sparse = faiss.IndexHNSWFlat(16, 8, faiss.METRIC_INNER_PRODUCT)
    dense = faiss.IndexHNSWFlat(16, 64, faiss.METRIC_INNER_PRODUCT)
    sparse.add(vectors)
    dense.add(vectors)

    assert _index_bytes(sparse) > vectors.nbytes
    assert _index_bytes(dense) > _index_bytes(sparse)
