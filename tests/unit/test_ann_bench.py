"""Unit tests for the ANN benchmark's measurement and selection logic.

The sweep itself is exercised by running it; what is not exercised by any run is the code that
*chooses* the operating point, because a wrong rule produces a confidently wrong answer rather than
an error. These tests pin that rule, plus the percentile and recall helpers it depends on.
"""

from __future__ import annotations

import sys

import faiss
import numpy as np
import pytest
from scripts.bench_ann_index import (
    Measurement,
    _choose,
    _index_bytes,
    _measure_recall,
    _neighbour_contrast,
    _percentile,
)


def _point(family: str, recall: float, p99: float, index_bytes: int = 1000) -> Measurement:
    return Measurement(
        family=family,
        params={},
        recall_at_top_k=recall,
        p50_ms=p99 / 2,
        p95_ms=p99 * 0.9,
        p99_ms=p99,
        index_bytes=index_bytes,
        build_seconds=1.0,
    )


def test_choose_picks_the_fastest_configuration_that_reaches_the_target() -> None:
    """Latency decides among configurations that qualify; recall does not."""
    points = [
        _point("hnsw", 0.96, p99=9.0),
        _point("hnsw", 0.99, p99=12.0),
        _point("hnsw", 0.91, p99=1.0),
    ]

    assert _choose(points, 0.95) is points[0]


def test_choose_falls_back_to_best_recall_when_nothing_reaches_the_target() -> None:
    """An unreachable target must surface the best available, not silence or a wrong winner."""
    points = [_point("hnsw", 0.80, p99=9.0), _point("hnsw", 0.70, p99=1.0)]

    chosen = _choose(points, 0.95)
    assert chosen is points[0]
    assert chosen is not None


def test_choose_on_an_empty_curve_returns_nothing() -> None:
    """No candidates means no claim, which is different from a default."""
    assert _choose([], 0.95) is None


def test_choose_excludes_configurations_below_the_target_however_fast() -> None:
    """A very fast index that misses the target must not win on latency alone."""
    points = [_point("hnsw", 0.40, p99=0.1), _point("hnsw", 0.97, p99=5.0)]

    chosen = _choose(points, 0.95)
    assert chosen is not None
    assert chosen.recall_at_top_k == 0.97


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
    assert _percentile([4.2], 50) == 4.2


def test_measure_recall_counts_the_overlap_with_the_exact_result() -> None:
    """Recall is the fraction of exact neighbours recovered, not a rank-based proxy."""

    class _Stub:
        def __init__(self, found: np.ndarray) -> None:
            self._found = found

        def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
            del queries
            return np.zeros((len(self._found), k), dtype=np.float32), self._found

    truth = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)
    perfect = _Stub(np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64))
    half = _Stub(np.array([[0, 1, 90, 91], [4, 5, 6, 7]], dtype=np.int64))

    assert _measure_recall(perfect, np.zeros((2, 2), dtype=np.float32), truth) == 1.0
    assert _measure_recall(half, np.zeros((2, 2), dtype=np.float32), truth) == 0.75


def test_measure_recall_ignores_extra_results_outside_the_exact_set() -> None:
    """A neighbour beyond top-k is not a miss and not a hit; recall is over the exact set only."""

    class _Stub:
        def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
            del queries
            return np.zeros((1, k), dtype=np.float32), np.array([[0, 1, 2, 3, 99]], dtype=np.int64)

    truth = np.array([[0, 1, 2, 3]], dtype=np.int64)
    assert _measure_recall(_Stub(), np.zeros((1, 2), dtype=np.float32), truth) == 1.0


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


def test_neighbour_contrast_reads_ranks_one_ten_and_k() -> None:
    """The contrast summary must read the columns it names, not neighbouring ones."""
    scores = np.tile(np.linspace(1.0, 0.0, 20, dtype=np.float32), (3, 1))

    contrast = _neighbour_contrast(scores)
    assert contrast["median_score_rank_1"] == 1.0
    assert contrast["median_score_rank_10"] == round(float(scores[0, 9]), 4)
    assert contrast["median_score_rank_k"] == 0.0
