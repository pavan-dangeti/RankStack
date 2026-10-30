"""Benchmark FAISS index families on the two-tower item vectors.

The design document (section 6.3) promised two things this makes measurable: a recall-versus-latency
curve per index configuration, and a chosen operating point with its measured recall and p99.

**What recall means here.** Queries are user-tower vectors for a seeded sample of users, the
workload the index would serve, and recall is measured against exact top-200 from a brute-force
inner-product search over the same vectors. Only trained items are indexed, because the model
scores untrained ones `-inf`; an exact top-200 containing them would be a target serving discards.
That is a statement about the *index*, not about recommendation quality: the two-tower model that
produced these vectors lost its own evaluation (HR@20 0.00025), and a perfect index over its vectors
would still rank badly.

**Why recall may be low regardless of the index.** Both towers are unit-normalised, and on this
model a query's 200th neighbour is barely farther than its 10th. When neighbours are that crowded,
any approximate method swaps near-equal items in and out of the top-k. `neighbour_contrast`
records that spread so a low curve can be read as a property of the vectors, not only of FAISS.

    uv run python scripts/bench_ann_index.py --queries 200
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import faiss
import numpy as np

from rankstack.config import settings
from rankstack.data.corpus import CORPORA, DEFAULT_CORPUS, get_corpus
from rankstack.logging import configure_logging, get_logger
from rankstack.retrieval.two_tower_model import load_model

_logger = get_logger(__name__)

DEFAULT_QUERIES = 200
DEFAULT_TOP_K = 200
#: design document 6.3. Kept as written; `--quick` trims the grids for a smoke run.
HNSW_M = (16, 32, 64)
HNSW_EF_CONSTRUCTION = (100, 200, 400)
HNSW_EF_SEARCH = (32, 64, 128, 256, 512)
#: 128 dims must divide evenly by `m`; 128/16 = 8 dims per sub-quantiser, the usual sweet spot.
IVF_PQ_M = (16, 32)
IVF_NLIST = (1024, 4096)
IVF_NPROBE = (1, 4, 16, 64)
#: The operating point must reach this recall against exact search to be considered usable.
RECALL_TARGET = 0.95


class Searchable(Protocol):
    """The only part of a FAISS index this benchmark uses.

    Typing the helpers against `faiss.Index` would make them untestable without building a real
    index, and the recall arithmetic is exactly the part worth testing.
    """

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return `(distances, ids)` for the k nearest neighbours of each query."""
        ...


@dataclass(slots=True)
class Measurement:
    """One point on a recall-versus-latency curve."""

    family: str
    params: dict[str, int]
    recall_at_top_k: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    #: Bytes `faiss.write_index` puts on disk, not raw vector bytes. An HNSW graph adds roughly
    #: `M * 8` bytes per node on top of the vectors, and the hosting decision in the design
    #: document turns on exactly this number.
    index_bytes: int
    build_seconds: float

    def as_dict(self) -> dict[str, Any]:
        """Return the measurement as a JSON-serialisable mapping."""
        return {
            "family": self.family,
            "params": self.params,
            "recall_at_top_k": round(self.recall_at_top_k, 6),
            "p50_ms": round(self.p50_ms, 4),
            "p95_ms": round(self.p95_ms, 4),
            "p99_ms": round(self.p99_ms, 4),
            "index_bytes": self.index_bytes,
            "index_mib": round(self.index_bytes / (1024 * 1024), 1),
            "build_seconds": round(self.build_seconds, 1),
        }


@dataclass(slots=True)
class Curve:
    """Every measurement for one index family."""

    family: str
    points: list[Measurement] = field(default_factory=list)

    def as_list(self) -> list[dict[str, Any]]:
        """Return the curve sorted by recall, as JSON-serialisable mappings."""
        return [m.as_dict() for m in sorted(self.points, key=lambda m: m.recall_at_top_k)]


def _percentile(samples: list[float], q: float) -> float:
    """Return the q-th percentile of latency samples, nearest-rank."""
    ordered = sorted(samples)
    index = min(round(q / 100 * len(ordered)) - 1, len(ordered) - 1)
    return ordered[max(index, 0)]


def _measure_recall(index: Searchable, queries: np.ndarray, truth: np.ndarray) -> float:
    """Return the fraction of exact top-k neighbours each query's ANN top-k contains."""
    _, found = index.search(queries, truth.shape[1])
    hits = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(found, truth, strict=True))
    total: float = float(truth.shape[0] * truth.shape[1])
    return float(hits) / total


def _time_search(
    index: faiss.Index,
    queries: np.ndarray,
    top_k: int,
    repeats: int,
) -> list[float]:
    """Return per-query search latency in milliseconds, one warmed-up pass excluded."""
    index.search(queries[:1], top_k)
    samples: list[float] = []
    for _ in range(repeats):
        for query in queries:
            started = time.perf_counter()
            index.search(query[None, :], top_k)
            samples.append((time.perf_counter() - started) * 1000)
    return samples


def _index_bytes(index: faiss.Index) -> int:
    """Return the on-disk size of an index.

    `faiss.serialize_index` reported exactly the raw float32 bytes for every HNSW graph over the
    full 1.33M vectors, whatever `M` was, so the size comes from a real file instead.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "index.faiss"
        faiss.write_index(index, str(path))
        return path.stat().st_size


def _exact_index(
    vectors: np.ndarray, queries: np.ndarray, top_k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact top-k `(scores, ids)` from brute-force inner-product search."""
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    scores, ids = index.search(queries, top_k)
    return scores, ids


def _neighbour_contrast(scores: np.ndarray) -> dict[str, float]:
    """Summarise how far apart the exact neighbours are, as median scores at ranks 1, 10 and k."""
    return {
        "median_score_rank_1": round(float(np.median(scores[:, 0])), 4),
        "median_score_rank_10": round(float(np.median(scores[:, min(9, scores.shape[1] - 1)])), 4),
        "median_score_rank_k": round(float(np.median(scores[:, -1])), 4),
    }


def _build_hnsw(
    vectors: np.ndarray,
    queries: np.ndarray,
    truth: np.ndarray,
    m: int,
    ef_construction: int,
    top_k: int,
    ef_search_values: tuple[int, ...],
) -> list[Measurement]:
    """Build one HNSW graph and sweep its search width."""
    started = time.perf_counter()
    index = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(vectors)
    build_seconds = time.perf_counter() - started

    out: list[Measurement] = []
    for ef_search in ef_search_values:
        index.hnsw.efSearch = ef_search
        samples = _time_search(index, queries, top_k, repeats=1)
        out.append(
            Measurement(
                family="hnsw",
                params={"M": m, "ef_construction": ef_construction, "ef_search": ef_search},
                recall_at_top_k=_measure_recall(index, queries, truth),
                p50_ms=_percentile(samples, 50),
                p95_ms=_percentile(samples, 95),
                p99_ms=_percentile(samples, 99),
                index_bytes=_index_bytes(index),
                build_seconds=build_seconds,
            )
        )
        _logger.info(
            "ann.hnsw_point",
            extra={
                "M": m,
                "ef_construction": ef_construction,
                "ef_search": ef_search,
                "recall": out[-1].recall_at_top_k,
            },
        )
    return out


def _build_ivfpq(
    vectors: np.ndarray,
    queries: np.ndarray,
    truth: np.ndarray,
    nlist: int,
    m: int,
    top_k: int,
    nprobe_values: tuple[int, ...],
) -> list[Measurement]:
    """Train one IVF-PQ index and sweep how many lists it probes."""
    quantizer = faiss.IndexFlatIP(vectors.shape[1])
    index = faiss.IndexIVFPQ(quantizer, vectors.shape[1], nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
    index.train(vectors)
    started = time.perf_counter()
    index.add(vectors)
    build_seconds = time.perf_counter() - started

    out: list[Measurement] = []
    for nprobe in nprobe_values:
        index.nprobe = nprobe
        samples = _time_search(index, queries, top_k, repeats=1)
        out.append(
            Measurement(
                family="ivfpq",
                params={"nlist": nlist, "m": m, "nprobe": nprobe},
                recall_at_top_k=_measure_recall(index, queries, truth),
                p50_ms=_percentile(samples, 50),
                p95_ms=_percentile(samples, 95),
                p99_ms=_percentile(samples, 99),
                index_bytes=_index_bytes(index),
                build_seconds=build_seconds,
            )
        )
        _logger.info(
            "ann.ivfpq_point",
            extra={
                "nlist": nlist,
                "m": m,
                "nprobe": nprobe,
                "recall": out[-1].recall_at_top_k,
            },
        )
    return out


def _choose(points: list[Measurement], target: float) -> Measurement | None:
    """Pick the lowest-p99 configuration that reaches the recall target, else the best recall.

    Lowest latency *among configurations that qualify*, rather than the best recall overall: an
    index nobody waits on is worth more than one that is marginally more accurate and 10x slower,
    and a target that cannot be met should surface as the best available rather than as silence.
    """
    qualifying = [p for p in points if p.recall_at_top_k >= target]
    if qualifying:
        return min(qualifying, key=lambda p: p.p99_ms)
    return max(points, key=lambda p: p.recall_at_top_k) if points else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, choices=sorted(CORPORA))
    parser.add_argument("--queries", type=int, default=DEFAULT_QUERIES)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--recall-target", type=float, default=RECALL_TARGET)
    parser.add_argument("--quick", action="store_true", help="Trim the grids for a smoke run.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the sweep and write `results/ann_index_<corpus>.json`."""
    args = parse_args(argv)
    configure_logging()
    corpus = get_corpus(args.corpus)
    root = settings.artifact_root / corpus.name / "two_tower.npz"
    if not root.exists():
        print(
            f"error: no two-tower vectors at {root}; run scripts/train_two_tower.py",
            file=sys.stderr,
        )
        return 1

    model = load_model(root)
    trained = (
        np.ones(model.n_items(), dtype=bool)
        if model.trained_items is None
        else np.asarray(model.trained_items, dtype=bool)
    )
    vectors = np.ascontiguousarray(model.item_embeddings[trained], dtype=np.float32)

    rng = np.random.default_rng(settings.seed)
    servable = np.flatnonzero(np.diff(model.history_indptr) >= 1)
    user_codes = rng.choice(servable, size=args.queries, replace=False)
    queries = np.ascontiguousarray(model.user_vectors(user_codes))

    started = time.perf_counter()
    exact_scores, truth = _exact_index(vectors, queries, args.top_k)
    exact_seconds = time.perf_counter() - started
    contrast = _neighbour_contrast(exact_scores)
    _logger.info(
        "ann.ground_truth",
        extra={
            "items": vectors.shape[0],
            "queries": args.queries,
            "top_k": args.top_k,
            "exact_seconds": round(exact_seconds, 1),
            **contrast,
        },
    )

    m_values = HNSW_M if not args.quick else (16, 64)
    efc_values = HNSW_EF_CONSTRUCTION if not args.quick else (100,)
    pq_m = IVF_PQ_M if not args.quick else (16,)
    nlists = IVF_NLIST if not args.quick else (1024,)

    points: list[Measurement] = []
    for m in m_values:
        for efc in efc_values:
            points.extend(_build_hnsw(vectors, queries, truth, m, efc, args.top_k, HNSW_EF_SEARCH))
    for nlist in nlists:
        for m in pq_m:
            points.extend(_build_ivfpq(vectors, queries, truth, nlist, m, args.top_k, IVF_NPROBE))

    hnsw = Curve("hnsw", [p for p in points if p.family == "hnsw"])
    ivfpq = Curve("ivfpq", [p for p in points if p.family == "ivfpq"])
    chosen_hnsw = _choose(hnsw.points, args.recall_target)
    chosen_ivfpq = _choose(ivfpq.points, args.recall_target)

    payload: dict[str, Any] = {
        "corpus": corpus.name,
        "command": (
            f"uv run python scripts/bench_ann_index.py --queries {args.queries}"
            f" --top-k {args.top_k} --recall-target {args.recall_target}"
        ),
        "quick": args.quick,
        "items_indexed": int(vectors.shape[0]),
        "dimensions": int(vectors.shape[1]),
        "queries": args.queries,
        "top_k": args.top_k,
        "recall_definition": (
            "Fraction of exact top-k neighbours recovered, against brute-force inner-product "
            "search over the trained item vectors, queried with user-tower vectors."
        ),
        "seed": settings.seed,
        "float32_bytes": int(vectors.shape[0] * vectors.shape[1] * 4),
        "exact_search_seconds": round(exact_seconds, 1),
        "items_untrained_excluded": int((~trained).sum()),
        "neighbour_contrast": contrast,
        "recall_target": args.recall_target,
        "curves": {"hnsw": hnsw.as_list(), "ivfpq": ivfpq.as_list()},
        "chosen": {
            "rule": (
                f"Lowest p99 latency among configurations reaching recall >= {args.recall_target}; "
                "if none qualifies, the highest recall available."
            ),
            "hnsw": chosen_hnsw.as_dict() if chosen_hnsw else None,
            "ivfpq": chosen_ivfpq.as_dict() if chosen_ivfpq else None,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    out = settings.results_root / f"ann_index_{corpus.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _print(payload, out)
    return 0


def _print(payload: dict[str, Any], out: object) -> None:
    """Print the chosen operating points and where the result file landed."""
    print(
        f"\n=== {payload['items_indexed']} vectors, {payload['dimensions']}d, "
        f"{payload['queries']} queries, top-{payload['top_k']} ==="
    )
    for family, chosen in payload["chosen"].items():
        if family == "rule" or chosen is None:
            continue
        print(
            f"\n  chosen {family}: {chosen['params']}"
            f"\n    recall {chosen['recall_at_top_k']:.4f}"
            f"  p50 {chosen['p50_ms']:.3f}ms  p95 {chosen['p95_ms']:.3f}ms"
            f"  p99 {chosen['p99_ms']:.3f}ms  index {chosen['index_mib']} MiB"
        )
    print(f"\n  float32 vectors: {payload['float32_bytes'] / (1024 * 1024):.0f} MiB")
    contrast = payload["neighbour_contrast"]
    print(
        f"  median exact score at rank 1 / 10 / {payload['top_k']}: "
        f"{contrast['median_score_rank_1']} / {contrast['median_score_rank_10']}"
        f" / {contrast['median_score_rank_k']}"
    )
    print(f"  wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
