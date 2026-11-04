"""Benchmark FAISS index families on retrieval vectors.

The design document (section 6.3) promised two things this makes measurable: a recall-versus-latency
curve per index configuration, and a chosen operating point with its measured recall and p99.

**Two vector sets, one protocol.** `--vectors als` (the default) benchmarks the ALS item factors
queried with real ALS user factors; `--vectors two_tower` benchmarks the two-tower item tower
queried with user-tower vectors. ALS is the primary result because it is a model that works on
this corpus; the two-tower lost its own evaluation (HR@20 0.00025) and is kept as the failure case
that shows what an index can and cannot recover from badly spread vectors.

**What recall means here.** Queries are user vectors for a seeded sample of servable users, the
workload the index would serve, and recall is measured against exact top-k from a brute-force
inner-product search over the same vectors, the score both models rank by. Only trained items are
indexed: the two-tower scores untrained items `-inf` and ALS leaves them as zero rows, so an exact
top-k containing them would be a target serving discards. Seen items are not masked; that is a
serving rule, not a property of the index.

**Strict and tie-aware recall.** Strict recall counts an item only if it is in the exact top-k by
id. When many items score within a hair of the k-th best, an index can return an equally good item
the exact search happened to rank 11th, and strict recall calls that a miss. Tie-aware recall@10
counts a returned item as a hit if its exact score is at least the true 10th-best score minus
`TIE_TOLERANCE`. The gap between the two numbers is how much of the miss rate is ties rather than
bad neighbours, and `neighbour_contrast` records how crowded the exact neighbours are.

**The MIPS-to-L2 reduction.** ALS item factors vary in length by four orders of magnitude, and a
graph or cluster index built directly on inner product, which is not a distance, routes badly over
them. By default every approximate index is built on transformed vectors instead: each item gains
one coordinate `sqrt(max_norm^2 - norm^2)` so all items have the same length, and each query gains
a zero, which makes L2 order on the transformed vectors identical to inner-product order on the
originals. Ground truth and every recall figure stay on the original vectors and exact inner
product, so the transform can change how well the index routes but never what counts as right.
`--raw-inner-product` builds on the original vectors with the inner-product metric, the ablation.
For the unit-norm two-tower vectors the extra coordinate is zero up to float rounding.

**Baselines inside the sweep.** `flat` is exact `IndexFlatIP` on the original vectors: recall 1.0
by construction, so its latency and size are what every approximate index has to beat. `ivfflat`
uses the same clustering as `ivfpq` without quantisation, so the gap between the two separates
recall lost to cluster routing from recall lost to compression.

    uv run python scripts/bench_ann_index.py --vectors als --queries 200
    uv run python scripts/bench_ann_index.py --vectors als --queries 200 --raw-inner-product
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from rankstack.config import settings
from rankstack.data.corpus import CORPORA, DEFAULT_CORPUS, get_corpus
from rankstack.logging import configure_logging, get_logger
from rankstack.retrieval import als, two_tower_model

_logger = get_logger(__name__)

DEFAULT_QUERIES = 200
DEFAULT_TOP_K = 200
#: design document 6.3. Kept as written; `--quick` trims the grids for a smoke run.
HNSW_M = (16, 32, 64)
HNSW_EF_CONSTRUCTION = (100, 200, 400)
HNSW_EF_SEARCH = (32, 64, 128, 256, 512, 1024)
IVF_PQ_M = (16, 32)
#: PQ splits a vector into `m` equal sub-vectors, so IVF-PQ input is zero-padded to a multiple of
#: the largest `m`; the reduction's extra coordinate makes the width odd. Zero columns change no
#: distance and no inner product.
PQ_WIDTH_MULTIPLE = 32
IVF_NLIST = (1024, 4096)
IVF_NPROBE = (1, 4, 16, 64)
#: The operating point must reach this recall against exact search to be considered usable.
RECALL_TARGET = 0.95
#: Short-list depth for the strict and tie-aware recall pair.
SHORT_K = 10
#: Absolute score slack for a tie. It is only comparable across vector sets whose scores are of
#: similar scale, which `neighbour_contrast` records so a reader can check.
TIE_TOLERANCE = 1e-3
VECTOR_SETS = ("als", "two_tower")
FAMILIES = ("flat", "hnsw", "ivfflat", "ivfpq")
#: Every latency figure is one query per `search` call, the shape of a single API request.
LATENCY_BATCH_SIZE = 1
#: Run in a fresh interpreter so the measured growth is the index alone, not whatever this process
#: allocated and freed before. `ru_maxrss` is bytes on macOS and kilobytes on Linux.
_MEMORY_PROBE = """
import resource, sys
import faiss
scale = 1 if sys.platform == "darwin" else 1024
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
index = faiss.read_index(sys.argv[1])
print((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before) * scale)
"""


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """Exact top-k for each query: ids and their inner-product scores, best first."""

    ids: np.ndarray
    scores: np.ndarray


@dataclass(frozen=True, slots=True)
class Recalls:
    """The three recall figures one search produces."""

    at_top_k: float
    at_10: float
    tie_aware_at_10: float


@dataclass(frozen=True, slots=True)
class Footprint:
    """How much an index costs to keep: on disk, and in memory once loaded."""

    #: Bytes `faiss.write_index` puts on disk. `faiss.serialize_index` reported the raw vector
    #: bytes for every HNSW graph at 1.33M vectors whatever `M` was, so this comes from a file.
    disk_bytes: int
    #: Peak resident-memory growth from loading that file in a fresh process: the RAM a server
    #: holding this index would need for it.
    memory_bytes: int


@dataclass(slots=True)
class Measurement:
    """One point on a recall-versus-latency curve."""

    family: str
    params: dict[str, int]
    recalls: Recalls
    p50_ms: float
    p95_ms: float
    p99_ms: float
    footprint: Footprint
    build_seconds: float
    #: Thread count FAISS was allowed while the latency samples were taken.
    faiss_threads: int

    def as_dict(self) -> dict[str, Any]:
        """Return the measurement as a JSON-serialisable mapping."""
        return {
            "family": self.family,
            "params": self.params,
            "recall_at_top_k": round(self.recalls.at_top_k, 6),
            "recall_at_10": round(self.recalls.at_10, 6),
            "tie_aware_recall_at_10": round(self.recalls.tie_aware_at_10, 6),
            "p50_ms": round(self.p50_ms, 4),
            "p95_ms": round(self.p95_ms, 4),
            "p99_ms": round(self.p99_ms, 4),
            "latency_batch_size": LATENCY_BATCH_SIZE,
            "faiss_threads": self.faiss_threads,
            "index_bytes": self.footprint.disk_bytes,
            "index_mib": round(self.footprint.disk_bytes / (1024 * 1024), 1),
            "memory_bytes": self.footprint.memory_bytes,
            "memory_mib": round(self.footprint.memory_bytes / (1024 * 1024), 1),
            "build_seconds": round(self.build_seconds, 1),
        }


@dataclass(slots=True)
class Curve:
    """Every measurement for one index family."""

    family: str
    points: list[Measurement] = field(default_factory=list)

    def as_list(self) -> list[dict[str, Any]]:
        """Return the curve sorted by recall, as JSON-serialisable mappings."""
        return [m.as_dict() for m in sorted(self.points, key=lambda m: m.recalls.at_top_k)]


@dataclass(frozen=True, slots=True)
class VectorSet:
    """Item vectors to index and the user vectors that query them."""

    name: str
    items: np.ndarray
    queries: np.ndarray
    items_untrained_excluded: int
    query_definition: str


@dataclass(frozen=True, slots=True)
class IndexSpace:
    """The vectors approximate indexes are built on, and the queries that search them."""

    name: str
    items: np.ndarray
    queries: np.ndarray
    metric: int


@dataclass(frozen=True, slots=True)
class Built:
    """One built index, how long it took, and the queries it must be searched with."""

    index: faiss.Index
    family: str
    params: dict[str, int]
    build_seconds: float
    queries: np.ndarray


def _mips_to_l2(items: np.ndarray, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return items and queries transformed so L2 order equals the original inner-product order.

    For a query q and item x with the extra coordinate e = sqrt(R^2 - |x|^2), where R is the
    largest item norm: |q' - x'|^2 = |q|^2 + R^2 - 2 q.x, and |q|^2 + R^2 is the same for every
    item, so the nearest item in L2 is the one with the largest inner product.
    """
    norms = np.linalg.norm(items, axis=1)
    extra = np.sqrt(np.clip(norms.max() ** 2 - norms**2, 0.0, None)).astype(np.float32)
    return (
        np.ascontiguousarray(np.hstack([items, extra[:, None]]), dtype=np.float32),
        np.ascontiguousarray(
            np.hstack([queries, np.zeros((queries.shape[0], 1), dtype=np.float32)]),
            dtype=np.float32,
        ),
    )


def _pad_columns(vectors: np.ndarray, multiple: int) -> np.ndarray:
    """Zero-pad columns up to the next multiple of `multiple`."""
    width = -(-vectors.shape[1] // multiple) * multiple
    padded = np.zeros((vectors.shape[0], width), dtype=np.float32)
    padded[:, : vectors.shape[1]] = vectors
    return padded


def _index_space(vectors: VectorSet, *, raw_inner_product: bool) -> IndexSpace:
    """Choose what the approximate indexes are built on; see the module docstring."""
    if raw_inner_product:
        return IndexSpace(
            "inner_product", vectors.items, vectors.queries, faiss.METRIC_INNER_PRODUCT
        )
    items, queries = _mips_to_l2(vectors.items, vectors.queries)
    return IndexSpace("mips_l2", items, queries, faiss.METRIC_L2)


def _percentile(samples: list[float], q: float) -> float:
    """Return the q-th percentile of latency samples, nearest-rank."""
    ordered = sorted(samples)
    index = min(round(q / 100 * len(ordered)) - 1, len(ordered) - 1)
    return ordered[max(index, 0)]


def _measure_recalls(
    found: np.ndarray,
    truth: GroundTruth,
    items: np.ndarray,
    queries: np.ndarray,
) -> Recalls:
    """Score one search's results against the exact top-k, strictly and allowing for ties.

    `found` may be wider than the truth; only its first `truth.ids.shape[1]` columns count. FAISS
    pads with -1 when an index returns fewer results than asked, which is a miss, never a tie.
    """
    k = truth.ids.shape[1]
    short = min(SHORT_K, k)
    found = found[:, :k]
    pairs = list(zip(found, truth.ids, strict=True))
    hits = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in pairs)
    short_hits = sum(len(set(a[:short].tolist()) & set(b[:short].tolist())) for a, b in pairs)

    returned = found[:, :short]
    valid = returned >= 0
    scores = np.einsum("qd,qkd->qk", queries, items[np.where(valid, returned, 0)])
    threshold = truth.scores[:, short - 1 : short] - TIE_TOLERANCE
    tie_hits = int(np.count_nonzero(valid & (scores >= threshold)))

    rows = truth.ids.shape[0]
    return Recalls(
        at_top_k=hits / (rows * k),
        at_10=short_hits / (rows * short),
        tie_aware_at_10=tie_hits / (rows * short),
    )


def _time_search(
    index: faiss.Index,
    queries: np.ndarray,
    top_k: int,
    repeats: int,
) -> list[float]:
    """Return per-call search latency in milliseconds, one warmed-up call excluded."""
    step = LATENCY_BATCH_SIZE
    index.search(queries[:step], top_k)
    samples: list[float] = []
    for _ in range(repeats):
        for start in range(0, len(queries), step):
            batch = queries[start : start + step]
            started = time.perf_counter()
            index.search(batch, top_k)
            samples.append((time.perf_counter() - started) * 1000)
    return samples


def _footprint(index: faiss.Index, *, measure_memory: bool = True) -> Footprint:
    """Write the index to disk and, unless told not to, load it in a fresh process to size it."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "index.faiss"
        faiss.write_index(index, str(path))
        memory = 0
        if measure_memory:
            probe = subprocess.run(  # noqa: S603 - fixed argv, no shell
                [sys.executable, "-c", _MEMORY_PROBE, str(path)],
                capture_output=True,
                text=True,
                check=True,
            )
            memory = int(probe.stdout.strip().splitlines()[-1])
        return Footprint(disk_bytes=path.stat().st_size, memory_bytes=memory)


def _exact_index(items: np.ndarray, queries: np.ndarray, top_k: int) -> GroundTruth:
    """Return exact top-k from brute-force inner-product search."""
    index = faiss.IndexFlatIP(items.shape[1])
    index.add(items)
    scores, ids = index.search(queries, top_k)
    return GroundTruth(ids=ids, scores=scores)


def _neighbour_contrast(scores: np.ndarray) -> dict[str, float]:
    """Summarise how crowded the exact neighbours are.

    The per-query gap between the 10th and the k-th exact score, as a median, both absolute and
    relative to the 10th score. A small gap means the top 10 is barely separated from the top 200,
    so any approximate search reorders near-equal items.
    """
    short = min(SHORT_K, scores.shape[1]) - 1
    gap = scores[:, short] - scores[:, -1]
    relative = gap / np.clip(np.abs(scores[:, short]), 1e-12, None)
    return {
        "median_score_rank_1": round(float(np.median(scores[:, 0])), 4),
        "median_score_rank_10": round(float(np.median(scores[:, short])), 4),
        "median_score_rank_k": round(float(np.median(scores[:, -1])), 4),
        "median_gap_rank_10_to_k": round(float(np.median(gap)), 4),
        "median_relative_gap_rank_10_to_k": round(float(np.median(relative)), 4),
    }


def _measure(
    built: Built,
    params: dict[str, int],
    vectors: VectorSet,
    truth: GroundTruth,
    footprint: Footprint,
) -> Measurement:
    """Time and score one index at its current search settings."""
    top_k = truth.ids.shape[1]
    samples = _time_search(built.index, built.queries, top_k, repeats=1)
    _, found = built.index.search(built.queries, top_k)
    point = Measurement(
        family=built.family,
        params=params,
        recalls=_measure_recalls(found, truth, vectors.items, vectors.queries),
        p50_ms=_percentile(samples, 50),
        p95_ms=_percentile(samples, 95),
        p99_ms=_percentile(samples, 99),
        footprint=footprint,
        build_seconds=built.build_seconds,
        faiss_threads=faiss.omp_get_max_threads(),
    )
    _logger.info(
        "ann.sweep_point",
        extra={"family": built.family, **params, "recall": point.recalls.at_top_k},
    )
    return point


def _sweep(
    built: Built,
    knob: tuple[str, str],
    values: tuple[int, ...],
    vectors: VectorSet,
    truth: GroundTruth,
) -> list[Measurement]:
    """Measure one built index at each value of its search-time knob.

    `knob` is `(name recorded in the result file, name FAISS sets it by)`.
    """
    footprint = _footprint(built.index)
    out: list[Measurement] = []
    for value in values:
        faiss.ParameterSpace().set_index_parameter(built.index, knob[1], value)
        out.append(_measure(built, {**built.params, knob[0]: value}, vectors, truth, footprint))
    return out


def _build_flat(vectors: VectorSet, truth: GroundTruth) -> list[Measurement]:
    """Exact inner-product search on the original vectors, the baseline every index must beat.

    Measured with one thread and with FAISS's default. The default is what a single idle request
    sees; one thread is closer to what each request gets when concurrent requests share the cores,
    and it is the figure to hold against the retrieval budget under load.
    """
    started = time.perf_counter()
    index = faiss.IndexFlatIP(vectors.items.shape[1])
    index.add(vectors.items)
    built = Built(index, "flat", {}, time.perf_counter() - started, vectors.queries)
    footprint = _footprint(index)
    default_threads = faiss.omp_get_max_threads()
    out: list[Measurement] = []
    for threads in sorted({1, default_threads}):
        faiss.omp_set_num_threads(threads)
        try:
            out.append(_measure(built, {"threads": threads}, vectors, truth, footprint))
        finally:
            faiss.omp_set_num_threads(default_threads)
    return out


def _build_hnsw(
    space: IndexSpace, vectors: VectorSet, truth: GroundTruth, m: int, ef_construction: int
) -> list[Measurement]:
    """Build one HNSW graph and sweep its search width."""
    started = time.perf_counter()
    index = faiss.IndexHNSWFlat(space.items.shape[1], m, space.metric)
    index.hnsw.efConstruction = ef_construction
    index.add(space.items)
    params = {"M": m, "ef_construction": ef_construction}
    built = Built(index, "hnsw", params, time.perf_counter() - started, space.queries)
    return _sweep(built, ("ef_search", "efSearch"), HNSW_EF_SEARCH, vectors, truth)


def _build_ivf(
    space: IndexSpace, vectors: VectorSet, truth: GroundTruth, nlist: int, pq_m: int | None
) -> list[Measurement]:
    """Train one IVF index, flat or product-quantised, and sweep how many lists it probes.

    Training the coarse quantiser (and the PQ codebooks) is part of every rebuild, so it is part
    of the build time; timing only `add` understated it.
    """
    items, queries = space.items, space.queries
    if pq_m is not None:
        items = _pad_columns(items, PQ_WIDTH_MULTIPLE)
        queries = _pad_columns(queries, PQ_WIDTH_MULTIPLE)
    dim = items.shape[1]
    started = time.perf_counter()
    quantizer = faiss.IndexFlat(dim, space.metric)
    index: faiss.Index = (
        faiss.IndexIVFFlat(quantizer, dim, nlist, space.metric)
        if pq_m is None
        else faiss.IndexIVFPQ(quantizer, dim, nlist, pq_m, 8, space.metric)
    )
    index.train(items)
    index.add(items)
    family, params = (
        ("ivfflat", {"nlist": nlist})
        if pq_m is None
        else (
            "ivfpq",
            {"nlist": nlist, "m": pq_m},
        )
    )
    built = Built(index, family, params, time.perf_counter() - started, queries)
    return _sweep(built, ("nprobe", "nprobe"), IVF_NPROBE, vectors, truth)


def _choose(points: list[Measurement], target: float) -> Measurement | None:
    """Pick the lowest-p99 configuration that reaches the recall target, or nothing.

    Lowest latency *among configurations that qualify*, rather than the best recall overall: an
    index nobody waits on is worth more than one that is marginally more accurate and 10x slower.
    When nothing qualifies there is no operating point to recommend, and returning the best of a
    failing set would read as one; the best recall is reported separately for that case.
    """
    qualifying = [p for p in points if p.recalls.at_top_k >= target]
    return min(qualifying, key=lambda p: p.p99_ms) if qualifying else None


def _best(points: list[Measurement]) -> Measurement | None:
    """Return the highest-recall configuration, which is not a recommendation."""
    return max(points, key=lambda p: p.recalls.at_top_k) if points else None


def _load_two_tower(root: Path, rng: np.random.Generator, queries: int) -> VectorSet:
    """Two-tower item vectors and recency-pooled user-tower queries."""
    model = two_tower_model.load_model(root / "two_tower.npz")
    trained = (
        np.ones(model.n_items(), dtype=bool)
        if model.trained_items is None
        else np.asarray(model.trained_items, dtype=bool)
    )
    servable = np.flatnonzero(np.diff(model.history_indptr) >= 1)
    user_codes = rng.choice(servable, size=queries, replace=False)
    return VectorSet(
        name="two_tower",
        items=np.ascontiguousarray(model.item_embeddings[trained], dtype=np.float32),
        queries=np.ascontiguousarray(model.user_vectors(user_codes)),
        items_untrained_excluded=int((~trained).sum()),
        query_definition="user-tower vectors (recency-weighted mean pool) of servable users",
    )


def _load_als(root: Path, rng: np.random.Generator, queries: int) -> VectorSet:
    """ALS item factors and the fitted user factors of users ALS trained."""
    model = als.load_model(root / "als.npz")
    trained = np.any(model.item_factors != 0, axis=1)
    servable = np.flatnonzero(np.any(model.user_factors != 0, axis=1))
    user_codes = rng.choice(servable, size=queries, replace=False)
    return VectorSet(
        name="als",
        items=np.ascontiguousarray(model.item_factors[trained], dtype=np.float32),
        queries=np.ascontiguousarray(model.user_factors[user_codes], dtype=np.float32),
        items_untrained_excluded=int((~trained).sum()),
        query_definition="fitted ALS user factors of users ALS trained (non-zero factor rows)",
    )


def _sysctl(name: str) -> str | None:
    """Return a macOS sysctl value, or None where sysctl does not exist."""
    try:
        return subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["/usr/sbin/sysctl", "-n", name], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _cpu_model() -> str:
    """Return the CPU model name, which `platform.processor()` reports only as `arm` on macOS."""
    if platform.system() == "Darwin":
        return _sysctl("machdep.cpu.brand_string") or "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _memory_total() -> int | None:
    """Return physical memory in bytes, where the platform reports it."""
    if platform.system() == "Darwin":
        value = _sysctl("hw.memsize")
        return int(value) if value else None
    pages, size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
    return int(pages * size) if pages > 0 and size > 0 else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, choices=sorted(CORPORA))
    parser.add_argument("--vectors", default="als", choices=VECTOR_SETS)
    parser.add_argument("--queries", type=int, default=DEFAULT_QUERIES)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--recall-target", type=float, default=RECALL_TARGET)
    parser.add_argument("--quick", action="store_true", help="Trim the grids for a smoke run.")
    parser.add_argument(
        "--raw-inner-product",
        action="store_true",
        help="Build on the original vectors with the inner-product metric (the ablation).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the sweep and write `results/ann_index_<corpus>_<vectors>[_raw_ip].json`."""
    args = parse_args(argv)
    configure_logging()
    corpus = get_corpus(args.corpus)
    root = settings.artifact_root / corpus.name
    artefact = root / ("als.npz" if args.vectors == "als" else "two_tower.npz")
    if not artefact.exists():
        print(f"error: no {args.vectors} artefact at {artefact}; train it first", file=sys.stderr)
        return 1

    rng = np.random.default_rng(settings.seed)
    loader = _load_als if args.vectors == "als" else _load_two_tower
    vectors = loader(root, rng, args.queries)
    space = _index_space(vectors, raw_inner_product=args.raw_inner_product)

    started = time.perf_counter()
    truth = _exact_index(vectors.items, vectors.queries, args.top_k)
    exact_seconds = time.perf_counter() - started
    contrast = _neighbour_contrast(truth.scores)
    _logger.info(
        "ann.ground_truth",
        extra={
            "vectors": vectors.name,
            "items": vectors.items.shape[0],
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

    points = _build_flat(vectors, truth)
    for m in m_values:
        for efc in efc_values:
            points.extend(_build_hnsw(space, vectors, truth, m, efc))
    for nlist in nlists:
        points.extend(_build_ivf(space, vectors, truth, nlist, None))
        for m in pq_m:
            points.extend(_build_ivf(space, vectors, truth, nlist, m))

    curves = {
        family: Curve(family, [p for p in points if p.family == family]) for family in FAMILIES
    }
    chosen = {family: _choose(c.points, args.recall_target) for family, c in curves.items()}
    best = {family: _best(c.points) for family, c in curves.items()}

    payload: dict[str, Any] = {
        "corpus": corpus.name,
        "vectors": vectors.name,
        "index_space": space.name,
        "command": " ".join(["uv run python scripts/bench_ann_index.py", *sys.argv[1:]]),
        "quick": args.quick,
        "items_indexed": int(vectors.items.shape[0]),
        "items_untrained_excluded": vectors.items_untrained_excluded,
        "dimensions": int(vectors.items.shape[1]),
        "queries": args.queries,
        "query_definition": vectors.query_definition,
        "top_k": args.top_k,
        "recall_definition": (
            "recall_at_top_k and recall_at_10: fraction of exact top-k (top-10) ids recovered, "
            "against brute-force inner-product search over the trained item vectors. "
            f"tie_aware_recall_at_10: a returned top-10 item is a hit if its exact score is at "
            f"least the exact 10th-best score minus {TIE_TOLERANCE}."
        ),
        "tie_tolerance": TIE_TOLERANCE,
        "seed": settings.seed,
        "float32_bytes": int(vectors.items.nbytes),
        "exact_search_seconds": round(exact_seconds, 1),
        "neighbour_contrast": contrast,
        "recall_target": args.recall_target,
        "curves": {family: c.as_list() for family, c in curves.items()},
        "chosen": {
            "rule": (
                f"Lowest p99 latency among configurations reaching recall_at_top_k >= "
                f"{args.recall_target}; null when none does."
            ),
            **{family: point.as_dict() if point else None for family, point in chosen.items()},
        },
        "highest_recall": {
            family: point.as_dict() if point else None for family, point in best.items()
        },
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "environment": {
            "date": datetime.now(UTC).date().isoformat(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "cpu": _cpu_model(),
            "logical_cpus": os.cpu_count(),
            "memory_bytes": _memory_total(),
            "python": platform.python_version(),
            "faiss": faiss.__version__,
            "faiss_threads": faiss.omp_get_max_threads(),
        },
    }
    suffix = "_raw_ip" if args.raw_inner_product else ""
    out = settings.results_root / f"ann_index_{corpus.name}_{vectors.name}{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _print(payload, out)
    return 0


def _print(payload: dict[str, Any], out: object) -> None:
    """Print the operating points, the best recall and where the result file landed."""
    print(
        f"\n=== {payload['vectors']} ({payload['index_space']}): "
        f"{payload['items_indexed']} vectors, "
        f"{payload['dimensions']}d, {payload['queries']} queries, top-{payload['top_k']} ==="
    )
    for family in FAMILIES:
        chosen = payload["chosen"][family]
        point = chosen or payload["highest_recall"][family]
        if chosen is None:
            print(f"\n  {family}: none meets the {payload['recall_target']} target")
        if point is None:
            continue
        print(
            f"\n  {'chosen' if chosen else 'highest recall'} {family}: {point['params']}"
            f"\n    recall@{payload['top_k']} {point['recall_at_top_k']:.4f}"
            f"  recall@10 {point['recall_at_10']:.4f}"
            f"  tie-aware@10 {point['tie_aware_recall_at_10']:.4f}"
            f"\n    p50 {point['p50_ms']:.3f}ms  p95 {point['p95_ms']:.3f}ms"
            f"  p99 {point['p99_ms']:.3f}ms  (batch {point['latency_batch_size']},"
            f" {point['faiss_threads']} threads)"
            f"\n    disk {point['index_mib']} MiB  memory {point['memory_mib']} MiB"
            f"  build {point['build_seconds']}s"
        )
    print(f"\n  float32 vectors: {payload['float32_bytes'] / (1024 * 1024):.0f} MiB")
    print(f"  neighbour contrast: {payload['neighbour_contrast']}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
