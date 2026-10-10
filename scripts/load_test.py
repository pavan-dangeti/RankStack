"""Measure serving latency against the design document's SLOs.

Targets (design document §3.3): warm p50 <= 60 ms, p95 <= 150 ms, p99 <= 300 ms; cold-start p95
<= 250 ms; degraded p95 <= 20 ms. These are targets, not results: this script measures them and
records pass or fail per line, so a missed target is visible rather than quietly dropped.

The measured request path is `Recommender.recommend`, not the ASGI layer, because the ASGI overhead
is a constant the design allocates separately (10 ms for serialisation, logging and tracing). Both
numbers are reported so the split is checkable.

    uv run python scripts/load_test.py --requests 400 --concurrency 8
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from rankstack.config import settings
from rankstack.data.corpus import CORPORA, DEFAULT_CORPUS, get_corpus
from rankstack.data.encode import load_encoding
from rankstack.logging import configure_logging, get_logger
from rankstack.serving.artifacts import ServingArtifacts, load_artifacts
from rankstack.serving.recommender import Recommender

_logger = get_logger(__name__)

#: A user needs at least this many stored interactions for the warm path to mean anything.
MIN_WARM_HISTORY = 2

#: design document §3.3, in milliseconds.
SLO_MS = {
    "warm_p50": 60.0,
    "warm_p95": 150.0,
    "warm_p99": 300.0,
    "cold_start_p95": 250.0,
    "degraded_p95": 20.0,
}


@dataclass(frozen=True, slots=True)
class Latency:
    """A measured latency distribution in milliseconds."""

    label: str
    samples: list[float]

    def percentile(self, q: float) -> float:
        """Return the q-th percentile, nearest-rank, in milliseconds."""
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        index = min(round(q / 100 * len(ordered)) - 1, len(ordered) - 1)
        return float(ordered[max(index, 0)])

    def as_dict(self) -> dict[str, object]:
        """Return the distribution as a JSON-serialisable mapping."""
        if not self.samples:
            return {"label": self.label, "n": 0}
        return {
            "label": self.label,
            "n": len(self.samples),
            "p50_ms": round(self.percentile(50), 3),
            "p95_ms": round(self.percentile(95), 3),
            "p99_ms": round(self.percentile(99), 3),
            "mean_ms": round(statistics.fmean(self.samples), 3),
            "max_ms": round(max(self.samples), 3),
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, choices=sorted(CORPORA))
    parser.add_argument("--requests", type=int, default=400)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--cold-requests", type=int, default=50)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--no-titles", action="store_true")
    return parser.parse_args(argv)


def _drive(recommender: Recommender, users: list[str], k: int, concurrency: int) -> list[float]:
    """Issue every request, `concurrency` at a time, and return the durations in milliseconds."""

    def one(user: str) -> float:
        started = time.perf_counter()
        response = recommender.recommend(user, k)
        elapsed = (time.perf_counter() - started) * 1000
        if not response.items:
            msg = "the request path returned an empty list"
            raise RuntimeError(msg)
        return elapsed

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(one, users))


def main(argv: list[str] | None = None) -> int:
    """Run the load test and write `results/load_test_<corpus>.json`."""
    args = parse_args(argv)
    configure_logging()
    corpus = get_corpus(args.corpus)
    root = settings.artifact_root / corpus.name
    if not (root / "serving_manifest.json").exists():
        print("error: run scripts/build_serving_manifest.py first", file=sys.stderr)
        return 1

    load_started = time.perf_counter()
    artifacts = load_artifacts(
        settings.artifact_root,
        corpus.name,
        with_titles=not args.no_titles,
        config=settings,
    )
    load_seconds = time.perf_counter() - load_started
    recommender = Recommender(artifacts=artifacts)

    encoding = load_encoding(settings.artifact_root, corpus.name)
    rng = np.random.default_rng(settings.seed)
    known = np.fromiter(encoding.user_to_code.keys(), dtype=object)
    # Users with real history: a cold user exercises a different path and is measured separately.
    depths = np.diff(artifacts.history_indptr)
    warm_pool = np.asarray(list(encoding.user_to_code.values()))
    warm_pool = warm_pool[depths[warm_pool] >= MIN_WARM_HISTORY]
    warm_ids = rng.choice(warm_pool, size=args.requests, replace=True)
    warm_users = [encoding.user_codes[int(code)] for code in warm_ids]
    cold_users = [f"unknown-{index}" for index in range(args.cold_requests)]
    del known

    _drive(recommender, warm_users[: min(20, len(warm_users))], args.k, 1)  # warm the caches
    warm = Latency("warm", _drive(recommender, warm_users, args.k, args.concurrency))
    cold = Latency("cold_start", _drive(recommender, cold_users, args.k, args.concurrency))

    degraded = Latency(
        "degraded",
        _drive(
            _broken_retriever(artifacts),
            warm_users[: args.cold_requests],
            args.k,
            args.concurrency,
        ),
    )

    measured = {
        "warm_p50": warm.percentile(50),
        "warm_p95": warm.percentile(95),
        "warm_p99": warm.percentile(99),
        "cold_start_p95": cold.percentile(95),
        "degraded_p95": degraded.percentile(95),
    }
    slo = {
        name: {
            "target_ms": target,
            "measured_ms": round(measured[name], 3),
            "met": measured[name] <= target,
        }
        for name, target in SLO_MS.items()
    }

    started = time.perf_counter()
    throughput_requests = 2_000
    throughput_users = (warm_users * (throughput_requests // len(warm_users) + 1))[
        :throughput_requests
    ]
    _drive(recommender, throughput_users, args.k, args.concurrency)
    sequential_elapsed = time.perf_counter() - started
    throughput = throughput_requests / sequential_elapsed

    payload = {
        "corpus": corpus.name,
        "command": (
            f"uv run python scripts/load_test.py --requests {args.requests}"
            f" --concurrency {args.concurrency} --k {args.k}"
        ),
        "requests": args.requests,
        "concurrency": args.concurrency,
        "k": args.k,
        "catalogue_items": artifacts.n_items,
        "titles_loaded": artifacts.catalogue.titles is not None,
        "served_source": type(artifacts.served_source()).__name__,
        "artifact_load_seconds": round(load_seconds, 2),
        "latency": [warm.as_dict(), cold.as_dict(), degraded.as_dict()],
        "slo": slo,
        "all_slos_met": all(entry["met"] for entry in slo.values()),
        "throughput_requests_per_second_single_process": round(throughput, 1),
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "python": platform.python_version(),
        },
        "note": (
            "Latency is measured on Recommender.recommend, the path the ASGI layer calls directly. "
            "Throughput is one process on one core-pool, not a deployed service under real HTTP."
        ),
    }
    out = settings.results_root / f"load_test_{corpus.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(
        f"\n=== {corpus.name}: {args.requests} requests,"
        f" concurrency {args.concurrency}, k={args.k} ==="
    )
    for label, data in (
        ("warm", warm.as_dict()),
        ("cold_start", cold.as_dict()),
        ("degraded", degraded.as_dict()),
    ):
        print(
            f"  {label:>11s} n={data['n']:>5d} p50={data.get('p50_ms')}ms "
            f"p95={data.get('p95_ms')}ms p99={data.get('p99_ms')}ms"
        )
    print("\n  SLO                       target    measured  met")
    for name, entry in slo.items():
        flag = "yes" if entry["met"] else "NO"
        print(f"  {name:<24s} {entry['target_ms']:>6.0f}ms {entry['measured_ms']:>9.2f}ms  {flag}")
    print(f"\n  throughput: {throughput:.0f} req/s single process")
    print(f"  wrote {out}")
    return 0


@dataclass(frozen=True, slots=True)
class ExplodingSource:
    """Stands in for the served source and always fails, as a corrupt index would."""

    n_items_: int

    @property
    def name(self) -> str:
        """Name reported in results files."""
        return "exploding"

    def n_items(self) -> int:
        """Return the catalogue size."""
        return self.n_items_

    def can_serve(self, user_code: int) -> bool:
        """Return True so scoring raises rather than the request looking cold."""
        del user_code
        return True

    def score(self, user_code: int, seen: np.ndarray) -> np.ndarray:
        """Raise unconditionally, as a corrupt index would."""
        del user_code, seen
        msg = "injected index failure"
        raise RuntimeError(msg)


def _broken_retriever(artifacts: ServingArtifacts) -> Recommender:
    """Return a recommender whose served source always raises, to time the degraded path."""
    return Recommender(artifacts=artifacts, source=lambda: ExplodingSource(artifacts.n_items))


if __name__ == "__main__":
    raise SystemExit(main())
