"""Measure the cost of personalising the served list by fusing two retrievers.

Serving `popularity_recent` alone returns the same list to every user, because it is a global
ranking. This measures what fusing it with co-occurrence costs in hit rate, on the same warm cohort
the harness uses for everything else, so the trade is a number rather than an assertion.

    uv run python scripts/measure_fusion.py --cohort-size 2000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import TYPE_CHECKING

import duckdb
import numpy as np
import scipy.sparse as sp

from rankstack.config import settings
from rankstack.data.corpus import CORPORA, DEFAULT_CORPUS, get_corpus
from rankstack.data.encode import build_cohort, load_encoding
from rankstack.eval.harness import CohortHistory, EvalConfig, compare_sources, evaluate_source
from rankstack.logging import configure_logging, get_logger
from rankstack.retrieval.cooccurrence import CooccurrenceModel
from rankstack.retrieval.fusion import DEFAULT_DEPTH, DEFAULT_RRF_K, FusedSource
from rankstack.retrieval.popularity import PopularityModel

if TYPE_CHECKING:
    from pathlib import Path

_logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, choices=sorted(CORPORA))
    parser.add_argument("--cohort", default="warm_users")
    parser.add_argument("--cohort-size", type=int, default=2_000)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--resamples", type=int, default=1_000)
    return parser.parse_args(argv)


def _load_cooccurrence(root: Path) -> CooccurrenceModel:
    """Load the trained co-occurrence neighbours and the capped history behind them."""
    metadata = json.loads((root / "cooccurrence.json").read_text())
    history = np.load(root / "cooccurrence.history.npz")
    return CooccurrenceModel(
        similarity=sp.load_npz(root / "cooccurrence.npz"),
        item_popularity=history["item_popularity"],
        history_indptr=history["history_indptr"],
        history_codes=history["history_codes"],
        max_history=int(metadata["max_history"]),
        top_k=int(metadata["top_k"]),
        user_block=int(metadata["user_block"]),
    )


def main(argv: list[str] | None = None) -> int:
    """Evaluate the fused retriever against its two components and write the result file."""
    args = parse_args(argv)
    configure_logging()
    corpus = get_corpus(args.corpus)
    root = settings.artifact_root / corpus.name
    if not (root / "cooccurrence.npz").exists():
        print("error: run scripts/train_baselines.py first", file=sys.stderr)
        return 1

    started = time.perf_counter()
    encoding = load_encoding(settings.artifact_root, corpus.name)
    history = np.load(root / "cooccurrence.history.npz")
    history_indptr = np.asarray(history["history_indptr"])
    history_codes = np.asarray(history["history_codes"])
    recent = PopularityModel.load(root / "popularity_recent.npy")
    cooccurrence = _load_cooccurrence(root)

    fused = FusedSource(
        sources={"popularity_recent": recent, "cooccurrence": cooccurrence},
        rrf_k=args.rrf_k,
        depth=args.depth,
    )

    connection = duckdb.connect(database=":memory:")
    cohort = build_cohort(
        connection,
        args.cohort,
        encoding,
        settings.processed(corpus.name),
        size=args.cohort_size,
        seed=settings.seed,
    )
    connection.close()
    users = np.asarray(cohort.user_codes, dtype=np.int64)
    targets = np.asarray(cohort.target_codes, dtype=np.int64)
    counts = np.bincount(history_codes.astype(np.int64), minlength=encoding.n_items).astype(
        np.float32
    )
    view = CohortHistory(
        indptr=history_indptr,
        codes=history_codes,
        n_items=encoding.n_items,
        popularity=counts,
    )

    config = EvalConfig(name="x", cohort_name=args.cohort)
    results = {
        "popularity_recent": evaluate_source(recent, users, targets, view, config),
        "cooccurrence": evaluate_source(cooccurrence, users, targets, view, config),
        "fused": evaluate_source(fused, users, targets, view, config),
    }

    # How much does fusing actually personalise the served list?
    personalisation = _personalisation(results["popularity_recent"], results["fused"])

    payload = {
        "corpus": corpus.name,
        "command": f"uv run python scripts/measure_fusion.py --cohort-size {args.cohort_size}",
        "cohort": args.cohort,
        "cohort_size": int(users.size),
        "rrf_k": args.rrf_k,
        "fusion_depth": args.depth,
        "sources": {name: result.as_dict() for name, result in results.items()},
        "paired_fused_vs_popularity_recent": compare_sources(
            results["fused"],
            results["popularity_recent"],
            resamples=args.resamples,
            seed=settings.seed,
        ),
        "personalisation": personalisation,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    out = settings.results_root / f"fusion_{corpus.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"\n=== {args.cohort} (n={users.size}) ===")
    print(
        f"{'source':>20s} {'served':>7s} {'HR@10':>8s} {'HR@20':>8s} {'HR@50':>8s}"
        f" {'distinct top-1':>14s}"
    )
    for name, result in results.items():
        ranking = result.ranking
        print(
            f"{name:>20s} {ranking.served_fraction:>7.3f} "
            f"{ranking.mean('hr')['hr@10']:>8.5f} {ranking.mean('hr')['hr@20']:>8.5f} "
            f"{ranking.mean('hr')['hr@50']:>8.5f} "
            f"{len({int(c[0]) for c in result.recommendations.values() if c.size}):>14d}"
        )
    print(f"\nwrote {out}")
    return 0


def _personalisation(global_ranking: object, fused_ranking: object) -> dict[str, object]:
    """Count distinct top-1 items, which is the visible symptom of a non-personalised service."""
    global_top = _distinct_top_1(global_ranking)
    fused_top = _distinct_top_1(fused_ranking)
    return {
        "users": int(_n_users(global_ranking)),
        "distinct_top_1_global": global_top,
        "distinct_top_1_fused": fused_top,
        "note": (
            "A global popularity ranking returns one top item to every user. Fusion is the only "
            "reason the served list varies between users at all."
        ),
    }


def _n_users(result: object) -> int:
    """Return how many users a result covered."""
    return len(getattr(result, "recommendations", {}))


def _distinct_top_1(result: object) -> int:
    recommendations = getattr(result, "recommendations", {})
    return len({int(codes[0]) for codes in recommendations.values() if codes.size})


if __name__ == "__main__":
    raise SystemExit(main())
