"""Evaluate every baseline on a frozen cohort and write the results file.

Produces exact full-catalogue Recall@K, HR@K, NDCG@K and MRR@K at K = 10, 20, 50, broken out by
segment, plus coverage, novelty, new-item share and popularity bias, plus paired bootstrap
comparisons against the recency-weighted popularity baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import TYPE_CHECKING, Any

import duckdb
import numpy as np
import scipy.sparse as sp

from rankstack.config import Settings
from rankstack.config import settings as default_settings
from rankstack.data.corpus import CORPORA, DEFAULT_CORPUS, Corpus, get_corpus
from rankstack.data.encode import Encoding, build_cohort, load_encoding
from rankstack.eval.harness import CohortHistory, EvalConfig, compare_sources, evaluate_source
from rankstack.logging import configure_logging, get_logger
from rankstack.retrieval.als import AlsModel
from rankstack.retrieval.cooccurrence import CooccurrenceModel
from rankstack.retrieval.popularity import PopularityModel
from rankstack.retrieval.two_tower import load_model

if TYPE_CHECKING:
    from pathlib import Path

    from rankstack.retrieval.base import CandidateSource

_logger = get_logger(__name__)

DEFAULT_COHORT_SIZE = 20_000
DEFAULT_COHORTS = ("all_test_users", "warm_users")
DEFAULT_SOURCES = (
    "popularity",
    "popularity_recent",
    "cooccurrence",
    "als",
    "two_tower",
)
BASELINE_NAME = "popularity_recent"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, choices=sorted(CORPORA))
    parser.add_argument("--cohorts", nargs="+", default=list(DEFAULT_COHORTS))
    parser.add_argument("--cohort-size", type=int, default=DEFAULT_COHORT_SIZE)
    parser.add_argument("--resamples", type=int, default=1_000)
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    return parser.parse_args(argv)


def load_sources(root: Path, names: list[str]) -> dict[str, CandidateSource]:
    """Load the trained artefacts for the requested sources."""
    loaded: dict[str, CandidateSource] = {}
    for name in names:
        match name:
            case "popularity":
                loaded[name] = PopularityModel.load(root / "popularity.npy")
            case "popularity_recent":
                loaded[name] = PopularityModel.load(root / "popularity_recent.npy")
            case "cooccurrence":
                loaded[name] = _load_cooccurrence(root)
            case "als":
                loaded[name] = _load_als(root)
            case "two_tower":
                loaded[name] = load_model(root / "two_tower.npz")
            case "two_tower_no_logq":
                loaded[name] = load_model(root / "two_tower_no_logq.npz")
            case _:
                msg = f"unknown source {name!r}"
                raise KeyError(msg)
    return loaded


def _load_cooccurrence(root: Path) -> CooccurrenceModel:
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


def _load_als(root: Path) -> AlsModel:
    payload = np.load(root / "als.npz")
    metadata = json.loads((root / "als.json").read_text())
    return AlsModel(
        user_factors=payload["user_factors"],
        item_factors=payload["item_factors"],
        history_indptr=payload["history_indptr"],
        history_codes=payload["history_codes"],
        min_user_history=int(metadata["min_user_history"]),
        factors=int(metadata["factors"]),
        iterations=int(metadata["iterations"]),
    )


def item_popularity_and_profile(root: Path, n_items: int) -> tuple[np.ndarray, np.ndarray]:
    """Return train-window item popularity and its L2-normalised profile.

    The profile is the only item embedding available before the two-tower model exists, so
    intra-list diversity is reported against it and labelled `interaction_profile` in the results.
    """
    codes = np.load(root / "cooccurrence.history.npz")["history_codes"].astype(np.int64)
    popularity = np.bincount(codes, minlength=n_items).astype(np.float32)
    norm = float(np.linalg.norm(popularity))
    profile = popularity / norm if norm else popularity
    return popularity, profile


def main(argv: list[str] | None = None) -> int:
    """Evaluate every source on every cohort and write one result file."""
    args = parse_args(argv)
    configure_logging()
    corpus = get_corpus(args.corpus)
    return evaluate(corpus, args, default_settings)


def evaluate(corpus: Corpus, args: argparse.Namespace, config: Settings) -> int:
    """Run the evaluation and write `results/evaluation_<corpus>.json`."""
    started = time.perf_counter()
    processed = config.processed(corpus.name)
    if not (processed / "interactions" / "train.parquet").exists():
        print("error: run scripts/build_dataset.py first", file=sys.stderr)
        return 1

    encoding = load_encoding(config.artifact_root, corpus.name)
    root = config.artifact_root / corpus.name
    sources = load_sources(root, list(args.sources))
    popularity, profile = item_popularity_and_profile(root, encoding.n_items)

    payload: dict[str, Any] = {
        "corpus": corpus.name,
        "command": f"uv run python scripts/evaluate_baselines.py --corpus {corpus.name}",
        "cohort_size": args.cohort_size,
        "resamples": args.resamples,
        "baseline_for_paired_comparison": BASELINE_NAME,
        "metric_note": "exact full-catalogue ranking, no sampled negatives",
        "intra_list_diversity_basis": "interaction_profile",
        "n_items": encoding.n_items,
        "n_users": encoding.n_users,
        "cohorts": {},
    }

    connection = duckdb.connect(database=":memory:")
    for cohort_name in args.cohorts:
        cohort = build_cohort(
            connection,
            cohort_name,
            encoding,
            processed,
            size=args.cohort_size,
            seed=config.seed,
        )
        users = np.asarray(cohort.user_codes, dtype=np.int64)
        targets = np.asarray(cohort.target_codes, dtype=np.int64)
        history = _history_for(sources, encoding, popularity)
        cohort_payload: dict[str, Any] = {"definition": cohort.as_dict(), "sources": {}}
        results = {}
        for source_name, source in sources.items():
            result = evaluate_source(
                source,
                users,
                targets,
                history,
                EvalConfig(name=source_name, cohort_name=cohort_name, profile_embeddings=profile),
            )
            results[source_name] = result
            cohort_payload["sources"][source_name] = result.as_dict()
        baseline = results.get(BASELINE_NAME)
        if baseline is not None:
            cohort_payload["paired_vs_baseline"] = {
                name: compare_sources(result, baseline, resamples=args.resamples, seed=config.seed)
                for name, result in results.items()
                if name != BASELINE_NAME
            }
        payload["cohorts"][cohort_name] = cohort_payload
    connection.close()

    payload["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    path = config.results_root / f"evaluation_{corpus.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _logger.info("evaluation.written", extra={"path": str(path)})
    print_table(payload)
    return 0


def _history_for(
    sources: dict[str, CandidateSource], encoding: Encoding, popularity: np.ndarray
) -> CohortHistory:
    """Build the evaluation history from whichever trained source carries one."""
    for source in sources.values():
        indptr = getattr(source, "history_indptr", None)
        codes = getattr(source, "history_codes", None)
        if indptr is not None and codes is not None:
            return CohortHistory(
                indptr=np.asarray(indptr),
                codes=np.asarray(codes),
                n_items=encoding.n_items,
                popularity=popularity,
            )
    msg = "none of the loaded sources carries a train-window history"
    raise ValueError(msg)


def print_table(payload: dict[str, Any]) -> None:
    """Print the headline table to stdout."""
    for cohort_name, cohort in payload["cohorts"].items():
        print(f"\n=== cohort {cohort_name} (n={cohort['definition']['size']}) ===")
        print(
            f"{'source':>20s} {'served':>7s} {'HR@10':>7s} {'HR@20':>7s}"
            f" {'HR@50':>7s} {'NDCG@20':>8s} {'cov@50':>7s}"
        )
        for name, result in cohort["sources"].items():
            ranking = result["ranking"]
            beyond = result["beyond_list"]
            print(
                f"{name:>20s} {ranking['served_fraction']:>7.3f} "
                f"{ranking['hr@10']:>7.4f} {ranking['hr@20']:>7.4f} {ranking['hr@50']:>7.4f} "
                f"{ranking['ndcg@20']:>8.4f} {beyond['coverage_at_50']:>7.4f}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
