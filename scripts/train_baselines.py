"""Train the retrieval baselines and write their artefacts.

Popularity, item-to-item co-occurrence and implicit ALS, all fitted on the train partition only and
all sharing one integer code space. Run before `scripts/evaluate_baselines.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from rankstack.config import settings
from rankstack.data.build import load_manifest
from rankstack.data.corpus import CORPORA, DEFAULT_CORPUS, get_corpus
from rankstack.data.encode import build_encoding, write_encoding
from rankstack.logging import configure_logging, get_logger
from rankstack.retrieval.als import AlsConfig, train_als
from rankstack.retrieval.als import write_model as write_als
from rankstack.retrieval.cooccurrence import train_cooccurrence
from rankstack.retrieval.cooccurrence import write_model as write_cooccurrence
from rankstack.retrieval.popularity import train_popularity

_logger = get_logger(__name__)

DEFAULT_DECAY_DAYS = 90.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, choices=sorted(CORPORA))
    parser.add_argument("--factors", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--min-user-history", type=int, default=2)
    parser.add_argument("--decay-days", type=float, default=DEFAULT_DECAY_DAYS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Fit every baseline and write a training manifest."""
    args = parse_args(argv)
    configure_logging()
    corpus = get_corpus(args.corpus)
    processed = settings.processed(corpus.name)
    train_parquet = processed / "interactions" / "train.parquet"
    if not train_parquet.exists():
        print(
            f"error: missing {train_parquet}; run scripts/build_dataset.py first", file=sys.stderr
        )
        return 1

    manifest = load_manifest(corpus.name, settings=settings)
    boundaries = {
        "train_end_ms": int(manifest["boundaries"]["train_end_ms"]),  # type: ignore[index]
        "test_start_ms": int(manifest["boundaries"]["test_start_ms"]),  # type: ignore[index]
    }
    train_end_ms = boundaries["train_end_ms"]
    artefact_dir = settings.artifact_root / corpus.name
    artefact_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    encoding = build_encoding(train_parquet, boundaries, seed=settings.seed)
    write_encoding(encoding, artefact_dir)

    popularity = train_popularity(train_parquet, encoding.item_to_code, train_end_ms)
    popularity.write(artefact_dir / "popularity.npy")
    recent = train_popularity(
        train_parquet, encoding.item_to_code, train_end_ms, decay_days=args.decay_days
    )
    recent.write(artefact_dir / "popularity_recent.npy")

    cooccurrence = train_cooccurrence(train_parquet, encoding.user_to_code, encoding.item_to_code)
    write_cooccurrence(cooccurrence, artefact_dir / "cooccurrence.npz")

    als_config = AlsConfig(
        factors=args.factors,
        iterations=args.iterations,
        min_user_history=args.min_user_history,
        seed=settings.seed,
    )
    als = train_als(
        cooccurrence.history_indptr,
        cooccurrence.history_codes,
        encoding.n_users,
        encoding.n_items,
        als_config,
    )
    write_als(als, artefact_dir / "als.npz")

    elapsed = time.perf_counter() - started
    training = {
        "corpus": corpus.name,
        "n_users": encoding.n_users,
        "n_items": encoding.n_items,
        "interactions": encoding.extras.get("interactions"),
        "popularity": {"nonzero_items": int((popularity.scores > 0).sum())},
        "popularity_recent": {"decay_days": args.decay_days},
        "cooccurrence": {
            "nnz": int(cooccurrence.similarity.nnz),
            "max_history": cooccurrence.max_history,
            "top_k": cooccurrence.top_k,
            "density": cooccurrence.similarity.nnz / max(cooccurrence.n_items() ** 2, 1),
        },
        "als": als_config.as_dict(),
        "elapsed_seconds": round(elapsed, 1),
        "command": f"uv run python scripts/train_baselines.py --corpus {args.corpus}",
    }
    path = settings.results_root / f"baselines_{corpus.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(training, indent=2, sort_keys=True) + "\n")
    _logger.info("baselines.trained", extra={"corpus": corpus.name, "seconds": round(elapsed, 1)})
    print(json.dumps(training, indent=2, sort_keys=True))
    print(f"\nartefacts in {artefact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
