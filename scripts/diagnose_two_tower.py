"""Is the two-tower undertrained, or is the objective the problem.

Phase 3 measured HR@20 well below every baseline. This script separates the two explanations by
training on a user subsample for several epochs and scoring a small cohort after each one. If the
metric climbs with epochs the model was undertrained; if it is flat or falling, more training is
not the answer and the objective is.

    uv run python scripts/diagnose_two_tower.py --users 400000 --cohort 2000 --epochs 1 2 3 5
"""

from __future__ import annotations

import argparse
import json
import time

import duckdb
import numpy as np

from rankstack.config import settings
from rankstack.data.corpus import CORPORA, DEFAULT_CORPUS, get_corpus
from rankstack.data.encode import build_cohort, load_encoding
from rankstack.eval.harness import CohortHistory, EvalConfig, evaluate_source
from rankstack.logging import configure_logging, get_logger
from rankstack.retrieval.two_tower import (
    DEFAULT_MIN_USER_HISTORY,
    DEFAULT_TEMPERATURE,
    TwoTowerConfig,
    item_log_popularity,
    train_two_tower,
)

_logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, choices=sorted(CORPORA))
    parser.add_argument("--users", type=int, default=400_000, help="Training users to subsample.")
    parser.add_argument("--cohort", type=int, default=2_000, help="Cohort size to score.")
    parser.add_argument("--epochs", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--factors", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--learning-rate-sweep", type=float, nargs="+", default=[])
    parser.add_argument("--temperature-sweep", type=float, nargs="+", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Train the two-tower repeatedly on one subsample and report HR@20 after each run."""
    args = parse_args(argv)
    configure_logging()
    corpus = get_corpus(args.corpus)
    root = settings.artifact_root / corpus.name
    encoding = load_encoding(settings.artifact_root, corpus.name)
    history = np.load(root / "cooccurrence.history.npz")
    indptr = history["history_indptr"]
    codes = history["history_codes"]

    rng = np.random.default_rng(settings.seed)
    n_users = int(indptr.size - 1)
    eligible = np.flatnonzero(np.diff(indptr) >= DEFAULT_MIN_USER_HISTORY)
    sample = np.sort(rng.choice(eligible, size=min(args.users, eligible.size), replace=False))

    # The subsample keeps the original user codes: unsampled users get an empty history, so they are
    # simply not eligible for training, and the trained model can still score any user code.
    keep = np.zeros(n_users, dtype=bool)
    keep[sample] = True
    depths = np.where(keep, np.diff(indptr), 0)
    sub_indptr = np.zeros(n_users + 1, dtype=np.int64)
    np.cumsum(depths, out=sub_indptr[1:])
    sub_codes = codes[np.concatenate([np.arange(indptr[u], indptr[u + 1]) for u in sample])]

    connection = duckdb.connect(database=":memory:")
    processed = settings.processed(corpus.name)
    cohort = build_cohort(
        connection, "warm_users", encoding, processed, size=args.cohort, seed=settings.seed
    )
    connection.close()
    users = np.asarray(cohort.user_codes, dtype=np.int64)
    targets = np.asarray(cohort.target_codes, dtype=np.int64)
    popularity = np.bincount(codes.astype(np.int64), minlength=encoding.n_items).astype(np.float32)

    log_q = item_log_popularity(sub_codes, encoding.n_items)
    history_view = CohortHistory(
        indptr=indptr, codes=codes, n_items=encoding.n_items, popularity=popularity
    )

    rows: list[dict[str, object]] = []
    learning_rates = args.learning_rate_sweep or [args.learning_rate]
    temperatures = args.temperature_sweep or [DEFAULT_TEMPERATURE]
    for temperature in temperatures:
        for learning_rate in learning_rates:
            for epochs in args.epochs:
                started = time.perf_counter()
                model, report = train_two_tower(
                    sub_indptr,
                    sub_codes,
                    int(indptr.size - 1),
                    encoding.n_items,
                    log_q,
                    TwoTowerConfig(
                        factors=args.factors,
                        batch_size=args.batch_size,
                        learning_rate=learning_rate,
                        temperature=temperature,
                        epochs=epochs,
                        seed=settings.seed,
                    ),
                )
                result = evaluate_source(
                    model,
                    users,
                    targets,
                    history_view,
                    EvalConfig(
                        name="two_tower",
                        cohort_name="warm_users",
                        keep_recommendations=False,
                    ),
                )
                row = {
                    "temperature": temperature,
                    "learning_rate": learning_rate,
                    "epochs": epochs,
                    "train_loss": round(report.final_loss, 4),
                    "train_steps": report.steps,
                    "served_fraction": round(result.ranking.served_fraction, 4),
                    "hr@10": round(result.ranking.mean("hr")["hr@10"], 6),
                    "hr@20": round(result.ranking.mean("hr")["hr@20"], 6),
                    "hr@50": round(result.ranking.mean("hr")["hr@50"], 6),
                    "ndcg@20": round(result.ranking.mean("ndcg")["ndcg@20"], 6),
                    "wall_clock_seconds": round(time.perf_counter() - started, 1),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)

    payload = {
        "corpus": corpus.name,
        "command": "uv run python scripts/diagnose_two_tower.py",
        "training_users": int(sample.size),
        "cohort_size": int(users.size),
        "note": (
            "Sampled from the warm_users cohort, so these numbers are not comparable with the "
            "committed evaluation file; the point is the trend across epochs and learning rates."
        ),
        "rows": rows,
    }
    path = settings.results_root / f"two_tower_diagnosis_{corpus.name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
