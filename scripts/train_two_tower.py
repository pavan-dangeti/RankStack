"""Train the two-tower retrieval model and write its artefact.

Reuses the encoding and history written by `scripts/train_baselines.py`, so the two-tower sees the
same integer code space and the same capped history as every collaborative baseline. That is what
makes the harness's paired comparisons meaningful.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace

import numpy as np

from rankstack.config import settings
from rankstack.data.corpus import CORPORA, DEFAULT_CORPUS, get_corpus
from rankstack.data.encode import load_encoding
from rankstack.logging import configure_logging, get_logger
from rankstack.retrieval.two_tower import (
    TwoTowerConfig,
    item_log_popularity,
    train_two_tower,
    write_model,
)

_logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, choices=sorted(CORPORA))
    parser.add_argument("--factors", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-history", type=int, default=50)
    parser.add_argument("--recency-decay", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument(
        "--no-log-q-correction",
        action="store_true",
        help="Turn off the popularity correction, as the ablation.",
    )
    parser.add_argument(
        "--ablation", action="store_true", help="Also train without the popularity correction."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Fit the two-tower and record the training run in the results directory."""
    args = parse_args(argv)
    configure_logging()
    corpus = get_corpus(args.corpus)
    root = settings.artifact_root / corpus.name
    history_path = root / "cooccurrence.history.npz"
    if not history_path.exists():
        print("error: no train history; run scripts/train_baselines.py first", file=sys.stderr)
        return 1

    encoding = load_encoding(settings.artifact_root, corpus.name)
    history = np.load(history_path)
    config = TwoTowerConfig(
        factors=args.factors,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        max_history=args.max_history,
        recency_decay=args.recency_decay,
        temperature=args.temperature,
        log_q_correction=not args.no_log_q_correction,
        seed=settings.seed,
    )

    started = time.perf_counter()
    model, report = train_two_tower(
        history["history_indptr"],
        history["history_codes"],
        encoding.n_users,
        encoding.n_items,
        item_log_popularity(history["history_codes"], encoding.n_items),
        config,
    )
    write_model(model, root / "two_tower.npz")

    payload: dict[str, object] = {
        "corpus": corpus.name,
        "command": (
            f"uv run python scripts/train_two_tower.py --corpus {corpus.name} "
            f"--factors {args.factors} --epochs {args.epochs}"
        ),
        "config": config.as_dict(),
        "training": report.as_dict(),
        "n_users": encoding.n_users,
        "n_items": encoding.n_items,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }

    if args.ablation:
        ablation_config = replace(config, log_q_correction=False)
        _logger.info("two_tower.ablation_started")
        ablation_model, ablation_report = train_two_tower(
            history["history_indptr"],
            history["history_codes"],
            encoding.n_users,
            encoding.n_items,
            item_log_popularity(history["history_codes"], encoding.n_items),
            ablation_config,
        )
        write_model(ablation_model, root / "two_tower_no_logq.npz")
        payload["ablation_without_log_q_correction"] = ablation_report.as_dict()

    path = settings.results_root / f"two_tower_{corpus.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nartefact in {root / 'two_tower.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
