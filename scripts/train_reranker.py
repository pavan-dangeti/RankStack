"""Build stage-2 training data: candidates, labels, features and LambdaMART fit.

**The label is the user's next train-window interaction.** For each sampled user the most recent
train item is held out, the earlier items are the history, and stage 1 proposes candidates from
that history. A candidate is labelled 1 when it is that held-out item. Nothing from the validation
or test window enters this file, so the ranker cannot learn from the period it will be judged on.

**Stage 1 is fitted on the same train window**, not on the evaluation cohort. The retrievers'
artefacts are read as they are, which is what production would do: stage 1 and stage 2 share a
training cut.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from rankstack.config import settings
from rankstack.data.corpus import CORPORA, DEFAULT_CORPUS, get_corpus
from rankstack.data.encode import load_encoding
from rankstack.logging import configure_logging, get_logger
from rankstack.rerank.candidates import CandidateSet, merge_sources, recall_ceiling
from rankstack.rerank.features import (
    FEATURE_NAMES,
    ItemFeatures,
    assemble,
    load_item_features,
    user_aggregates,
    write_feature_schema,
)
from rankstack.rerank.model import RankerConfig, RuleConfig, train_ranker
from rankstack.retrieval.als import DEFAULT_MIN_USER_HISTORY
from rankstack.retrieval.base import CandidateSource, top_k
from rankstack.retrieval.cooccurrence import CooccurrenceModel
from rankstack.retrieval.popularity import PopularityModel

if TYPE_CHECKING:
    from pathlib import Path

_logger = get_logger(__name__)

DEFAULT_TRAIN_USERS = 120_000
DEFAULT_CANDIDATES_PER_SOURCE = 120
DEFAULT_VALIDATION_USERS = 20_000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, choices=sorted(CORPORA))
    parser.add_argument("--train-users", type=int, default=DEFAULT_TRAIN_USERS)
    parser.add_argument("--validation-users", type=int, default=DEFAULT_VALIDATION_USERS)
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES_PER_SOURCE)
    parser.add_argument("--trees", type=int, default=400)
    return parser.parse_args(argv)


def _score(
    model: CandidateSource, user_code: int, seen: np.ndarray, history: np.ndarray
) -> np.ndarray:
    """Score every catalogue item for one training user, without letting the holdout leak.

    The co-occurrence source is history-based, and the held-out item is still inside the history the
    model was trained on. Scoring from an explicit history that excludes it is what makes
    leave-one-out honest: otherwise the target boosts itself through its own co-occurrence row.
    """
    if isinstance(model, CooccurrenceModel):
        return model.score_with_history(history, seen)
    return model.score(user_code, seen)


def _retriever_scores(
    entry: CandidateSet,
    models: dict[str, CandidateSource],
    user_code: int,
    seen: np.ndarray,
    history: np.ndarray,
) -> None:
    """Fill `entry.scores` with each retriever's score for that candidate.

    The whole catalogue is scored once per candidate list, then gathered down to the candidates.
    Re-deriving each candidate's score from stored factors would avoid the full pass, but the
    retrievers only expose `score(user, seen)`, and duplicating their internals here would let the
    features drift away from what the source actually computes.
    """
    for name, model in models.items():
        entry.scores[name] = _score(model, user_code, seen, history)[entry.items].astype(np.float32)


def build_dataset(
    users: np.ndarray,
    history_ptr: np.ndarray,
    history_codes: np.ndarray,
    models: dict[str, CandidateSource],
    item_features: ItemFeatures,
    candidates_per_source: int,
    n_items: int,
    chunk: int = 5_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Assemble `(features, labels, group_sizes)` and the candidate ceiling for a user sample."""
    feature_blocks, label_blocks, group_blocks = [], [], []
    candidate_lists: list[np.ndarray] = []
    targets: list[int] = []
    missed = 0
    for start in range(0, users.size, chunk):
        features, labels, groups, lists, block_targets, block_missed = _build_block(
            users[start : start + chunk],
            history_ptr,
            history_codes,
            models,
            item_features,
            candidates_per_source,
            n_items,
        )
        feature_blocks.append(features)
        label_blocks.append(labels)
        group_blocks.append(groups)
        candidate_lists.extend(lists)
        targets.extend(block_targets)
        missed += block_missed
        _logger.info(
            "rerank.block",
            extra={"done": min(start + chunk, users.size), "total": int(users.size)},
        )

    considered = max(missed + len(candidate_lists), 1)
    ceiling = recall_ceiling(candidate_lists, np.asarray(targets, dtype=np.int64))
    # The ceiling is measured over every eligible user, not only the ones kept for training: a user
    # whose target stage 1 missed contributes no positive example, and hiding that would make the
    # ranker look like it had solved retrieval.
    ceiling["candidate_recall"] = round(len(candidate_lists) / considered, 6)
    ceiling["eligible_users"] = float(considered)
    return (
        np.concatenate(feature_blocks),
        np.concatenate(label_blocks),
        np.concatenate(group_blocks),
        ceiling,
    )


def main(argv: list[str] | None = None) -> int:
    """Build stage-2 training data and fit the ranker."""
    args = parse_args(argv)
    configure_logging()
    corpus = get_corpus(args.corpus)
    root = settings.artifact_root / corpus.name
    processed = settings.processed(corpus.name)
    if not (root / "cooccurrence.npz").exists():
        print("error: run scripts/train_baselines.py first", file=sys.stderr)
        return 1

    started = time.perf_counter()
    encoding = load_encoding(settings.artifact_root, corpus.name)
    manifest = json.loads((processed / "manifest.json").read_text())
    train_end_ms = int(manifest["boundaries"]["train_end_ms"])

    history = np.load(root / "cooccurrence.history.npz")
    history_ptr = np.asarray(history["history_indptr"])
    history_codes = np.asarray(history["history_codes"])
    n_items = encoding.n_items

    models: dict[str, CandidateSource] = {
        "popularity": PopularityModel.load(root / "popularity.npy"),
        "popularity_recent": PopularityModel.load(root / "popularity_recent.npy"),
        "cooccurrence": _load_cooccurrence(root),
    }
    item_features = load_item_features(processed, train_end_ms, encoding.item_to_code)
    write_feature_schema(root / "rerank_features.json")

    eligible = np.flatnonzero(np.diff(history_ptr) >= DEFAULT_MIN_USER_HISTORY)
    rng = np.random.default_rng(settings.seed)
    sample = rng.choice(eligible, size=min(args.train_users, eligible.size), replace=False)
    sample.sort()

    features, all_labels, group_sizes, ceiling = build_dataset(
        sample,
        history_ptr,
        history_codes,
        models,
        item_features,
        args.candidates,
        n_items,
    )
    _logger.info(
        "rerank.dataset",
        extra={"rows": int(features.shape[0]), "users": int(group_sizes.size), **ceiling},
    )

    ranker = train_ranker(
        features,
        all_labels,
        group_sizes,
        FEATURE_NAMES,
        RankerConfig(n_estimators=args.trees, seed=settings.seed),
    )
    ranker.save(root / "reranker.npz")

    payload = {
        "corpus": corpus.name,
        "command": (
            f"uv run python scripts/train_reranker.py --corpus {corpus.name} "
            f"--train-users {args.train_users} --candidates {args.candidates}"
        ),
        "rows": int(features.shape[0]),
        "users": int(group_sizes.size),
        "candidates_per_source": args.candidates,
        "positive_rate": round(float(all_labels.mean()), 6),
        "as_of_ms": train_end_ms,
        "candidate_ceiling": ceiling,
        "ranker": RankerConfig(n_estimators=args.trees, seed=settings.seed).as_dict(),
        "rules": RuleConfig().as_dict(),
        "feature_importance": ranker.importance,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    out = settings.results_root / f"reranker_{corpus.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _load_cooccurrence(root: Path) -> CooccurrenceModel:
    """Load the trained co-occurrence neighbour matrix and its capped history."""
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


def _build_block(
    users: np.ndarray,
    history_ptr: np.ndarray,
    history_codes: np.ndarray,
    models: dict[str, CandidateSource],
    item_features: ItemFeatures,
    candidates_per_source: int,
    n_items: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], list[int], int]:
    """Build one chunk of `(features, labels, group_sizes)` plus the candidate lists."""
    feature_blocks, label_blocks, group_blocks = [], [], []
    kept_lists: list[np.ndarray] = []
    kept_targets: list[int] = []

    for raw_user in users:
        user = int(raw_user)
        start, end = int(history_ptr[user]), int(history_ptr[user + 1])
        target = int(history_codes[start])
        seen = np.zeros(n_items, dtype=bool)
        seen[history_codes[start + 1 : end]] = True

        ranked: dict[str, np.ndarray] = {}
        history = history_codes[start + 1 : end]
        for name, model in models.items():
            ranked[name] = top_k(_score(model, user, seen, history), candidates_per_source)[0]

        items, agreement = merge_sources(ranked, seen)
        if items.size == 0 or not np.any(items == target):
            # No candidate is the held-out item, so this user contributes no learning signal and,
            # more importantly, would otherwise teach the ranker that nothing is ever relevant.
            continue

        entry = CandidateSet(user_code=user, items=items, scores={}, target=target)
        _retriever_scores(entry, models, user, seen, history_codes[start + 1 : end])
        user_row = np.full(items.size, user, dtype=np.int64)
        aggregates = user_aggregates(user_row, history_ptr, history_codes, item_features)
        features = assemble(user_row, items, entry.scores, item_features, aggregates)
        labels = (items == target).astype(np.int32)
        feature_blocks.append(features)
        label_blocks.append(labels)
        group_blocks.append(np.asarray([items.size], dtype=np.int32))
        kept_lists.append(items)
        kept_targets.append(target)
        del agreement

    if not feature_blocks:
        return (
            np.empty((0, len(FEATURE_NAMES)), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            [],
            [],
            int(users.size),
        )
    return (
        np.concatenate(feature_blocks),
        np.concatenate(label_blocks),
        np.concatenate(group_blocks),
        kept_lists,
        kept_targets,
        int(users.size) - len(kept_lists),
    )


if __name__ == "__main__":
    raise SystemExit(main())
