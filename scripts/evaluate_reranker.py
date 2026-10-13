"""Evaluate the two-stage recommender against the retriever that feeds it.

The two-stage model is scored by the *same* harness, on the *same* frozen cohort, as every
retriever. That matters more than usual here: a stage-2 number measured on a different cohort, or
over a different candidate list, is not evidence that the re-ranker helped.

**The candidate ceiling is reported next to the metrics.** Stage 2 cannot rank an item stage 1
never returned, so the meaningful comparison is whether re-ranking moved the target up *within* the
list stage 1 produced. The ceiling says how much was available to move.

`--rules` off scores LambdaMART alone, which is the comparison that isolates the model from the
hand-written post-ranking rules.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import TYPE_CHECKING, Any, cast

import duckdb
import numpy as np
import scipy.sparse as sp

from rankstack.config import settings
from rankstack.data.corpus import CORPORA, DEFAULT_CORPUS, Corpus, get_corpus
from rankstack.data.encode import build_cohort, load_encoding
from rankstack.eval.harness import CohortHistory, EvalConfig, compare_sources, evaluate_source
from rankstack.logging import configure_logging, get_logger
from rankstack.rerank.candidates import CandidateSet, merge_sources, recall_ceiling
from rankstack.rerank.features import load_item_features
from rankstack.rerank.model import Ranker, RuleConfig
from rankstack.rerank.stage2 import StageTwoContext, TwoStageSource
from rankstack.retrieval.base import top_k
from rankstack.retrieval.cooccurrence import CooccurrenceModel
from rankstack.retrieval.popularity import PopularityModel

if TYPE_CHECKING:
    from pathlib import Path

    from rankstack.retrieval.base import CandidateSource

_logger = get_logger(__name__)

DEFAULT_COHORT_SIZE = 20_000
DEFAULT_PROBE_USERS = 2_000
DEFAULT_CANDIDATES = 120


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, choices=sorted(CORPORA))
    parser.add_argument("--cohorts", nargs="+", default=["warm_users", "all_test_users"])
    parser.add_argument("--cohort-size", type=int, default=DEFAULT_COHORT_SIZE)
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument("--resamples", type=int, default=1_000)
    parser.add_argument(
        "--probe-users",
        type=int,
        default=DEFAULT_PROBE_USERS,
        help="Users to probe for target movement; each needs a full candidate list.",
    )
    parser.add_argument(
        "--no-rules",
        action="store_true",
        help="Score LambdaMART alone, isolating it from the post-ranking rules.",
    )
    return parser.parse_args(argv)


def load_sources(root: Path, names: list[str]) -> dict[str, CandidateSource]:
    """Load the trained retrievers that propose candidates and supply features."""
    loaded: dict[str, CandidateSource] = {}
    for name in names:
        if name == "popularity":
            loaded[name] = PopularityModel.load(root / "popularity.npy")
        elif name == "popularity_recent":
            loaded[name] = PopularityModel.load(root / "popularity_recent.npy")
        elif name == "cooccurrence":
            loaded[name] = _load_cooccurrence(root)
        else:
            msg = f"unknown source {name!r}"
            raise KeyError(msg)
    return loaded


def _load_cooccurrence(root: Path) -> CooccurrenceModel:
    """Load the trained co-occurrence neighbours and the capped history they were built from."""
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


def build_candidates(
    users: np.ndarray,
    targets: np.ndarray,
    models: dict[str, CandidateSource],
    history_ptr: np.ndarray,
    history_codes: np.ndarray,
    n_items: int,
    candidates_per_source: int,
) -> dict[int, CandidateSet]:
    """Stage 1: propose candidates for each cohort user and record the retrievers' scores."""
    entries: dict[int, CandidateSet] = {}
    for raw_user, target in zip(users, targets, strict=True):
        user = int(raw_user)
        start, end = int(history_ptr[user]), int(history_ptr[user + 1])
        seen = np.zeros(n_items, dtype=bool)
        seen[history_codes[start:end]] = True

        ranked: dict[str, np.ndarray] = {}
        scores: dict[str, np.ndarray] = {}
        for name, model in models.items():
            full = model.score(user, seen)
            scores[name] = full.astype(np.float32)
            ranked[name] = top_k(full, candidates_per_source)[0]
        items, agreement = merge_sources(ranked, seen)
        entries[user] = CandidateSet(
            user_code=user,
            items=items,
            scores={name: values[items] for name, values in scores.items()},
            target=int(target),
        )
        del agreement
    return entries


def probe_target_movement(
    entries: dict[int, CandidateSet],
    two_stage: TwoStageSource,
    users: np.ndarray,
    limit: int,
) -> dict[str, float]:
    """Count whether re-ranking moved each user's target up or down within the candidate list.

    Only users whose target stage 1 actually returned are counted. For everyone else the question is
    meaningless: the item was not in the list to be moved. Positions are 0-based, so "up" is a
    smaller index.
    """
    positions_before: list[float] = []
    positions_after: list[float] = []
    moved_up = 0
    moved_down = 0
    unchanged = 0

    for raw_user in users:
        entry = entries.get(int(raw_user))
        if entry is None or entry.target is None or entry.target < 0:
            continue
        hits = np.flatnonzero(entry.items == entry.target)
        if hits.size == 0:
            continue
        if len(positions_before) >= limit:
            break

        before = int(hits[0])
        after = int(np.flatnonzero(two_stage.rank(int(raw_user)) == entry.target)[0])
        positions_before.append(before)
        positions_after.append(after)
        if after < before:
            moved_up += 1
        elif after > before:
            moved_down += 1
        else:
            unchanged += 1

    considered = moved_up + moved_down + unchanged
    return {
        "users_with_target_in_list": considered,
        "moved_up": moved_up,
        "moved_down": moved_down,
        "unchanged": unchanged,
        "median_position_before": float(np.median(positions_before)) if positions_before else -1.0,
        "median_position_after": float(np.median(positions_after)) if positions_after else -1.0,
        "net_movement": moved_up - moved_down,
    }


def _command(args: argparse.Namespace) -> str:
    """Return the command that reproduces this file, flags included.

    Hand-written command strings go stale the moment a flag is added: an earlier version
    recorded the no-rules ablation without `--no-rules`, so the file claimed a command that
    produced something else. Building it from argv means the recorded command is the command.
    """
    flags = [
        f"--corpus {args.corpus}",
        f"--candidates {args.candidates}",
        f"--cohort-size {args.cohort_size}",
        f"--resamples {args.resamples}",
    ]
    if args.no_rules:
        flags.append("--no-rules")
    if args.probe_users != DEFAULT_PROBE_USERS:
        flags.append(f"--probe-users {args.probe_users}")
    return "uv run python scripts/evaluate_reranker.py " + " ".join(flags)


def main(argv: list[str] | None = None) -> int:
    """Evaluate stage 2 against its stage-1 retriever and write the result file."""
    args = parse_args(argv)
    configure_logging()
    corpus = get_corpus(args.corpus)
    return evaluate(corpus, args)


def evaluate(corpus: Corpus, args: argparse.Namespace) -> int:
    """Run the two-stage evaluation and write `results/two_stage_<corpus>.json`."""
    started = time.perf_counter()
    root = settings.artifact_root / corpus.name
    processed = settings.processed(corpus.name)
    if not (root / "reranker.txt").exists():
        print("error: run scripts/train_reranker.py first", file=sys.stderr)
        return 1

    encoding = load_encoding(settings.artifact_root, corpus.name)
    manifest = json.loads((processed / "manifest.json").read_text())
    train_end_ms = int(manifest["boundaries"]["train_end_ms"])

    models = load_sources(root, ["popularity", "popularity_recent", "cooccurrence"])
    history = np.load(root / "cooccurrence.history.npz")
    history_ptr = np.asarray(history["history_indptr"])
    history_codes = np.asarray(history["history_codes"])
    popularity = np.bincount(history_codes.astype(np.int64), minlength=encoding.n_items).astype(
        np.float32
    )
    item_features = load_item_features(processed, train_end_ms, encoding.item_to_code)
    ranker = Ranker.load(root / "reranker.npz")
    context = StageTwoContext(
        item_features=item_features,
        history_indptr=history_ptr,
        history_codes=history_codes,
        rules=RuleConfig(),
    )
    history_view = CohortHistory(
        indptr=history_ptr,
        codes=history_codes,
        n_items=encoding.n_items,
        popularity=popularity,
    )

    connection = duckdb.connect(database=":memory:")
    cohorts: dict[str, object] = {}
    payload: dict[str, object] = {
        "corpus": corpus.name,
        "command": _command(args),
        "candidates_per_source": args.candidates,
        "rules_enabled": not args.no_rules,
        "metric_note": (
            "stage 2 ranks only what stage 1 returned; the candidate ceiling is reported alongside"
        ),
        "feature_importance": ranker.importance,
        "cohorts": cohorts,
    }

    for cohort_name in args.cohorts:
        cohort = build_cohort(
            connection,
            cohort_name,
            encoding,
            processed,
            size=args.cohort_size,
            seed=settings.seed,
        )
        users = np.asarray(cohort.user_codes, dtype=np.int64)
        targets = np.asarray(cohort.target_codes, dtype=np.int64)
        entries = build_candidates(
            users,
            targets,
            models,
            history_ptr,
            history_codes,
            encoding.n_items,
            args.candidates,
        )
        ceiling = recall_ceiling(
            [entry.items for entry in entries.values()],
            np.asarray([entry.target for entry in entries.values()], dtype=np.int64),
        )

        config = EvalConfig(name="two_stage", cohort_name=cohort_name)
        stage_one = evaluate_source(
            models["popularity_recent"], users, targets, history_view, config
        )
        two_stage = TwoStageSource(ranker, entries, context, use_rules=not args.no_rules)
        stage_two = evaluate_source(two_stage, users, targets, history_view, config)

        cohorts[cohort_name] = {
            "definition": cohort.as_dict(),
            "candidate_ceiling": ceiling,
            "stage_1_popularity_recent": stage_one.as_dict(),
            "stage_2": stage_two.as_dict(),
            "paired_two_stage_vs_stage_1": compare_sources(
                stage_two, stage_one, resamples=args.resamples, seed=settings.seed
            ),
            "target_movement_probe": probe_target_movement(
                entries, two_stage, users, args.probe_users
            ),
        }
    connection.close()

    payload["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    suffix = "" if not args.no_rules else "_model_only"
    out = settings.results_root / f"two_stage{suffix}_{corpus.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print_table(payload)
    return 0


def print_table(payload: dict[str, object]) -> None:
    """Print the stage-1 versus stage-2 comparison to stdout."""
    cohorts = cast("dict[str, dict[str, Any]]", payload["cohorts"])
    for name, cohort in cohorts.items():
        ceiling = cohort["candidate_ceiling"]
        print(f"\n=== {name} (n={cohort['definition']['size']}) ===")
        print(
            f"  candidate ceiling recall: {ceiling['recall_full_list']:.4f}"
            f"  mean candidates: {ceiling['mean_candidates']:.0f}"
        )
        print(f"  {'source':>20s} {'served':>7s} {'HR@10':>8s} {'HR@20':>8s} {'HR@50':>8s}")
        for label, key in (
            ("stage_1", "stage_1_popularity_recent"),
            ("stage_2", "stage_2"),
        ):
            ranking = cohort[key]["ranking"]
            print(
                f"  {label:>20s} {ranking['served_fraction']:>7.3f} {ranking['hr@10']:>8.5f}"
                f" {ranking['hr@20']:>8.5f} {ranking['hr@50']:>8.5f}"
            )
        probe = cohort.get("target_movement_probe")
        if probe and probe["users_with_target_in_list"]:
            print(
                f"  probe: of {probe['users_with_target_in_list']} users whose target was in the"
                f" list, re-ranking moved it up {probe['moved_up']}x and down"
                f" {probe['moved_down']}x; median position"
                f" {probe['median_position_before']:.0f} -> {probe['median_position_after']:.0f}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
