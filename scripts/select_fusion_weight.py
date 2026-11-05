"""Choose the served co-occurrence weight by the rule pre-registered in ADR-0007.

    uv run python scripts/select_fusion_weight.py sweep     # validation split, every weight
    uv run python scripts/select_fusion_weight.py confirm   # test split, the selected weight, once

Every candidate is built by `ServingArtifacts.served_source`, so what is measured is exactly what
the service would serve at that weight. `sweep` applies the selection rule and records the choice;
`confirm` reads that choice and refuses to run if a test result already exists, because the test
split is evaluated once.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import duckdb
import numpy as np

from rankstack.config import settings
from rankstack.data.corpus import CORPORA, DEFAULT_CORPUS, get_corpus
from rankstack.data.encode import build_cohort
from rankstack.eval.harness import CohortHistory, EvalConfig, SourceResult, evaluate_source
from rankstack.eval.stats import bootstrap_mean, paired_bootstrap
from rankstack.logging import configure_logging
from rankstack.serving.artifacts import load_artifacts

if TYPE_CHECKING:
    from pathlib import Path

    from rankstack.serving.artifacts import ServingArtifacts

WEIGHTS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
COHORT_SIZE = 2_000
VALID_COHORT = "warm_users_valid"
TEST_COHORT = "warm_users"
K = 20
RESAMPLES = 1_000
RULE = "docs/adr/0007-serving-fusion-weight.md"


@dataclass(frozen=True, slots=True)
class Candidate:
    """What the selection rule needs to know about one weight."""

    weight: float
    qualifies: bool
    coverage_items: int


def select_weight(candidates: list[Candidate]) -> float:
    """Apply ADR-0007: the highest coverage@20 among qualifying weights, ties to the smaller weight.

    w = 0 always qualifies, so when no positive weight does this returns 0, which serves
    `popularity_recent` alone.
    """
    qualifying = [candidate for candidate in candidates if candidate.qualifies]
    if not qualifying:
        return 0.0
    return max(qualifying, key=lambda c: (c.coverage_items, -c.weight)).weight


def summarise(
    weight: float, result: SourceResult, baseline: SourceResult, n_items: int
) -> dict[str, object]:
    """Return the four pre-registered quantities for one evaluated weight."""
    k_index = result.ranking.ks.index(K)
    hits = result.per_user_values["hr"][k_index]
    interval = bootstrap_mean(hits, resamples=RESAMPLES, seed=settings.seed)
    paired = paired_bootstrap(
        hits,
        baseline.per_user_values["hr"][k_index],
        metric=f"hr@{K}",
        resamples=RESAMPLES,
        seed=settings.seed,
    )
    lists = [codes for codes in result.recommendations.values() if codes.size]
    covered = int(np.unique(np.concatenate([codes[:K] for codes in lists])).size) if lists else 0
    return {
        "weight": weight,
        f"hr@{K}": interval.as_dict(),
        f"paired_hr@{K}_vs_popularity_recent": paired.as_dict(),
        # Decided on the unrounded interval, so rounding in the file cannot flip a borderline case.
        "qualifies": bool(paired.low <= 0.0 <= paired.high),
        "distinct_top_1": len({int(codes[0]) for codes in lists}),
        f"coverage_at_{K}": {"items": covered, "fraction": round(covered / n_items, 8)},
    }


def _evaluate(
    artifacts: ServingArtifacts, cohort_name: str, weights: tuple[float, ...]
) -> tuple[dict[float, SourceResult], SourceResult, int]:
    """Evaluate `popularity_recent` and each weight on one sampled cohort."""
    connection = duckdb.connect(database=":memory:")
    cohort = build_cohort(
        connection,
        cohort_name,
        artifacts.encoding,
        settings.processed(artifacts.corpus),
        size=COHORT_SIZE,
        seed=settings.seed,
    )
    connection.close()
    users = np.asarray(cohort.user_codes, dtype=np.int64)
    targets = np.asarray(cohort.target_codes, dtype=np.int64)
    view = CohortHistory(
        indptr=artifacts.history_indptr,
        codes=artifacts.history_codes,
        n_items=artifacts.n_items,
        popularity=np.bincount(
            artifacts.history_codes.astype(np.int64), minlength=artifacts.n_items
        ).astype(np.float32),
    )
    config = EvalConfig(name="fusion_weight", cohort_name=cohort_name)
    baseline = evaluate_source(artifacts.popularity_recent, users, targets, view, config)
    results = {
        weight: evaluate_source(
            replace(artifacts, cooccurrence_weight=weight).served_source(),
            users,
            targets,
            view,
            config,
        )
        for weight in weights
    }
    return results, baseline, int(users.size)


def _common(artifacts: ServingArtifacts, cohort_name: str, cohort_size: int) -> dict[str, object]:
    return {
        "corpus": artifacts.corpus,
        "rule": RULE,
        "cohort": cohort_name,
        "cohort_size": cohort_size,
        "seed": settings.seed,
        "resamples": RESAMPLES,
        "k": K,
        "catalogue_items": artifacts.n_items,
    }


def sweep(artifacts: ServingArtifacts, out_dir: Path) -> Path:
    """Evaluate every weight on the validation split and record the rule's choice."""
    started = time.perf_counter()
    results, baseline, size = _evaluate(artifacts, VALID_COHORT, WEIGHTS)
    rows = [summarise(w, results[w], baseline, artifacts.n_items) for w in WEIGHTS]
    selected = select_weight(
        [
            Candidate(
                weight=float(row["weight"]),  # type: ignore[arg-type]
                qualifies=bool(row["qualifies"]),
                coverage_items=int(row[f"coverage_at_{K}"]["items"]),  # type: ignore[index]
            )
            for row in rows
        ]
    )
    payload = {
        **_common(artifacts, VALID_COHORT, size),
        "command": "uv run python scripts/select_fusion_weight.py sweep",
        "weights": rows,
        "by_weight": {str(row["weight"]): row for row in rows},
        "selected_weight": selected,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    path = out_dir / f"fusion_weight_sweep_{artifacts.corpus}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def confirm(artifacts: ServingArtifacts, out_dir: Path) -> Path:
    """Evaluate the selected weight once on the test split."""
    sweep_path = out_dir / f"fusion_weight_sweep_{artifacts.corpus}.json"
    path = out_dir / f"fusion_weight_test_{artifacts.corpus}.json"
    if path.exists():
        msg = f"{path.name} exists: the test split is evaluated once, by ADR-0007"
        raise FileExistsError(msg)
    selected = float(json.loads(sweep_path.read_text())["selected_weight"])
    started = time.perf_counter()
    results, baseline, size = _evaluate(artifacts, TEST_COHORT, (selected,))
    payload = {
        **_common(artifacts, TEST_COHORT, size),
        "command": "uv run python scripts/select_fusion_weight.py confirm",
        "selected_weight": selected,
        "selected_in": sweep_path.name,
        "result": summarise(selected, results[selected], baseline, artifacts.n_items),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    """Run the sweep or the confirmation and print where the result went."""
    parser = argparse.ArgumentParser(description="Choose the served fusion weight by ADR-0007.")
    parser.add_argument("mode", choices=("sweep", "confirm"))
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, choices=sorted(CORPORA))
    args = parser.parse_args(argv)
    configure_logging()
    corpus = get_corpus(args.corpus)
    artifacts = load_artifacts(settings.artifact_root, corpus.name, with_titles=False)
    if artifacts.cooccurrence is None:
        print("error: no co-occurrence model; run scripts/train_baselines.py", file=sys.stderr)
        return 1
    settings.results_root.mkdir(parents=True, exist_ok=True)
    try:
        path = (sweep if args.mode == "sweep" else confirm)(artifacts, settings.results_root)
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
