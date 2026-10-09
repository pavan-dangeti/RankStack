"""The two-stage recommender on the synthetic corpus.

The assertions are about the contract rather than about accuracy: five users cannot train a useful
ranker, so what matters is that stage 2 only ranks what stage 1 returned, that it never leaks the
held-out item through a self-count, that the rules are applied, and that the ceiling is reported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb
import numpy as np
import pytest

from rankstack.data.encode import build_cohort
from rankstack.eval.harness import CohortHistory, EvalConfig, evaluate_source
from rankstack.rerank.candidates import CandidateSet, merge_sources, recall_ceiling
from rankstack.rerank.features import FEATURE_NAMES, assemble, load_item_features, user_aggregates
from rankstack.rerank.model import (
    RankerConfig,
    RuleConfig,
    train_ranker,
)
from rankstack.rerank.stage2 import StageTwoContext, TwoStageSource
from rankstack.retrieval.base import top_k

if TYPE_CHECKING:
    from tests.conftest import Trained


def build_candidates(trained: Trained, per_source: int = 20) -> dict[int, CandidateSet]:
    """Propose candidates for every user, holding out each user's most recent train item."""
    encoding = trained.encoding
    cooccurrence = trained.cooccurrence
    entries: dict[int, CandidateSet] = {}
    for user in range(encoding.n_users):
        start, end = (
            int(cooccurrence.history_indptr[user]),
            int(cooccurrence.history_indptr[user + 1]),
        )
        if end - start < 2:
            continue
        history = cooccurrence.history_codes[start + 1 : end]
        target = int(cooccurrence.history_codes[start])
        seen = np.zeros(encoding.n_items, dtype=bool)
        seen[history] = True

        ranked, scores = {}, {}
        for name, model in (
            ("popularity", trained.popularity),
            ("cooccurrence", cooccurrence),
        ):
            full = (
                model.score_with_history(history, seen)
                if hasattr(model, "score_with_history")
                else model.score(user, seen)
            )
            scores[name] = full.astype(np.float32)
            ranked[name] = top_k(full, per_source)[0]
        items, _ = merge_sources(ranked, seen)
        entries[user] = CandidateSet(
            user_code=user,
            items=items,
            scores={name: values[items] for name, values in scores.items()},
            target=target,
        )
    return entries


@pytest.fixture
def stage_two(trained: Trained) -> TwoStageSource:
    """Build a tiny ranker over the synthetic corpus, wired through the real stage-2 source."""
    items = load_item_features(
        trained.processed, trained.boundaries["train_end_ms"], trained.encoding.item_to_code
    )
    entries = build_candidates(trained)

    feature_blocks, label_blocks, group_blocks = [], [], []
    for user, entry in entries.items():
        if not entry.hits():
            continue
        rows = np.full(len(entry), user, dtype=np.int64)
        aggregates = user_aggregates(
            rows, trained.cooccurrence.history_indptr, trained.cooccurrence.history_codes, items
        )
        feature_blocks.append(assemble(rows, entry.items, entry.scores, items, aggregates))
        label_blocks.append((entry.items == entry.target).astype(np.int32))
        group_blocks.append(np.asarray([len(entry)], dtype=np.int32))
    ranker = train_ranker(
        np.concatenate(feature_blocks),
        np.concatenate(label_blocks),
        np.concatenate(group_blocks),
        FEATURE_NAMES,
        RankerConfig(n_estimators=10, seed=trained.settings.seed),
    )
    similarity = np.asarray(trained.cooccurrence.similarity.todense(), dtype=np.float32)
    return TwoStageSource(
        ranker,
        entries,
        StageTwoContext(
            item_features=items,
            history_indptr=trained.cooccurrence.history_indptr,
            history_codes=trained.cooccurrence.history_codes,
            rules=RuleConfig(),
        ),
        similarity=similarity,
    )


def test_stage_two_is_finite_only_inside_the_candidate_list(stage_two: TwoStageSource) -> None:
    user = next(iter(stage_two._candidates))
    entry = stage_two._candidates[user]
    seen = np.zeros(stage_two.n_items(), dtype=bool)
    scores = stage_two.score(user, seen)
    finite = set(np.flatnonzero(np.isfinite(scores)).tolist())
    assert finite == set(entry.items.tolist())


def test_stage_two_never_scores_an_item_the_user_already_saw(stage_two: TwoStageSource) -> None:
    user = next(iter(stage_two._candidates))
    seen = np.zeros(stage_two.n_items(), dtype=bool)
    seen[stage_two._candidates[user].items[:2]] = True
    scores = stage_two.score(user, seen)
    assert not np.isfinite(scores[seen]).any()


def test_stage_two_reports_the_retrieval_ceiling(stage_two: TwoStageSource) -> None:
    ceiling = stage_two.candidate_recall()
    assert 0.0 <= ceiling["recall_full_list"] <= 1.0
    assert ceiling["mean_candidates"] > 0


def test_rules_change_the_order_the_ranker_produced(stage_two: TwoStageSource) -> None:
    """The rules must actually fire, otherwise `--no-rules` would be measuring nothing."""
    plain = TwoStageSource(
        stage_two._ranker,
        stage_two._candidates,
        stage_two._context,
        use_rules=False,
    )
    differences = 0
    for user in list(stage_two._candidates)[:20]:
        if not np.array_equal(plain.rank(user), stage_two.rank(user)):
            differences += 1
    assert differences > 0, "the rules never changed an ordering"


def test_stage_two_runs_through_the_shared_harness(
    trained: Trained, stage_two: TwoStageSource
) -> None:
    """Same harness, same cohort, same metric definitions as every retriever."""
    connection = duckdb.connect(database=":memory:")
    cohort = build_cohort(
        connection,
        "all_test_users",
        trained.encoding,
        trained.processed,
        size=100,
        seed=5,
    )
    connection.close()

    users = np.asarray(cohort.user_codes, dtype=np.int64)
    targets = np.asarray(cohort.target_codes, dtype=np.int64)
    popularity = np.bincount(
        trained.cooccurrence.history_codes.astype(np.int64), minlength=trained.encoding.n_items
    ).astype(np.float32)
    history = CohortHistory(
        indptr=trained.cooccurrence.history_indptr,
        codes=trained.cooccurrence.history_codes,
        n_items=trained.encoding.n_items,
        popularity=popularity,
    )
    result = evaluate_source(
        stage_two,
        users,
        targets,
        history,
        EvalConfig(name="two_stage", cohort_name="all_test_users"),
    )
    assert result.ranking.users == len(cohort)
    assert 0.0 <= result.ranking.mean("hr")["hr@20"] <= 1.0


def test_recall_ceiling_counts_targets_the_candidates_missed() -> None:
    """A target outside every list must count against the ceiling, not be dropped."""
    entries = [
        CandidateSet(user_code=0, items=np.array([1, 2]), scores={}, target=1),
        CandidateSet(user_code=1, items=np.array([3, 4]), scores={}, target=99),
    ]
    ceiling = recall_ceiling([entry.items for entry in entries], np.array([1, 99]))
    assert ceiling["recall_full_list"] == 0.5
