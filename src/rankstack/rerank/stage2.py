"""The two-stage recommender, as a `CandidateSource` the existing harness can score.

Stage 1 proposes candidates; LambdaMART reorders them; the rules trim the result. This class is the
only place that sequence exists, and it presents the same interface as every retriever, so the same
frozen cohort and the same metrics score it. That is the point: a two-stage system evaluated on a
different cohort than its own stage 1 is not comparable to anything.

**The evaluation domain is the candidate list.** A stage-2 model cannot rank an item stage 1 never
returned, so this source reports a finite score only for items in the user's candidate list and
`-inf` for the rest. `candidate_recall` records the ceiling that implies, and the harness result is
reported against it rather than as an unconditional number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rankstack.logging import get_logger
from rankstack.rerank.candidates import CandidateSet, recall_ceiling
from rankstack.rerank.features import (
    ItemFeatures,
    UserAggregates,
    assemble,
    user_aggregates,
)
from rankstack.rerank.model import Ranker, RuleConfig, apply_mmr, cap_categories, freshness_bonus

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class StageTwoContext:
    """Everything the rules need that is not in the feature matrix."""

    item_features: ItemFeatures
    history_indptr: np.ndarray
    history_codes: np.ndarray
    rules: RuleConfig = field(default_factory=RuleConfig)

    def aggregates(self, user_codes: np.ndarray) -> UserAggregates:
        """Return history aggregates for a batch of users."""
        return user_aggregates(
            user_codes, self.history_indptr, self.history_codes, self.item_features
        )


class TwoStageSource:
    """Re-ranks each user's candidate list, exposing the retriever interface.

    `similarity` is an optional `(n_items, n_items)`-free call-off: MMR needs item-item similarity,
    and with no learned item embedding the interaction profile is the only honest basis available.
    It is passed in rather than constructed here so the caller decides what similarity means.
    """

    def __init__(
        self,
        ranker: Ranker,
        candidates: dict[int, CandidateSet],
        context: StageTwoContext,
        *,
        similarity: np.ndarray | None = None,
        use_rules: bool = True,
    ) -> None:
        """Bind a ranker, its candidates and the context the rules need."""
        self._ranker = ranker
        self._candidates = candidates
        self._context = context
        self._similarity = similarity
        self._use_rules = use_rules

    @property
    def name(self) -> str:
        """Name reported in results files."""
        return "two_stage"

    def n_items(self) -> int:
        """Return the size of the catalogue this source scores over."""
        return self._context.item_features.n_items()

    def can_serve(self, user_code: int) -> bool:
        """Return True when stage 1 produced at least one candidate for this user."""
        return user_code in self._candidates

    def ordered(self, user_code: int) -> tuple[np.ndarray, np.ndarray]:
        """Return this user's candidates and their final scores, best-first.

        `score` and `rank` both go through here so a candidate is never feature-extracted twice.
        """
        empty = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32))
        entry = self._candidates.get(int(user_code))
        if entry is None or not len(entry):
            return empty

        items = entry.items
        scores = self._model_scores(int(user_code), entry)
        if not self._use_rules:
            keep = np.argsort(-scores, kind="stable")
            return items[keep], scores[keep]

        items_features = self._context.item_features
        scores = scores + freshness_bonus(
            items_features.days_since_first[items],
            weight=self._context.rules.freshness_weight,
            halflife_days=self._context.rules.freshness_halflife_days,
        )
        keep = np.argsort(-scores, kind="stable")
        items, scores = items[keep], scores[keep]
        capped = cap_categories(
            items, items_features.category[items], items.size, cap=self._context.rules.category_cap
        )
        items, scores = items[capped], scores[capped]
        if self._similarity is not None:
            chosen = apply_mmr(
                items,
                scores,
                self._similarity[items[:, None], items[None, :]],
                items.size,
                mmr_lambda=self._context.rules.mmr_lambda,
            )
            items, scores = items[chosen], scores[chosen]
        return items, scores

    def rank(self, user_code: int) -> np.ndarray:
        """Return this user's candidates ordered best-first, after the rules."""
        return self.ordered(user_code)[0]

    def _model_scores(self, user_code: int, entry: CandidateSet) -> np.ndarray:
        items = entry.items
        user_row = np.full(items.size, int(user_code), dtype=np.int64)
        features = assemble(
            user_row,
            items,
            entry.scores,
            self._context.item_features,
            self._context.aggregates(user_row),
        )
        return self._ranker.predict(features).astype(np.float32)

    def score(self, user_code: int, seen: np.ndarray) -> np.ndarray:
        """Score the whole catalogue, finite only for this user's candidates."""
        scores = np.full(self.n_items(), -np.inf, dtype=np.float32)
        entry = self._candidates.get(int(user_code))
        if entry is not None and len(entry):
            ranked, ranked_scores = self.ordered(int(user_code))
            position = {int(item): index for index, item in enumerate(entry.items)}
            del position
            for item, value in zip(ranked, ranked_scores, strict=True):
                scores[int(item)] = float(value)
        scores[seen] = -np.inf
        return scores

    def candidate_recall(self) -> dict[str, float]:
        """Return the recall of stage 1, which is the ceiling on every stage-2 metric."""
        entries = [entry for entry in self._candidates.values() if entry.target is not None]
        if not entries:
            return {}
        return recall_ceiling(
            [entry.items for entry in entries], np.array([e.target for e in entries])
        )
