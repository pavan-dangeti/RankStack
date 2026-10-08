"""Evaluation harness.

Runs every candidate source over a frozen cohort and produces per-user ranks, per-user top-K lists,
segment breakdowns, beyond-list metrics and paired comparisons against a baseline. Everything it
writes is derived from one pass over the cohort, so two sources are always compared on the same
users in the same order.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

from rankstack.eval.metrics import (
    DEFAULT_KS,
    RankingMetrics,
    coverage,
    gini,
    intra_list_diversity,
    new_item_share,
    novelty,
    popularity_bias,
    ranking_metrics,
    slice_metrics,
)
from rankstack.eval.segments import segment_masks
from rankstack.eval.stats import DEFAULT_SEED, PairedResult, paired_metric_comparison
from rankstack.logging import get_logger
from rankstack.retrieval.base import exact_rank, top_k

if TYPE_CHECKING:
    from rankstack.retrieval.base import CandidateSource

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """What to measure and how, reported in the result file."""

    name: str
    cohort_name: str
    ks: tuple[int, ...] = DEFAULT_KS
    top_k_size: int = 50
    profile_embeddings: np.ndarray | None = None
    keep_recommendations: bool = True

    def as_dict(self) -> dict[str, object]:
        """Return the configuration as a JSON-serialisable mapping."""
        return {
            "name": self.name,
            "cohort": self.cohort_name,
            "ks": list(self.ks),
            "top_k_size": self.top_k_size,
            "intra_list_diversity_basis": (
                "interaction_profile" if self.profile_embeddings is not None else None
            ),
        }


@dataclass(slots=True)
class SourceResult:
    """Everything measured for one source on one cohort."""

    source: str
    cohort: str
    ranking: RankingMetrics
    segments: dict[str, object]
    beyond_list: dict[str, object]
    recommendations: dict[int, np.ndarray] = field(default_factory=dict)
    per_user_ranks: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    per_user_values: dict[str, np.ndarray] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def as_dict(self, *, include_beyond_list: bool = True) -> dict[str, object]:
        """Return the measured numbers as a JSON-serialisable mapping."""
        payload: dict[str, object] = {
            "source": self.source,
            "cohort": self.cohort,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "ranking": self.ranking.as_dict(),
            "segments": self.segments,
        }
        if include_beyond_list:
            payload["beyond_list"] = self.beyond_list
        return payload


class HistoryProvider(Protocol):
    """Supplies the seen-item mask for a user."""

    def seen_mask(self, user_code: int) -> np.ndarray:
        """Boolean mask over item codes: True where the user already interacted with the item."""
        ...


class ArrayHistory:
    """History stored as a CSR structure over item codes."""

    def __init__(self, indptr: np.ndarray, codes: np.ndarray, n_items: int) -> None:
        """Build a history view over a CSR-style structure."""
        self._indptr = indptr
        self._codes = codes
        self._n_items = n_items

    def seen_mask(self, user_code: int) -> np.ndarray:
        """Return a boolean mask of the items this user already interacted with."""
        mask = np.zeros(self._n_items, dtype=bool)
        start, end = self._indptr[user_code], self._indptr[user_code + 1]
        mask[self._codes[start:end]] = True
        return mask

    def depth(self, user_code: int) -> int:
        """Return the number of train-window interactions for this user."""
        start, end = self._indptr[user_code], self._indptr[user_code + 1]
        return int(end - start)


class CohortHistory(ArrayHistory):
    """History plus the item popularity of each item, for segments and novelty."""

    def __init__(
        self, indptr: np.ndarray, codes: np.ndarray, n_items: int, popularity: np.ndarray
    ) -> None:
        """Build a history view that also carries per-item train popularity."""
        super().__init__(indptr, codes, n_items)
        self.popularity = popularity


def evaluate_source(
    source: CandidateSource,
    cohort_users: np.ndarray,
    cohort_targets: np.ndarray,
    history: CohortHistory,
    config: EvalConfig,
) -> SourceResult:
    """Score every cohort user and compute every metric defined in `rankstack.eval.metrics`.

    `served` records whether the source could produce any finite score for a user, which is how the
    57% of test users with no train history are reported rather than silently scored as misses.
    """
    started = time.perf_counter()
    name, cohort_name = config.name, config.cohort_name
    ks, top_k_size = config.ks, config.top_k_size
    n_items = source.n_items()
    ranks = np.full(cohort_users.size, -1, dtype=np.int64)
    served = np.zeros(cohort_users.size, dtype=bool)
    recommendations: dict[int, np.ndarray] = {}
    depths = np.zeros(cohort_users.size, dtype=np.int64)
    target_popularity = np.zeros(cohort_users.size, dtype=np.float64)

    max_k = max(*ks, top_k_size)
    pairs = zip(cohort_users, cohort_targets, strict=True)
    for position, (user_code, target_code) in enumerate(pairs):
        seen = history.seen_mask(int(user_code))
        depths[position] = int(seen.sum())
        popularity = history.popularity[target_code] if target_code >= 0 else 0.0
        target_popularity[position] = float(popularity)
        scores = source.score(int(user_code), seen)
        if np.isfinite(scores).any():
            served[position] = True
        ranks[position] = exact_rank(scores, int(target_code))
        if config.keep_recommendations:
            codes, _ = top_k(scores, max_k)
            recommendations[int(user_code)] = codes[:top_k_size]

    values = ranking_metrics(ranks, served, ks=ks)
    masks = segment_masks(depths, target_popularity)
    segments: dict[str, object] = {
        "new_user": slice_metrics(values, masks.new_user),
        "light_user": slice_metrics(values, masks.light_user),
        "heavy_user": slice_metrics(values, masks.heavy_user),
        "long_tail_target": slice_metrics(values, masks.long_tail_target),
        "popular_target": slice_metrics(values, masks.popular_target),
    }
    segments["segment_sizes"] = masks.as_dict()

    recommended_codes = (
        np.concatenate([codes for codes in recommendations.values() if codes.size])
        if any(codes.size for codes in recommendations.values())
        else np.empty(0, dtype=np.int64)
    )
    new_items = (history.popularity <= 0).astype(np.float64)
    beyond: dict[str, object] = {
        "coverage_at_50": round(coverage(recommendations, n_items), 6),
        "novelty_at_50": round(
            novelty(history.popularity, recommendations, int(history.popularity.sum())), 6
        ),
        "new_item_share_at_50": round(new_item_share(recommendations, new_items), 6),
        "popularity_bias_at_50": round(popularity_bias(history.popularity, recommendations), 6),
        "catalogue_gini": round(gini(history.popularity), 6),
        "recommended_gini": round(gini(history.popularity[recommended_codes]), 6),
    }
    if config.profile_embeddings is not None:
        beyond["intra_list_diversity_at_50"] = round(
            intra_list_diversity(recommendations, config.profile_embeddings), 6
        )
        beyond["intra_list_diversity_basis"] = "interaction_profile"
    beyond["mean_user_depth"] = round(float(depths.mean()), 4)

    elapsed = time.perf_counter() - started
    _logger.info(
        "eval.source_done",
        extra={
            "source": name,
            "cohort": cohort_name,
            "users": int(cohort_users.size),
            "served_fraction": round(float(served.mean()), 4),
            "hr20": round(values.mean("hr")["hr@20"], 6),
            "seconds": round(elapsed, 1),
        },
    )
    return SourceResult(
        source=name,
        cohort=cohort_name,
        ranking=values,
        segments=segments,
        beyond_list=beyond,
        recommendations=recommendations,
        per_user_ranks=ranks,
        per_user_values=dict(values.values),
        elapsed_seconds=elapsed,
    )


def compare_sources(
    alternative: SourceResult,
    baseline: SourceResult,
    *,
    resamples: int = 1_000,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, object]]:
    """Paired bootstrap between two sources evaluated on the same cohort.

    The K values come from the results themselves. Taking them as an argument with a default would
    let a caller compare at K values the metrics were never computed at.
    """
    ks = alternative.ranking.ks
    if baseline.ranking.ks != ks:
        msg = (
            f"cohort mismatch: {alternative.source} used {alternative.ranking.ks} "
            f"and {baseline.source} used {baseline.ranking.ks}"
        )
        raise ValueError(msg)
    comparisons: dict[str, dict[str, object]] = {}
    for metric in ("hr", "ndcg", "recall", "mrr"):
        results: dict[str, PairedResult] = paired_metric_comparison(
            alternative.per_user_values[metric],
            baseline.per_user_values[metric],
            metric=metric,
            ks=ks,
            resamples=resamples,
            seed=seed,
        )
        comparisons[metric] = {name: result.as_dict() for name, result in results.items()}
    comparisons["served_fraction"] = {
        "alternative": round(alternative.ranking.served_fraction, 6),
        "baseline": round(baseline.ranking.served_fraction, 6),
    }
    return comparisons


def summarise_segments(result: SourceResult) -> dict[str, object]:
    """Segment table plus the fraction of users each source was able to serve."""
    return {
        "cohort": result.cohort,
        "source": result.source,
        "served_fraction": round(result.ranking.served_fraction, 6),
        "segments": result.segments,
    }
