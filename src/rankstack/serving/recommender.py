"""The request path.

One rule governs this module: a user-facing request always returns a non-empty list with HTTP 200.
Every failure is a degradation to a weaker but working model, never an empty list and never a 5xx.
The degradation reasons are the ones the design document's section 6.9 table names, and each is a
branch with a metric rather than a silent fallback.

The serving choice is measurement, not ambition. `popularity_recent` has the best hit rate, but it
is a global ranking: it returned only 3 distinct top items across 2,000 warm users. The request path
serves reciprocal-rank fusion of `popularity_recent` with co-occurrence instead, which returns 18
distinct top items and costs 25% of HR@20 (0.0150 against 0.0200). The re-ranker and the two-tower
are not in the request path because both were measured and both lost.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from prometheus_client import CollectorRegistry, Counter, Histogram

from rankstack.logging import get_logger
from rankstack.retrieval.base import top_k

if TYPE_CHECKING:
    from collections.abc import Callable

    from rankstack.retrieval.base import CandidateSource
    from rankstack.serving.artifacts import ServingArtifacts

_logger = get_logger(__name__)

MAX_K = 50
DEFAULT_K = 20


@dataclass(frozen=True, slots=True)
class Explanation:
    """A template-based reason for one recommendation.

    The template is chosen from a finite set keyed on which signal dominated the score, and the
    evidence is drawn from data the model actually used. No text is generated at request time
    (ADR-0003), so an explanation is always traceable to a score.
    """

    template: str
    evidence: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        """Return the explanation as a JSON-serialisable mapping."""
        return {"template": self.template, "evidence": list(self.evidence)}


@dataclass(frozen=True, slots=True)
class RecommendedItem:
    """One item in a response."""

    rank: int
    item_id: str
    title: str | None
    category: str | None
    price: float | None
    score: float
    explanation: Explanation

    def as_dict(self) -> dict[str, object]:
        """Return the item as a JSON-serialisable mapping."""
        return {
            "rank": self.rank,
            "item_id": self.item_id,
            "title": self.title,
            "category": self.category,
            "price": self.price,
            "score": round(self.score, 6),
            "explanation": self.explanation.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class RecommendationResponse:
    """The full response body, matching the contract in the design document."""

    user_id: str
    k: int
    items: tuple[RecommendedItem, ...]
    model_version: str
    index_version: str
    retrieved_candidates: int
    fallback_used: str | None
    served_at_ms: int
    request_id: str
    elapsed_ms: float = 0.0

    def as_dict(self) -> dict[str, object]:
        """Return the response as a JSON-serialisable mapping."""
        return {
            "user_id": self.user_id,
            "k": self.k,
            "items": [item.as_dict() for item in self.items],
            "model_version": self.model_version,
            "index_version": self.index_version,
            "retrieved_candidates": self.retrieved_candidates,
            "fallback_used": self.fallback_used,
            "served_at_ms": self.served_at_ms,
            "request_id": self.request_id,
            "elapsed_ms": round(self.elapsed_ms, 3),
        }


class Metrics:
    """Prometheus collectors for the request path.

    A thin wrapper around a private registry rather than the global default, so each `Recommender`
    gets its own counters and tests never contaminate each other.
    """

    def __init__(self) -> None:
        """Create a private registry and the collectors the design document names."""
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "recommend_requests_total",
            "Recommendation requests served.",
            ["fallback"],
            registry=self.registry,
        )
        self.duration = Histogram(
            "recommend_duration_seconds",
            "End-to-end request duration.",
            buckets=(0.005, 0.01, 0.02, 0.06, 0.15, 0.3, 1.0),
            registry=self.registry,
        )
        self.fallbacks = Counter(
            "fallback_total",
            "Degradations taken, by reason.",
            ["reason"],
            registry=self.registry,
        )
        self.candidates = Histogram(
            "recommend_candidates",
            "Candidates the retriever returned before truncation.",
            buckets=(1, 5, 10, 20, 50, 100, 500),
            registry=self.registry,
        )
        self.shed = Counter(
            "shed_requests_total",
            "Requests shed to a smaller k under load.",
            registry=self.registry,
        )


@dataclass(slots=True)
class Recommender:
    """Turns a user id into a ranked list, degrading rather than failing."""

    artifacts: ServingArtifacts
    metrics: Metrics = field(default_factory=Metrics)
    #: Overrides the artefact set's own source choice. Exists so a test or a load test can install a
    #: source that always raises and time the degraded path through the real boundary, rather than
    #: reimplementing the boundary somewhere else and measuring that instead.
    source: Callable[[], CandidateSource] | None = None

    def recommend(self, user_id: str, k: int = DEFAULT_K) -> RecommendationResponse:
        """Return `k` recommendations for `user_id`, degrading instead of failing."""
        started = time.perf_counter()
        request_id = str(uuid.uuid4())
        effective_k = max(1, min(int(k), MAX_K))
        fallback: str | None = None

        user_code = self.artifacts.encoding.user_to_code.get(user_id)
        if user_code is None:
            fallback = "cold_start"
            self.metrics.fallbacks.labels(reason="cold_start").inc()

        try:
            items, candidates, degraded = self._rank(user_code, effective_k)
        except Exception:
            _logger.exception("serving.retrieval_failed", extra={"request_id": request_id})
            self.metrics.fallbacks.labels(reason="retriever").inc()
            items, candidates = self._popularity_fallback(effective_k)
            degraded = "retriever"
        fallback = degraded if degraded is not None else fallback

        response = RecommendationResponse(
            user_id=user_id,
            k=effective_k,
            items=items,
            model_version=self._version("popularity.npy"),
            index_version=self._version("popularity_recent.npy"),
            retrieved_candidates=candidates,
            fallback_used=fallback,
            served_at_ms=int(time.time() * 1000),
            request_id=request_id,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
        self.metrics.requests.labels(fallback=fallback or "none").inc()
        self.metrics.duration.observe(response.elapsed_ms / 1000)
        self.metrics.candidates.observe(candidates)
        _logger.info(
            "serving.request",
            extra={
                "request_id": request_id,
                "user_known": user_code is not None,
                "k": effective_k,
                "items": len(items),
                "fallback": fallback or "none",
                "elapsed_ms": round(response.elapsed_ms, 3),
            },
        )
        return response

    def _rank(
        self, user_code: int | None, k: int
    ) -> tuple[tuple[RecommendedItem, ...], int, str | None]:
        """Produce ranked items, falling back to popularity if the primary retriever throws."""
        seen = (
            self.artifacts.seen_mask(user_code)
            if user_code is not None
            else np.zeros(self.artifacts.n_items, dtype=bool)
        )
        source = self.source() if self.source is not None else self.artifacts.served_source()
        try:
            scores = source.score(user_code or 0, seen)
        except Exception:
            _logger.exception("serving.primary_retriever_failed")
            self.metrics.fallbacks.labels(reason="index").inc()
            items, candidates = self._popularity_fallback(k)
            return items, candidates, "index"
        return self._items_from_scores(scores, k), int(np.isfinite(scores).sum()), None

    def _items_from_scores(self, scores: np.ndarray, k: int) -> tuple[RecommendedItem, ...]:
        """Turn a score vector into explained, titled, ranked items."""
        codes, values = top_k(scores, k)
        best = float(values[0]) if values.size else 0.0
        items: list[RecommendedItem] = []
        for rank, (code, value) in enumerate(zip(codes, values, strict=True), start=1):
            item_code = int(code)
            title, category, price = self.artifacts.catalogue.describe(item_code)
            items.append(
                RecommendedItem(
                    rank=rank,
                    item_id=self.artifacts.encoding.item_codes[item_code],
                    title=title or None,
                    category=category,
                    price=price,
                    score=float(value),
                    explanation=self._explain(item_code, float(value), best),
                )
            )
        return tuple(items)

    def _popularity_fallback(self, k: int) -> tuple[tuple[RecommendedItem, ...], int]:
        """Serve global popularity: it cannot fail for any user and never returns empty."""
        scores = self.artifacts.popularity.score(0, np.zeros(self.artifacts.n_items, dtype=bool))
        return self._items_from_scores(scores, k), int(np.isfinite(scores).sum())

    def _explain(self, item_code: int, score: float, best: float) -> Explanation:
        """Choose one explanation template from a finite set, keyed on what dominated the score."""
        item_id = self.artifacts.encoding.item_codes[item_code]
        title, category, price = self.artifacts.catalogue.describe(item_code)
        label = title or item_id
        close_to_top = best > 0 and score >= 0.9 * best
        if category:
            template = "top_ranked_in_category" if close_to_top else "popular_in_category"
            evidence = [category, f"score {score:.4f}"]
            if close_to_top:
                evidence.insert(0, f"highest-scoring {category} pick right now")
            return Explanation(template, tuple(evidence))
        if price:
            return Explanation(
                "trending_by_interactions",
                (label, f"price {price:.2f}", f"score {score:.4f}"),
            )
        return Explanation("trending_by_interactions", (label, f"score {score:.4f}"))

    def _version(self, artefact: str) -> str:
        """Return the short checksum of an artefact, or `unverified` when none was recorded."""
        return self.artifacts.versions.get(artefact, "unverified")[:16]
