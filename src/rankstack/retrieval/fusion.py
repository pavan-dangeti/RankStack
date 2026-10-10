"""Rank fusion over several retrievers.

Serving the measured-best retriever alone turned out to serve the *same* list to every user:
`popularity_recent` is a global ranking, so personalisation requires a personalised source in the
mix even though that source scores worse on its own (co-occurrence HR@20 0.00295 against
popularity_recent's 0.02395).

Reciprocal Rank Fusion combines them without tuning weights: an item scores
`sum over sources of 1 / (rrf_k + its rank in that source)`, and only the order matters, so the
sources need no score calibration. `rrf_k = 60` is the value from the original Cormack et al. paper,
not a number fitted to this corpus.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rankstack.logging import get_logger
from rankstack.retrieval.base import NEG_INF, exact_rank, top_k

_logger = get_logger(__name__)

DEFAULT_RRF_K = 60
DEFAULT_DEPTH = 500


@dataclass(frozen=True, slots=True)
class FusedSource:
    """Reciprocal-rank fusion of several `CandidateSource`s.

    Fusion is done on ranks, computed per request, so cost is `n_sources * depth` rather than a full
    catalogue pass per source. A source with no history for a user contributes nothing for them,
    which is the correct behaviour rather than an empty ranking.
    """

    sources: dict[str, object]
    rrf_k: int = DEFAULT_RRF_K
    depth: int = DEFAULT_DEPTH

    @property
    def name(self) -> str:
        """Name reported in results files."""
        return "fused"

    def n_items(self) -> int:
        """Return the size of the catalogue."""
        return int(next(iter(self.sources.values())).n_items())  # type: ignore[attr-defined]

    def can_serve(self, user_code: int) -> bool:
        """Return True: fusion ranks from whichever source responds."""
        del user_code
        return True

    def score(self, user_code: int, seen: np.ndarray) -> np.ndarray:
        """Return fused scores for every item, with already-seen items masked to `-inf`."""
        fused = np.zeros(self.n_items(), dtype=np.float64)
        contributed = 0
        for source in self.sources.values():
            scores = source.score(user_code, seen)  # type: ignore[attr-defined]
            codes, _ = top_k(scores, self.depth)
            if codes.size == 0:
                continue
            contributed += 1
            fused[codes] += 1.0 / (self.rrf_k + np.arange(1, codes.size + 1, dtype=np.float64))
        if contributed == 0:
            # No source produced a ranking, which only happens if every source raised and the
            # exception boundary above already logged it. Returning -inf keeps the item out rather
            # than inventing a score.
            fused[:] = NEG_INF
        fused[seen] = NEG_INF
        return fused.astype(np.float32)

    def rank_of(self, user_code: int, item_code: int) -> int:
        """Return the exact fused rank of one item, for tests and diagnosis."""
        return exact_rank(self.score(user_code, np.zeros(self.n_items(), dtype=bool)), item_code)
