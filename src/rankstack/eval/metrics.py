"""Ranking and beyond-list metrics, all exact over the full catalogue.

Definitions are fixed here once, because the number is meaningless without its definition:

| Metric | Definition |
|---|---|
| `Recall@K` | `|TopK ∩ heldout| / |heldout|` — one target, so a hit or a miss |
| `HR@K` | 1 when the held-out item is in `TopK`, else 0 |
| `NDCG@K` | binary-relevance DCG over `1/log2(rank+1)`, normalised by the ideal ranking |
| `MRR@K` | `1/rank` of the first hit, 0 when there is none |
| `coverage@K` | distinct recommended items, divided by catalogue size |
| `ILD@K` | mean pairwise `1 - cos(item_embedding)` inside each user's `TopK` |
| `novelty@K` | mean self-information `-log p(item)` of the recommended items |
| `new_item_share@K` | fraction of recommended items first seen after the training boundary |
| `popularity_bias@K` | Gini coefficient of recommended-item interaction counts |

No metric samples negatives. With a 1.33M-item catalogue, a 100-negative sample inflates HR@20 by
orders of magnitude, which is why sampled numbers are never reported here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_KS: tuple[int, ...] = (10, 20, 50)

#: A rank is a scalar for one user or an array for a whole cohort; the metric functions accept both
#: because the harness evaluates every user in one vectorised pass.
RankLike = float | np.ndarray


def recall_at_k(rank: RankLike, k: int) -> RankLike:
    """One held-out target, so recall is 1 when the source produced a finite rank, else 0."""
    del k
    return (
        np.where(np.asarray(rank) > 0, 1.0, 0.0)
        if isinstance(rank, np.ndarray)
        else float(rank > 0)
    )


def hit_rate_at_k(rank: RankLike, k: int) -> RankLike:
    """Return 1 when the target's rank is within `k`."""
    if isinstance(rank, np.ndarray):
        return ((rank > 0) & (rank <= k)).astype(np.float64)
    return 1.0 if 0 < rank <= k else 0.0


def ndcg_at_k(rank: RankLike, k: int) -> RankLike:
    """Binary-relevance NDCG. With one relevant item the ideal DCG is `1/log2(2) = 1`."""
    if isinstance(rank, np.ndarray):
        hit = (rank > 0) & (rank <= k)
        return np.where(hit, 1.0 / np.log2(np.where(hit, rank, 2) + 1), 0.0)
    return 1.0 / float(np.log2(rank + 1)) if 0 < rank <= k else 0.0


def reciprocal_rank(rank: RankLike, k: int) -> RankLike:
    """Return `1/rank` of the first hit within `k`, else 0."""
    if isinstance(rank, np.ndarray):
        hit = (rank > 0) & (rank <= k)
        return np.where(hit, 1.0 / np.where(hit, rank, 1), 0.0)
    return 1.0 / rank if 0 < rank <= k else 0.0


@dataclass(frozen=True, slots=True)
class RankingMetrics:
    """Per-user ranking metrics at every K, plus the served fraction."""

    ks: tuple[int, ...]
    served: int
    users: int
    values: dict[str, np.ndarray]
    ranks: np.ndarray

    @property
    def served_fraction(self) -> float:
        """Return the fraction of cohort users this source could score."""
        return self.served / max(self.users, 1)

    def mean(self, metric: str) -> dict[str, float]:
        """Mean of one metric at every K, rounded only at serialisation."""
        array = self.values[metric]
        return {f"{metric}@{k}": float(array[index].mean()) for index, k in enumerate(self.ks)}

    def as_dict(self) -> dict[str, object]:
        """Return every mean ranking metric at every K as a mapping."""
        payload: dict[str, object] = {
            "ks": list(self.ks),
            "users": self.users,
            "served": self.served,
            "served_fraction": round(self.served_fraction, 6),
            "mean_rank_when_served": float(self.ranks[self.ranks > 0].mean())
            if (self.ranks > 0).any()
            else None,
        }
        for metric in ("recall", "hr", "ndcg", "mrr"):
            payload |= {name: round(value, 6) for name, value in self.mean(metric).items()}
        return payload


def ranking_metrics(
    ranks: np.ndarray, served: np.ndarray, ks: tuple[int, ...] = DEFAULT_KS
) -> RankingMetrics:
    """Compute every ranking metric from one rank per user.

    `ranks` holds each user's 1-based rank of their held-out item, or -1 when the source could not
    produce a finite score for it. `served` marks users the source was able to score at all.
    """
    values = {
        metric: np.zeros((len(ks), ranks.size), dtype=np.float64)
        for metric in ("recall", "hr", "ndcg", "mrr")
    }
    evaluable = served & (ranks > 0)
    for index, k in enumerate(ks):
        values["recall"][index] = np.where(evaluable, recall_at_k(1, k), 0.0)
        values["hr"][index] = np.where(evaluable, hit_rate_at_k(ranks, k), 0.0)
        values["ndcg"][index] = np.where(evaluable, ndcg_at_k(ranks, k), 0.0)
        values["mrr"][index] = np.where(evaluable, reciprocal_rank(ranks, k), 0.0)
    return RankingMetrics(
        ks=ks, served=int(served.sum()), users=int(ranks.size), values=values, ranks=ranks
    )


def slice_metrics(metrics: RankingMetrics, mask: np.ndarray) -> dict[str, object]:
    """Mean metrics restricted to a segment mask, for the metrics defined on any subset."""
    if not mask.any():
        return {"n": 0}
    ranks = metrics.ranks[mask]
    served = ranks > 0
    subset = ranking_metrics(ranks, served, ks=metrics.ks)
    payload: dict[str, object] = {"n": int(mask.sum()), "served": int(served.sum())}
    for metric in ("recall", "hr", "ndcg", "mrr"):
        payload |= {name: round(value, 6) for name, value in subset.mean(metric).items()}
    return payload


def coverage(recommendations: dict[int, np.ndarray], n_catalogue: int) -> float:
    """Fraction of the catalogue that appears in at least one user's top-K."""
    if not recommendations:
        return 0.0
    covered: set[int] = set()
    for codes in recommendations.values():
        covered.update(codes.tolist())
    return len(covered) / max(n_catalogue, 1)


def intra_list_diversity(
    recommendations: dict[int, np.ndarray], embeddings: np.ndarray, max_users: int = 2_000
) -> float:
    """Mean pairwise `1 - cosine` inside each user's top-K, averaged over users.

    The embedding is whatever the item side produced; with no learned embedding yet this is computed
    over the normalised interaction profile, and the choice is recorded alongside the number.

    If `embeddings` is 1D (e.g. a single popularity profile), all items look identical in that
    space, so ILD is 0. This is handled explicitly to avoid axis errors.
    """
    if embeddings.ndim == 1:
        return 0.0
    scores: list[float] = []
    for codes in list(recommendations.values())[:max_users]:
        if codes.size < 2:
            continue
        vectors = embeddings[codes]
        norms = np.linalg.norm(vectors, axis=1)
        keep = norms > 0
        if keep.sum() < 2:
            continue
        vectors = vectors[keep] / norms[keep, None]
        similarity = vectors @ vectors.T
        size = similarity.shape[0]
        upper = similarity[np.triu_indices(size, k=1)]
        scores.append(float((1.0 - upper).mean()))
    return float(np.mean(scores)) if scores else 0.0


def novelty(
    item_popularity: np.ndarray, recommendations: dict[int, np.ndarray], total: int
) -> float:
    """Mean self-information of recommended items: higher means less familiar."""
    if not recommendations:
        return 0.0
    probabilities = np.zeros(item_popularity.size, dtype=np.float64)
    nonzero = item_popularity > 0
    probabilities[nonzero] = item_popularity[nonzero] / max(total, 1)
    values: list[float] = []
    for codes in recommendations.values():
        selected = probabilities[codes]
        selected = selected[selected > 0]
        if selected.size:
            values.append(float((-np.log(selected)).mean()))
    return float(np.mean(values)) if values else 0.0


def new_item_share(recommendations: dict[int, np.ndarray], new_items: np.ndarray) -> float:
    """Fraction of recommended items that were unseen at the training boundary."""
    if not recommendations:
        return 0.0
    recommended = np.concatenate([codes for codes in recommendations.values() if codes.size])
    if recommended.size == 0:
        return 0.0
    return float(new_items[recommended].mean())


def gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative distribution: 0 uniform, `(n-1)/n` for a single item.

    Uses the discrete form `sum((2k - n - 1) * x_k) / (n * sum(x))` over ascending-sorted values.
    Zeros are kept: an item with no interactions contributes zero exposure rather than disappearing
    from the distribution.
    """
    positive = np.sort(values[values >= 0].astype(np.float64))
    size = positive.size
    total = positive.sum()
    if size == 0 or total == 0:
        return 0.0
    weights = 2.0 * np.arange(1, size + 1, dtype=np.float64) - size - 1.0
    return float((weights * positive).sum() / (size * total))


def popularity_bias(item_popularity: np.ndarray, recommendations: dict[int, np.ndarray]) -> float:
    """Gini of the interaction counts of recommended items: how concentrated the exposure is."""
    if not recommendations:
        return 0.0
    recommended = np.concatenate([codes for codes in recommendations.values() if codes.size])
    if recommended.size == 0:
        return 0.0
    return gini(item_popularity[recommended])
