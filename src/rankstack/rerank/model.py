"""The re-ranking model and the rules applied after it.

**LambdaMART.** LightGBM's `lambdarank` objective optimises a listwise metric directly, which is
what a re-ranker needs: the candidates for one user compete with each other, not with a global
scale. A binary classifier trained row-by-row would optimise the wrong thing, and a pointwise
regressor on a retrieval score would only learn to reproduce stage 1.

**Validation is temporal, not random.** The ranker is trained on one time window and validated on a
later one, because a randomly split candidate list leaks the future through the popularity features:
an item that only becomes popular after the test window looks attractive to a model that saw it.

**The rules are pure functions.** Seen filter, MMR, category cap and freshness boost all take
`(candidates, user_context, config)` and return an ordering. None of them reads global state, so
each is unit-tested against a hand-computed expectation rather than only end to end.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path  # noqa: TC003 - used at runtime by the writers

import lightgbm as lgb
import numpy as np

from rankstack.logging import get_logger

_logger = get_logger(__name__)

DEFAULT_LAMBDA_POSITION = 0.7
DEFAULT_MMR_LAMBDA = 0.7
DEFAULT_CATEGORY_CAP = 3
DEFAULT_FRESHNESS_WEIGHT = 0.05
DEFAULT_FRESHNESS_HALFLIFE_DAYS = 30.0


@dataclass(frozen=True, slots=True)
class RankerConfig:
    """LambdaMART hyperparameters, written into the result file verbatim."""

    objective: str = "lambdarank"
    metric: str = "ndcg"
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 400
    min_child_samples: int = 50
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq = 1
    lambdarank_truncation_level: int = 30
    seed: int = 20261008

    def as_dict(self) -> dict[str, object]:
        """Return the hyperparameters as a JSON-serialisable mapping."""
        return {
            "objective": self.objective,
            "metric": self.metric,
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "min_child_samples": self.min_child_samples,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "lambdarank_truncation_level": self.lambdarank_truncation_level,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class RuleConfig:
    """Post-ranking rule parameters. Defaults come from the design document."""

    mmr_lambda: float = DEFAULT_MMR_LAMBDA
    category_cap: int = DEFAULT_CATEGORY_CAP
    freshness_weight: float = DEFAULT_FRESHNESS_WEIGHT
    freshness_halflife_days: float = DEFAULT_FRESHNESS_HALFLIFE_DAYS

    def as_dict(self) -> dict[str, object]:
        """Return the rule parameters as a JSON-serialisable mapping."""
        return {
            "mmr_lambda": self.mmr_lambda,
            "category_cap": self.category_cap,
            "freshness_weight": self.freshness_weight,
            "freshness_halflife_days": self.freshness_halflife_days,
        }


@dataclass(frozen=True, slots=True)
class Ranker:
    """A fitted LambdaMART booster plus the feature order it was trained against."""

    booster: lgb.Booster
    feature_names: tuple[str, ...]
    importance: dict[str, float] = field(default_factory=dict)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Score rows of features; higher is better."""
        return np.asarray(self.booster.predict(features, num_threads=4), dtype=np.float64)

    def save(self, path: Path) -> Path:
        """Persist the booster and the feature order it expects."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path.with_suffix(".txt")))
        path.with_suffix(".json").write_text(
            json.dumps(
                {"features": list(self.feature_names), "importance": self.importance}, indent=2
            )
            + "\n"
        )
        return path

    @classmethod
    def load(cls, path: Path) -> Ranker:
        """Load a fitted ranker from its artefact path."""
        metadata = json.loads(path.with_suffix(".json").read_text())
        booster = lgb.Booster(model_file=str(path.with_suffix(".txt")))
        return cls(
            booster=booster,
            feature_names=tuple(metadata["features"]),
            importance=metadata.get("importance", {}),
        )


def train_ranker(
    features: np.ndarray,
    labels: np.ndarray,
    group_sizes: np.ndarray,
    feature_names: tuple[str, ...],
    config: RankerConfig | None = None,
    *,
    valid_features: np.ndarray | None = None,
    valid_labels: np.ndarray | None = None,
    valid_groups: np.ndarray | None = None,
) -> Ranker:
    """Fit LambdaMART on grouped candidate lists.

    `group_sizes` gives the number of rows per user, in row order. LightGBM uses it to scope each
    listwise gradient to one user's candidates; getting it wrong silently trains a pointwise model.
    """
    settings = config or RankerConfig()
    train_set = lgb.Dataset(
        features,
        label=labels,
        group=group_sizes,
        feature_name=list(feature_names),
        free_raw_data=False,
    )
    valid_sets = None
    if valid_features is not None and valid_labels is not None and valid_groups is not None:
        valid_sets = [
            lgb.Dataset(
                valid_features,
                label=valid_labels,
                group=valid_groups,
                feature_name=list(feature_names),
                reference=train_set,
                free_raw_data=False,
            )
        ]

    booster = lgb.train(
        settings.as_dict(),
        train_set,
        num_boost_round=settings.n_estimators,
        valid_sets=valid_sets,
        callbacks=[lgb.log_evaluation(period=0)] if valid_sets is not None else None,
    )
    gains = booster.feature_importance(importance_type="gain")
    total = float(gains.sum()) or 1.0
    importance = {
        name: round(float(value) / total, 6)
        for name, value in zip(feature_names, gains, strict=True)
    }
    ordered = sorted(importance.items(), key=lambda pair: -pair[1])[:8]
    _logger.info("rerank.ranker_trained", extra={"top_features": ordered})
    return Ranker(booster=booster, feature_names=tuple(feature_names), importance=importance)


def apply_mmr(
    items: np.ndarray,
    scores: np.ndarray,
    similarity: np.ndarray,
    k: int,
    *,
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
) -> np.ndarray:
    """Maximal Marginal Relevance re-ranking of `items`, returning the selected indices.

    `lambda * relevance - (1 - lambda) * max similarity to what is already selected`. A lambda of 1
    reduces to relevance ordering, so the rule is a no-op when no diversification is wanted.
    """
    if items.size == 0 or k <= 0:
        return np.empty(0, dtype=np.int64)
    selected: list[int] = []
    remaining = list(range(items.size))
    best_similarity = np.zeros(items.size, dtype=np.float32)
    while remaining and len(selected) < k:
        candidates = np.asarray(remaining)
        utility = mmr_lambda * scores[candidates] - (1.0 - mmr_lambda) * best_similarity[candidates]
        chosen = int(candidates[int(np.argmax(utility))])
        selected.append(chosen)
        remaining.remove(chosen)
        if remaining:
            best_similarity[remaining] = np.maximum(
                best_similarity[remaining], similarity[chosen, remaining]
            )
    return np.asarray(selected, dtype=np.int64)


def cap_categories(
    items: np.ndarray, categories: np.ndarray, k: int, *, cap: int = DEFAULT_CATEGORY_CAP
) -> np.ndarray:
    """Keep at most `cap` items per category, promoting the next best item instead.

    Items with an unknown category (code -1) are not counted against the cap: the cap exists to stop
    ten near-identical cables, and an item whose category is unknown is not evidence of a category.
    """
    if cap <= 0 or items.size == 0:
        return np.arange(min(k, items.size), dtype=np.int64)
    seen: dict[int, int] = {}
    kept: list[int] = []
    deferred: list[int] = []
    for position in range(items.size):
        category = int(categories[position])
        if category < 0:
            kept.append(position)
            continue
        if seen.get(category, 0) < cap:
            seen[category] = seen.get(category, 0) + 1
            kept.append(position)
        else:
            deferred.append(position)
        if len(kept) >= k:
            break
    for position in deferred:
        if len(kept) >= k:
            break
        kept.append(position)
    return np.asarray(kept[:k], dtype=np.int64)


def freshness_bonus(
    days_since_first: np.ndarray,
    *,
    weight: float = DEFAULT_FRESHNESS_WEIGHT,
    halflife_days: float = DEFAULT_FRESHNESS_HALFLIFE_DAYS,
) -> np.ndarray:
    """Additive score bonus decaying with an item's age since its first train interaction.

    An item never seen before has an undefined age; it gets the maximum bonus rather than being
    penalised, because newness is the property being rewarded.
    """
    finite = np.isfinite(days_since_first)
    decay = np.zeros(days_since_first.size, dtype=np.float32)
    decay[finite] = np.exp(-np.log(2.0) * days_since_first[finite] / max(halflife_days, 1e-6))
    decay[~finite] = 1.0
    return (weight * decay).astype(np.float32)
