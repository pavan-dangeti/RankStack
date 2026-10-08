"""Bootstrap confidence intervals and paired comparisons.

Averages over a 20,000-user cohort still have a standard error worth reporting, and the difference
between two models on the *same* users is estimated far more precisely than either model alone. Both
are handled with a bootstrap over users.

`PairedResult.bootstrap_probability` is the proportion of resamples where the alternative beat the
baseline: a bootstrap probability, not a frequentist p-value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import numpy as np

DEFAULT_RESAMPLES = 1_000
DEFAULT_SEED = 20261008

T = TypeVar("T", bound=np.ndarray)


def _percentile_interval(samples: np.ndarray, confidence: float) -> tuple[float, float]:
    tail = (1.0 - confidence) / 2.0
    low, high = np.percentile(samples, [tail * 100, (1 - tail) * 100])
    return float(low), float(high)


@dataclass(frozen=True, slots=True)
class Interval:
    """A point estimate with its bootstrap interval."""

    point: float
    low: float
    high: float
    resamples: int
    confidence: float

    @property
    def excludes_zero(self) -> bool:
        """Return True when the interval excludes zero."""
        return self.low > 0.0 or self.high < 0.0

    def as_dict(self) -> dict[str, object]:
        """Return the interval as a JSON-serialisable mapping."""
        return {
            "point": round(self.point, 6),
            "ci_low": round(self.low, 6),
            "ci_high": round(self.high, 6),
            "resamples": self.resamples,
            "confidence": self.confidence,
        }


def bootstrap_mean(
    values: np.ndarray,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> Interval:
    """Bootstrap the mean of `values` by resampling its entries with replacement."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return Interval(point=0.0, low=0.0, high=0.0, resamples=0, confidence=confidence)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, array.size, size=(resamples, array.size))
    means = array[draws].mean(axis=1)
    low, high = _percentile_interval(means, confidence)
    return Interval(
        point=float(array.mean()), low=low, high=high, resamples=resamples, confidence=confidence
    )


@dataclass(frozen=True, slots=True)
class PairedResult:
    """Difference between two models on the same users.

    Report `bootstrap_probability` as a bootstrap probability, never as a p-value.
    """

    metric: str
    point_difference: float
    low: float
    high: float
    bootstrap_probability: float
    resamples: int
    n: int

    @property
    def measurable(self) -> bool:
        """True when the interval excludes zero, i.e. the difference is real and not noise."""
        return self.low > 0.0 or self.high < 0.0

    def as_dict(self) -> dict[str, object]:
        """Return the comparison as a JSON-serialisable mapping."""
        return {
            "metric": self.metric,
            "difference": round(self.point_difference, 6),
            "ci_low": round(self.low, 6),
            "ci_high": round(self.high, 6),
            "bootstrap_probability_alternative_wins": round(self.bootstrap_probability, 4),
            "measurable": self.measurable,
            "resamples": self.resamples,
            "n": self.n,
        }


def paired_bootstrap(
    alternative: np.ndarray,
    baseline: np.ndarray,
    *,
    metric: str,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> PairedResult:
    """Bootstrap the per-user difference between two models on the same cohort.

    Both arrays must be aligned user by user, which is what makes the comparison paired and why the
    cohort is sampled once and reused for every source.
    """
    left = np.asarray(alternative, dtype=np.float64)
    right = np.asarray(baseline, dtype=np.float64)
    if left.shape != right.shape:
        msg = f"paired bootstrap needs aligned arrays, got {left.shape} and {right.shape}"
        raise ValueError(msg)
    difference = left - right
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, difference.size, size=(resamples, difference.size))
    resampled = difference[draws].mean(axis=1)
    low, high = _percentile_interval(resampled, confidence)
    return PairedResult(
        metric=metric,
        point_difference=float(difference.mean()),
        low=low,
        high=high,
        bootstrap_probability=float((resampled > 0).mean()),
        resamples=resamples,
        n=int(difference.size),
    )


def paired_metric_comparison(
    alternative: np.ndarray,
    baseline: np.ndarray,
    *,
    metric: str,
    ks: tuple[int, ...],
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, PairedResult]:
    """Run `paired_bootstrap` for a per-user metric matrix shaped `(n_ks, n_users)`."""
    left = np.atleast_2d(np.asarray(alternative, dtype=np.float64))
    right = np.atleast_2d(np.asarray(baseline, dtype=np.float64))
    results: dict[str, PairedResult] = {}
    for index, k in enumerate(ks):
        results[f"{metric}@{k}"] = paired_bootstrap(
            left[index],
            right[index],
            metric=f"{metric}@{k}",
            resamples=resamples,
            seed=seed,
        )
    return results
