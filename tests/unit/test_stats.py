"""Bootstrap statistics and paired comparisons."""

from __future__ import annotations

import numpy as np
import pytest

from rankstack.eval.stats import (
    Interval,
    PairedResult,
    bootstrap_mean,
    paired_bootstrap,
    paired_metric_comparison,
)


def test_bootstrap_interval_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(loc=0.2, scale=1.0, size=2_000)
    interval = bootstrap_mean(values, resamples=500, seed=11)
    assert interval.low < interval.point < interval.high
    assert interval.resamples == 500


def test_bootstrap_of_a_constant_has_a_degenerate_interval() -> None:
    interval = bootstrap_mean(np.full(100, 0.5))
    assert interval.low == interval.high == 0.5


def test_bootstrap_of_an_empty_sample_is_zero_not_an_error() -> None:
    assert bootstrap_mean(np.empty(0)).as_dict()["point"] == 0.0


def test_a_real_difference_is_measurable_and_a_null_difference_is_not() -> None:
    rng = np.random.default_rng(3)
    baseline = rng.binomial(1, 0.10, size=4_000).astype(float)
    better = baseline + (rng.random(4_000) < 0.05)
    real = paired_bootstrap(better, baseline, metric="hr@20", resamples=500)
    noise = paired_bootstrap(
        rng.binomial(1, 0.10, size=4_000).astype(float),
        rng.binomial(1, 0.10, size=4_000).astype(float),
        metric="hr@20",
        resamples=500,
    )
    assert real.measurable
    assert real.bootstrap_probability > 0.99
    assert not noise.measurable


def test_paired_bootstrap_refuses_unaligned_arrays() -> None:
    with pytest.raises(ValueError, match="aligned"):
        paired_bootstrap(np.zeros(5), np.zeros(6), metric="hr@20")


def test_paired_comparison_returns_one_result_per_k() -> None:
    rng = np.random.default_rng(5)
    alternative = rng.random((3, 500))
    baseline = rng.random((3, 500))
    results = paired_metric_comparison(
        alternative, baseline, metric="hr", ks=(10, 20, 50), resamples=100
    )
    assert set(results) == {"hr@10", "hr@20", "hr@50"}
    assert all(isinstance(result, PairedResult) for result in results.values())


def test_interval_reports_whether_it_excludes_zero() -> None:
    assert Interval(point=0.2, low=0.1, high=0.3, resamples=10, confidence=0.95).excludes_zero
    assert not Interval(point=0.0, low=-0.1, high=0.1, resamples=10, confidence=0.95).excludes_zero
