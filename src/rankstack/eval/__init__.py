"""Evaluation: exact full-catalogue metrics, segments, and bootstrap statistics."""

from rankstack.eval.harness import SourceResult, compare_sources, evaluate_source
from rankstack.eval.metrics import DEFAULT_KS, RankingMetrics, ranking_metrics
from rankstack.eval.segments import SegmentMasks, segment_masks
from rankstack.eval.stats import Interval, PairedResult, bootstrap_mean, paired_bootstrap

__all__ = [
    "DEFAULT_KS",
    "Interval",
    "PairedResult",
    "RankingMetrics",
    "SegmentMasks",
    "SourceResult",
    "bootstrap_mean",
    "compare_sources",
    "evaluate_source",
    "paired_bootstrap",
    "ranking_metrics",
    "segment_masks",
]
