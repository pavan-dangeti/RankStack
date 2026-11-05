"""Tests for the weighted fusion and the pre-registered rule that picks its weight (ADR-0007)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import duckdb
import numpy as np
from scripts.select_fusion_weight import Candidate, select_weight

from rankstack.config import settings
from rankstack.data.encode import build_cohort
from rankstack.retrieval.fusion import FusedSource
from rankstack.retrieval.popularity import PopularityModel
from rankstack.serving.artifacts import ServingArtifacts

if TYPE_CHECKING:
    from tests.conftest import Trained


class _Fixed:
    """A source returning one fixed score vector for every user."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = np.asarray(scores, dtype=np.float32)

    def n_items(self) -> int:
        return int(self._scores.size)

    def score(self, user_code: int, seen: np.ndarray) -> np.ndarray:
        del user_code
        return np.where(seen, -np.inf, self._scores).astype(np.float32)


def _fused(weight: float) -> np.ndarray:
    sources: dict[str, object] = {
        "recent": _Fixed([4, 3, 2, 1]),
        "cooccurrence": _Fixed([0, 0, 1, 2]),
    }
    fused = FusedSource(sources=sources, rrf_k=60, depth=4, weights={"cooccurrence": weight})
    return fused.score(0, np.zeros(4, dtype=bool))


def test_weight_scales_only_the_weighted_sources_rank_credit() -> None:
    """Co-occurrence ranks item 3 first and item 2 second; at w = 0.5 each credit is halved."""
    expected = np.array([1 / 61, 1 / 62, 1 / 63 + 0.5 / 62, 1 / 64 + 0.5 / 61], dtype=np.float32)
    assert np.allclose(_fused(0.5), expected)


def test_weight_zero_is_the_other_source_alone() -> None:
    """Weight 0 must not reorder anything: the fused order is the popularity order."""
    scores = _fused(0.0)
    assert np.allclose(scores, [1 / 61, 1 / 62, 1 / 63, 1 / 64])
    assert list(np.argsort(-scores)) == [0, 1, 2, 3]


def test_rule_picks_the_highest_coverage_among_qualifying_weights() -> None:
    """A non-qualifying weight is ignored however much it covers."""
    candidates = [
        Candidate(weight=0.0, qualifies=True, coverage_items=40),
        Candidate(weight=0.1, qualifies=True, coverage_items=300),
        Candidate(weight=0.25, qualifies=True, coverage_items=900),
        Candidate(weight=0.5, qualifies=False, coverage_items=5_000),
    ]
    assert select_weight(candidates) == 0.25


def test_rule_breaks_a_coverage_tie_towards_the_smaller_weight() -> None:
    """Equal coverage: prefer the weight closer to popularity alone."""
    candidates = [
        Candidate(weight=0.75, qualifies=True, coverage_items=900),
        Candidate(weight=0.25, qualifies=True, coverage_items=900),
    ]
    assert select_weight(candidates) == 0.25


def test_rule_falls_back_to_popularity_alone() -> None:
    """When only w = 0 qualifies, or nothing does, the rule serves popularity_recent alone."""
    only_zero = [
        Candidate(weight=0.0, qualifies=True, coverage_items=40),
        Candidate(weight=1.0, qualifies=False, coverage_items=5_000),
    ]
    assert select_weight(only_zero) == 0.0
    assert select_weight([Candidate(weight=1.0, qualifies=False, coverage_items=1)]) == 0.0


def _artifacts(trained: Trained, **overrides: Any) -> ServingArtifacts:
    return ServingArtifacts(
        encoding=trained.encoding,
        popularity=trained.popularity,
        popularity_recent=trained.popularity,
        cooccurrence=trained.cooccurrence,
        catalogue=None,  # type: ignore[arg-type]  # served_source never reads the catalogue
        history_indptr=trained.cooccurrence.history_indptr,
        history_codes=trained.cooccurrence.history_codes,
        corpus="synthetic",
        versions={},
        **overrides,
    )


def test_serving_weight_zero_serves_popularity_alone(trained: Trained) -> None:
    """Weight 0 is not a degenerate fusion: it is the global ranking itself."""
    source = _artifacts(trained, cooccurrence_weight=0.0).served_source()
    assert isinstance(source, PopularityModel)


def test_serving_weight_reaches_the_fused_source(trained: Trained) -> None:
    """The configured weight is the one the request path fuses with."""
    source = _artifacts(trained, cooccurrence_weight=0.25).served_source()
    assert isinstance(source, FusedSource)
    assert source.weights == {"cooccurrence": 0.25}


def test_serving_weight_defaults_to_the_config_value(trained: Trained) -> None:
    """An artefact set built without an explicit weight serves the configured one."""
    assert _artifacts(trained).cooccurrence_weight == settings.fusion_cooccurrence_weight


def test_validation_cohort_targets_come_from_the_validation_window(trained: Trained) -> None:
    """Each target is the user's last validation-window item, and every user is warm."""
    connection = duckdb.connect(database=":memory:")
    cohort = build_cohort(
        connection, "warm_users_valid", trained.encoding, trained.processed, size=500, seed=7
    )
    code_to_user = {code: user for user, code in trained.encoding.user_to_code.items()}
    code_to_item = {code: item for item, code in trained.encoding.item_to_code.items()}
    valid = (trained.processed / "interactions" / "valid.parquet").as_posix()
    train = (trained.processed / "interactions" / "train.parquet").as_posix()
    last = dict(
        connection.execute(
            "SELECT user_id, arg_max(parent_asin, timestamp) FROM read_parquet(?) GROUP BY user_id",
            [valid],
        ).fetchall()
    )
    depth = dict(
        connection.execute(
            "SELECT user_id, count(*) FROM read_parquet(?) GROUP BY user_id", [train]
        ).fetchall()
    )
    connection.close()

    assert len(cohort) > 0
    for user_code, target_code in zip(cohort.user_codes, cohort.target_codes, strict=True):
        user = code_to_user[user_code]
        assert last[user] == code_to_item[target_code]
        assert depth[user] >= 2
