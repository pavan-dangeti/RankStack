"""Serving-path tests, including one failure injection per degradation row.

The request path's whole claim is that it degrades rather than fails. Each test below breaks one
thing the design document's section 6.9 table names and asserts both halves of that claim: the
request still returns a non-empty list, and the reason is reported rather than swallowed.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import duckdb
import numpy as np
import pytest
from fastapi.testclient import TestClient
from prometheus_client import generate_latest

from rankstack.logging import get_logger
from rankstack.retrieval.fusion import FusedSource
from rankstack.retrieval.popularity import PopularityModel
from rankstack.serving.app import create_app
from rankstack.serving.artifacts import (
    ItemCatalogue,
    ServingArtifacts,
    build_manifest,
    load_artifacts,
    load_catalogue,
    sha256_file,
    verify,
)
from rankstack.serving.recommender import DEFAULT_K, MAX_K, Recommender

if TYPE_CHECKING:
    from pathlib import Path

    from rankstack.config import Settings
    from tests.conftest import Trained

_logger = get_logger(__name__)


def _exposition(recommender: Recommender) -> str:
    """Render a recommender's own Prometheus registry as text."""
    return generate_latest(recommender.metrics.registry).decode()


def _catalogue(n_items: int, *, with_titles: bool) -> ItemCatalogue:
    """Build a deterministic catalogue: category and price both keyed on the item index."""
    titles = (
        np.array([f"Item {code}" for code in range(n_items)], dtype=object) if with_titles else None
    )
    categories = np.array(
        ["cat_a" if code % 2 == 0 else "cat_b" for code in range(n_items)], dtype=object
    )
    prices = np.full(n_items, 9.99, dtype=np.float32)
    return ItemCatalogue(titles=titles, categories=categories, prices=prices)


@pytest.fixture
def artifacts(trained: Trained) -> ServingArtifacts:
    """Assemble a serving artefact set from the trained synthetic baselines."""
    return ServingArtifacts(
        encoding=trained.encoding,
        popularity=trained.popularity,
        popularity_recent=trained.popularity,
        cooccurrence=trained.cooccurrence,
        catalogue=_catalogue(trained.encoding.n_items, with_titles=True),
        history_indptr=trained.cooccurrence.history_indptr,
        history_codes=trained.cooccurrence.history_codes,
        corpus="synthetic",
        versions={"popularity.npy": "a" * 64, "popularity_recent.npy": "b" * 64},
    )


class _Exploding:
    """A source that always raises, standing in for a corrupt or unreachable index."""

    @property
    def name(self) -> str:
        return "exploding"

    def n_items(self) -> int:
        return 0

    def can_serve(self, user_code: int) -> bool:
        del user_code
        return True

    def score(self, user_code: int, seen: np.ndarray) -> np.ndarray:
        del user_code, seen
        msg = "injected failure"
        raise RuntimeError(msg)


def test_warm_user_gets_k_ranked_items(artifacts: ServingArtifacts) -> None:
    """A known user receives exactly k items, ranked by descending score with unique ids."""
    user = artifacts.encoding.user_codes[0]
    response = Recommender(artifacts=artifacts).recommend(user, k=5)

    assert len(response.items) == 5
    assert [item.rank for item in response.items] == [1, 2, 3, 4, 5]
    scores = [item.score for item in response.items]
    assert scores == sorted(scores, reverse=True)
    assert len({item.item_id for item in response.items}) == 5
    assert response.fallback_used is None


def test_never_recommends_an_item_the_user_already_saw(artifacts: ServingArtifacts) -> None:
    """The `seen` mask reaches the served items: no returned id is in the stored history."""
    recommender = Recommender(artifacts=artifacts)
    for code in range(min(artifacts.encoding.n_users, 10)):
        user = artifacts.encoding.user_codes[code]
        response = recommender.recommend(user, k=3)
        start, end = artifacts.history_indptr[code], artifacts.history_indptr[code + 1]
        seen = {int(c) for c in artifacts.history_codes[start:end]}
        served = {artifacts.encoding.item_to_code[item.item_id] for item in response.items}
        assert not served & seen, "a served item was already interacted with"


def test_unknown_user_degrades_to_cold_start(artifacts: ServingArtifacts) -> None:
    """An unknown user still gets items, and the response says why."""
    response = Recommender(artifacts=artifacts).recommend("no-such-user", k=3)

    assert response.fallback_used == "cold_start"
    assert len(response.items) == 3
    assert all(item.item_id for item in response.items)


def test_broken_index_degrades_to_popularity(artifacts: ServingArtifacts) -> None:
    """A retriever that raises falls back to global popularity rather than erroring."""
    response = Recommender(artifacts=artifacts, source=_Exploding).recommend(
        artifacts.encoding.user_codes[0], k=3
    )

    assert response.fallback_used == "index"
    assert len(response.items) == 3
    assert [item.rank for item in response.items] == [1, 2, 3]


def test_degraded_path_still_explains_every_item(artifacts: ServingArtifacts) -> None:
    """Degradation must not lose the explanation: every item carries a template and evidence."""
    response = Recommender(artifacts=artifacts, source=_Exploding).recommend(
        artifacts.encoding.user_codes[0], k=5
    )

    for item in response.items:
        assert item.explanation.template
        assert item.explanation.evidence
        assert all(str(evidence) for evidence in item.explanation.evidence)


def test_explanation_template_is_from_a_finite_set(artifacts: ServingArtifacts) -> None:
    """No free-form text is generated: templates come from the fixed set the module defines."""
    allowed = {"top_ranked_in_category", "popular_in_category", "trending_by_interactions"}
    recommender = Recommender(artifacts=artifacts)
    for code in range(min(artifacts.encoding.n_users, 15)):
        response = recommender.recommend(artifacts.encoding.user_codes[code], k=4)
        assert {item.explanation.template for item in response.items} <= allowed


def test_k_is_clamped_to_the_documented_range(artifacts: ServingArtifacts) -> None:
    """K is clamped into 1..MAX_K rather than rejected, so a client cannot get an empty page."""
    recommender = Recommender(artifacts=artifacts)
    user = artifacts.encoding.user_codes[0]

    assert len(recommender.recommend(user, k=0).items) == 1
    assert recommender.recommend(user, k=10_000).k == MAX_K
    assert len(recommender.recommend(user).items) == DEFAULT_K


def test_k_above_the_catalogue_size_returns_everything_available(
    artifacts: ServingArtifacts,
) -> None:
    """Asking for more items than exist returns everything available, never an invented item.

    k is still honoured as the request. The shortfall is the catalogue minus what the user has
    already seen, which is why the synthetic fixture, holding fewer items than MAX_K, cannot prove
    the cap returns 50 items.
    """
    recommender = Recommender(artifacts=artifacts)
    code = 0
    response = recommender.recommend(artifacts.encoding.user_codes[code], k=MAX_K)
    seen = int(np.diff(artifacts.history_indptr)[code])

    assert len(response.items) == min(MAX_K, artifacts.n_items - seen)
    assert len({item.item_id for item in response.items}) == len(response.items)


def test_metrics_count_requests_and_fallbacks_separately(artifacts: ServingArtifacts) -> None:
    """Warm, cold-start and degraded requests land in distinct metric series."""
    recommender = Recommender(artifacts=artifacts)
    user = artifacts.encoding.user_codes[0]
    recommender.recommend(user, k=2)
    recommender.recommend("nobody", k=2)
    recommender.source = _Exploding
    recommender.recommend(user, k=2)

    text = _exposition(recommender)
    assert 'recommend_requests_total{fallback="none"}' in text
    assert 'recommend_requests_total{fallback="cold_start"}' in text
    assert 'recommend_requests_total{fallback="index"}' in text
    assert 'fallback_total{reason="cold_start"}' in text


def test_request_carries_a_unique_id_and_version(artifacts: ServingArtifacts) -> None:
    """Every response carries a request id and the artefact checksum that served it."""
    recommender = Recommender(artifacts=artifacts)
    first = recommender.recommend(artifacts.encoding.user_codes[0], k=2)
    second = recommender.recommend(artifacts.encoding.user_codes[0], k=2)

    assert first.request_id != second.request_id
    assert first.model_version == "a" * 16
    assert first.index_version == "b" * 16
    assert first.served_at_ms > 0


def test_response_is_json_serialisable(artifacts: ServingArtifacts) -> None:
    """The response renders to JSON with no numpy scalars leaking through."""
    payload = (
        Recommender(artifacts=artifacts).recommend(artifacts.encoding.user_codes[0], k=3).as_dict()
    )

    restored = json.loads(json.dumps(payload))
    assert restored["items"][0]["rank"] == 1
    assert restored["items"][0]["explanation"]["template"]


def test_manifest_detects_a_changed_artifact(trained: Trained, test_settings: Settings) -> None:
    """A tampered artefact is refused at load time rather than served silently."""
    root = test_settings.artifact_root / trained.spec.name
    root.mkdir(parents=True, exist_ok=True)
    artefact = root / "popularity.npy"
    np.save(artefact, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    build_manifest(root)
    assert verify(root)["popularity.npy"] == sha256_file(artefact)

    np.save(artefact, np.array([9.0, 9.0, 9.0], dtype=np.float32))
    with pytest.raises(ValueError, match="does not match"):
        verify(root)


def test_manifest_detects_a_missing_artifact(trained: Trained, test_settings: Settings) -> None:
    """A listed artefact that has been deleted is a load failure, not an empty model."""
    root = test_settings.artifact_root / trained.spec.name
    root.mkdir(parents=True, exist_ok=True)
    artefact = root / "popularity.npy"
    np.save(artefact, np.zeros(3, dtype=np.float32))
    build_manifest(root)
    artefact.unlink()

    with pytest.raises(FileNotFoundError, match="but it is missing"):
        verify(root)


def test_manifest_is_required_when_verification_is_on(tmp_path: Path) -> None:
    """Serving with no manifest at all refuses rather than trusting unverified files."""
    with pytest.raises(FileNotFoundError, match="no serving manifest"):
        verify(tmp_path)


def test_titles_may_be_omitted(artifacts: ServingArtifacts) -> None:
    """Starting without titles is supported: the response carries `title: null`, not a crash."""
    without = replace(artifacts, catalogue=_catalogue(artifacts.n_items, with_titles=False))
    response = Recommender(artifacts=without).recommend(artifacts.encoding.user_codes[0], k=2)

    assert len(response.items) == 2
    assert all(item.title is None for item in response.items)


def test_served_source_is_fusion_when_cooccurrence_is_loaded(artifacts: ServingArtifacts) -> None:
    """Fuse two retrievers, because the best single one is a global ranking."""
    assert isinstance(artifacts.served_source(), FusedSource)


def test_served_source_falls_back_to_global_without_cooccurrence(
    artifacts: ServingArtifacts,
) -> None:
    """With co-occurrence unloaded the memory-constrained configuration still serves."""
    without = replace(artifacts, cooccurrence=None)
    assert isinstance(without.served_source(), PopularityModel)
    assert len(Recommender(artifacts=without).recommend("nobody", k=2).items) == 2


def test_http_surface_returns_200_for_unknown_and_known_users(
    artifacts: ServingArtifacts,
) -> None:
    """The endpoint never 404s or 500s on a user-facing query; the request id comes back too."""
    client = TestClient(create_app(artifacts))

    for user in (artifacts.encoding.user_codes[0], "not-a-user"):
        response = client.get("/recommend", params={"user_id": user, "k": 3})
        assert response.status_code == 200
        body = response.json()
        assert len(body["items"]) == 3
        assert response.headers["X-Request-Id"] == body["request_id"]


def test_http_surface_reports_health_and_metrics(artifacts: ServingArtifacts) -> None:
    """/healthz reports load state and /metrics exposes the exposition."""
    client = TestClient(create_app(artifacts))

    health = client.get("/healthz").json()
    assert health["status"] == "ok"
    assert health["n_items"] == artifacts.n_items
    assert health["titles"] is True

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "recommend_requests_total" in metrics.text


def test_load_artifacts_rejects_a_corrupt_artefact(
    trained: Trained, test_settings: Settings
) -> None:
    """The loader verifies before serving: a hash mismatch stops startup, it does not warn."""
    root = test_settings.artifact_root / trained.spec.name
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "popularity.npy", np.zeros(3, dtype=np.float32))
    build_manifest(root)
    np.save(root / "popularity.npy", np.ones(3, dtype=np.float32))

    with pytest.raises(ValueError, match="does not match"):
        load_artifacts(test_settings.artifact_root, trained.spec.name, with_titles=False)


def _write_items_parquet(path: Path, rows: list[tuple[str, str, float | None, str | None]]) -> None:
    """Write a minimal items table with the columns the catalogue reader selects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(database=":memory:")
    connection.execute(
        "CREATE TABLE items(parent_asin VARCHAR, main_category VARCHAR, price DOUBLE,"
        " title VARCHAR)"
    )
    connection.executemany("INSERT INTO items VALUES (?, ?, ?, ?)", rows)
    connection.execute(f"COPY items TO {path.as_posix()!r} (FORMAT PARQUET)")
    connection.close()


def test_load_catalogue_indexes_metadata_by_item_code(
    trained: Trained, test_settings: Settings
) -> None:
    """Titles, categories and prices land on the right codes, and unknown items stay empty."""
    codes = trained.encoding.item_codes
    rows: list[tuple[str, str, float | None, str | None]] = [
        (codes[0], "cat_x", 12.5, "First"),
        (codes[1], "", None, None),
        ("NOT-IN-ENCODING", "cat_y", 1.0, "Ignored"),
    ]
    _write_items_parquet(
        test_settings.processed(trained.spec.name) / "items" / "items.parquet", rows
    )
    catalogue = load_catalogue(
        trained.spec.name, trained.encoding, with_titles=True, config=test_settings
    )

    assert catalogue.describe(0) == ("First", "cat_x", 12.5)
    assert catalogue.describe(1) == (None, None, None)
    assert catalogue.titles is not None


def test_load_catalogue_can_skip_titles(trained: Trained, test_settings: Settings) -> None:
    """Without titles the table still loads, and describe reports no title at all."""
    _write_items_parquet(
        test_settings.processed(trained.spec.name) / "items" / "items.parquet",
        [(trained.encoding.item_codes[0], "cat_x", 3.0, "First")],
    )
    catalogue = load_catalogue(
        trained.spec.name, trained.encoding, with_titles=False, config=test_settings
    )

    assert catalogue.titles is None
    assert catalogue.describe(0) == (None, "cat_x", 3.0)


def test_seen_mask_marks_only_the_users_history(artifacts: ServingArtifacts) -> None:
    """The seen mask is per-user, not shared: one user's items are not another user's."""
    first_mask = artifacts.seen_mask(0)
    second_mask = artifacts.seen_mask(1)

    assert first_mask.sum() == np.diff(artifacts.history_indptr)[0]
    assert second_mask.sum() == np.diff(artifacts.history_indptr)[1]
    assert np.array_equal(first_mask, second_mask) is False
    assert not first_mask.all()


def test_seen_mask_is_empty_for_a_user_with_no_history(artifacts: ServingArtifacts) -> None:
    """A user with an empty history slice yields an all-false mask rather than misindexing."""
    empty = np.flatnonzero(np.diff(artifacts.history_indptr) == 0)
    if not empty.size:
        pytest.skip("the synthetic corpus has no history-free user")
    mask = artifacts.seen_mask(int(empty[0]))
    assert not mask.any()
    assert mask.size == artifacts.n_items
