"""Tests for the demo page's fixture and routes."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from scripts.build_demo_fixture import THEMES

from rankstack.serving.app import create_app
from rankstack.serving.demo import (
    FIXTURE_PATH,
    Shopper,
    build_demo_router,
    load_shoppers,
    source_badges,
)
from rankstack.serving.recommender import Recommender
from tests.conftest import demo_parts

if TYPE_CHECKING:
    from rankstack.serving.artifacts import ServingArtifacts
    from tests.conftest import Trained


def test_committed_fixture_holds_only_the_curated_shoppers() -> None:
    """Eight shoppers, one per theme, and nothing beyond what the page shows."""
    payload = json.loads(FIXTURE_PATH.read_text())
    shoppers = payload["shoppers"]

    assert set(payload) == {"rule", "corpus", "seed", "shoppers"}
    assert [s["label"] for s in shoppers] == [f"Shopper {n}" for n in range(1, len(THEMES) + 1)]
    assert [s["theme"] for s in shoppers] == [theme for theme, _ in THEMES]
    assert len({s["user_id"] for s in shoppers}) == len(shoppers)
    for shopper in shoppers:
        assert set(shopper) == {"label", "theme", "user_id", "history"}
        assert 5 <= len(shopper["history"]) <= 8
        for entry in shopper["history"]:
            assert set(entry) == {"title", "category", "year"}
    assert len(load_shoppers()) == len(THEMES)


@pytest.fixture
def parts(trained: Trained) -> tuple[ServingArtifacts, tuple[Shopper, ...]]:
    """Build a synthetic artefact set with three demo shoppers."""
    return demo_parts(trained)


@pytest.fixture
def client(parts: tuple[ServingArtifacts, tuple[Shopper, ...]]) -> TestClient:
    """Serve the real app with the demo routes mounted over the synthetic artefacts."""
    return TestClient(create_app(*parts))


def test_page_loads_nothing_from_elsewhere(client: TestClient) -> None:
    """No CDN, no external font or script: every URL the page uses is relative."""
    page = client.get("/demo")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert not re.search(r"https?://|//[a-z0-9.-]+\.[a-z]{2,}/", page.text)


def test_shoppers_never_carry_a_reviewer_id(
    client: TestClient, parts: tuple[ServingArtifacts, tuple[Shopper, ...]]
) -> None:
    """The browser addresses shoppers by position; their ids stay on the server."""
    body = client.get("/demo/shoppers").text
    for shopper in parts[1]:
        assert shopper.user_id not in body
    assert "user_id" not in body
    recommendation = client.get("/demo/recommend", params={"shopper": "1"}).text
    assert parts[1][0].user_id not in recommendation


@pytest.mark.parametrize(
    ("params", "status"),
    [
        ({"shopper": "1", "mode": "popular"}, 200),
        ({"shopper": "1", "mode": "blended"}, 200),
        ({"shopper": "1"}, 200),
        ({"shopper": "1", "mode": "both"}, 422),
        ({"shopper": "1", "mode": "0.75"}, 422),
        ({"shopper": "4"}, 422),
        ({"shopper": "0"}, 422),
        ({"shopper": "user-a"}, 422),
    ],
)
def test_query_parameters_accept_exactly_the_documented_values(
    client: TestClient, params: dict[str, str], status: int
) -> None:
    """Mode is popular or blended; a shopper is a fixture position or anonymous; nothing else."""
    assert client.get("/demo/recommend", params=params).status_code == status


def test_anonymous_visitor_takes_the_cold_start_path(client: TestClient) -> None:
    """Every anonymous item is from the trending list, and the response says why."""
    body = client.get("/demo/recommend", params={"shopper": "anonymous"}).json()
    assert body["fallback_used"] == "cold_start"
    assert body["shopper"] == "Anonymous visitor"
    assert {item["source"] for item in body["items"]} == {"trending"}


def test_modes_serve_their_own_weight_and_differ_for_the_first_shopper(
    client: TestClient,
) -> None:
    """The toggle reaches the request path: weight 0 is popularity alone, and the lists differ."""
    popular = client.get("/demo/recommend", params={"shopper": "1", "mode": "popular"}).json()
    blended = client.get("/demo/recommend", params={"shopper": "1", "mode": "blended"}).json()

    assert (popular["weight"], blended["weight"]) == (0.0, 0.75)
    assert {item["source"] for item in popular["items"]} == {"trending"}
    assert [i["item_id"] for i in popular["items"]] != [i["item_id"] for i in blended["items"]]
    for field in ("elapsed_ms", "fallback_used", "model_version", "request_id"):
        assert field in blended


def test_badges_name_the_lists_each_item_came_from(
    parts: tuple[ServingArtifacts, tuple[Shopper, ...]],
) -> None:
    """An item is badged with co-occurrence exactly when co-occurrence scored it."""
    artifacts, shoppers = parts
    recommender = Recommender(artifacts=artifacts)
    code = artifacts.encoding.user_to_code[shoppers[0].user_id]
    items = [item.item_id for item in recommender.recommend(shoppers[0].user_id, 10).items]
    badges = source_badges(recommender, code, items)

    assert artifacts.cooccurrence is not None
    paired = artifacts.cooccurrence.score(code, artifacts.seen_mask(code))
    for item_id, badge in zip(items, badges, strict=True):
        has_pair_score = paired[artifacts.encoding.item_to_code[item_id]] > 0
        assert (badge in {"both", "co-occurrence"}) == has_pair_score


def test_router_refuses_a_shopper_the_encoding_does_not_know(
    parts: tuple[ServingArtifacts, tuple[Shopper, ...]],
) -> None:
    """A fixture built for another corpus must fail at startup, not serve cold starts silently."""
    artifacts, _ = parts
    stranger = Shopper(label="Shopper 1", theme="x", user_id="not-encoded", history=())
    with pytest.raises(ValueError, match="missing from the encoding"):
        build_demo_router(Recommender(artifacts=artifacts), (stranger,))
