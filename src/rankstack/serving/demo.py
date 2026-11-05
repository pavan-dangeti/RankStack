"""The demo page: curated shoppers, served recommendations, and the served blend against popularity.

Only what the page needs is exposed. Shoppers are addressed by their position in a committed
fixture, so no reviewer id ever reaches the browser, and there is no general history endpoint: the
histories shown are the fixture's, for its shoppers only. Blended mode is the app's own served
recommender, so a demo request is a real request, counted in `/metrics` like any other; popular
mode is the same request path with the co-occurrence weight set to 0.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse

from rankstack.retrieval.base import top_k
from rankstack.retrieval.fusion import DEFAULT_DEPTH
from rankstack.serving.recommender import Recommender

FIXTURE_PATH = Path(__file__).with_name("demo_shoppers.json")
PAGE_PATH = Path(__file__).with_name("demo.html")
DEMO_K = 10
#: Not a reviewer id: unknown to every encoding, so the request takes the cold-start path.
ANONYMOUS_USER_ID = "demo-anonymous-visitor"

Mode = Literal["popular", "blended"]


@dataclass(frozen=True, slots=True)
class Shopper:
    """One curated demo user. `user_id` stays on the server."""

    label: str
    theme: str
    user_id: str
    history: tuple[dict[str, object], ...]


def load_shoppers(path: Path = FIXTURE_PATH) -> tuple[Shopper, ...]:
    """Read the committed shopper fixture."""
    payload = json.loads(path.read_text())
    return tuple(
        Shopper(
            label=str(entry["label"]),
            theme=str(entry["theme"]),
            user_id=str(entry["user_id"]),
            history=tuple(entry["history"]),
        )
        for entry in payload["shoppers"]
    )


def source_badges(
    recommender: Recommender, user_code: int | None, item_ids: list[str]
) -> list[str]:
    """Name the source that ranked each served item: `trending`, `co-occurrence` or `both`.

    Membership is each source's top `DEFAULT_DEPTH` non-zero scores, the same lists the fusion
    ranks, so a badge says which list the item was in, not which one moved it most.
    """
    artifacts = recommender.artifacts
    if user_code is None or artifacts.served_source() is artifacts.popularity_recent:
        return ["trending"] * len(item_ids)
    assert artifacts.cooccurrence is not None  # noqa: S101 - served_source fuses only with it
    seen = artifacts.seen_mask(user_code)
    trending_codes, _ = top_k(artifacts.popularity_recent.global_scores(seen), DEFAULT_DEPTH)
    pair_codes, pair_scores = top_k(artifacts.cooccurrence.score(user_code, seen), DEFAULT_DEPTH)
    trending = {int(code) for code in trending_codes}
    paired = {int(code) for code in pair_codes[pair_scores != 0]}
    badges: list[str] = []
    for item_id in item_ids:
        code = artifacts.encoding.item_to_code[item_id]
        in_trending, in_paired = code in trending, code in paired
        badges.append(
            "both" if in_trending and in_paired else "co-occurrence" if in_paired else "trending"
        )
    return badges


def build_demo_router(served: Recommender, shoppers: tuple[Shopper, ...]) -> APIRouter:
    """Return the `/demo` routes around the served recommender and the curated shoppers."""
    unknown = [s.label for s in shoppers if s.user_id not in served.artifacts.encoding.user_to_code]
    if unknown:
        msg = f"demo shoppers missing from the encoding: {', '.join(unknown)}"
        raise ValueError(msg)
    recommenders: dict[str, Recommender] = {
        "blended": served,
        "popular": Recommender(artifacts=replace(served.artifacts, cooccurrence_weight=0.0)),
    }
    page = PAGE_PATH.read_text()
    router = APIRouter(prefix="/demo")

    @router.get("", response_class=HTMLResponse)
    def demo_page() -> str:
        """Serve the page: one file, no build step, nothing loaded from elsewhere."""
        return page

    @router.get("/shoppers")
    def demo_shoppers() -> dict[str, object]:
        """Shopper labels, themes and histories, and the two weights the toggle switches between."""
        return {
            "shoppers": [
                {"id": number, "label": s.label, "theme": s.theme, "history": list(s.history)}
                for number, s in enumerate(shoppers, start=1)
            ],
            "weights": {name: r.artifacts.cooccurrence_weight for name, r in recommenders.items()},
        }

    @router.get("/recommend")
    def demo_recommend(
        shopper: Annotated[str, Query(pattern=r"^(anonymous|[1-9][0-9]?)$")],
        mode: Annotated[Mode, Query()] = "blended",
    ) -> dict[str, object]:
        """Recommend for a shopper by position, or for an anonymous visitor."""
        if shopper == "anonymous":
            label, user_id = "Anonymous visitor", ANONYMOUS_USER_ID
        elif int(shopper) <= len(shoppers):
            chosen = shoppers[int(shopper) - 1]
            label, user_id = chosen.label, chosen.user_id
        else:
            raise HTTPException(status_code=422, detail=f"no shopper {shopper}")
        recommender = recommenders[mode]
        payload = recommender.recommend(user_id, DEMO_K).as_dict()
        del payload["user_id"]
        items = payload["items"]
        assert isinstance(items, list)  # noqa: S101 - narrows the type of as_dict()'s value
        user_code = recommender.artifacts.encoding.user_to_code.get(user_id)
        degraded = payload["fallback_used"] in {"index", "retriever"}
        badges = (
            ["trending"] * len(items)
            if degraded
            else source_badges(recommender, user_code, [str(item["item_id"]) for item in items])
        )
        for item, badge in zip(items, badges, strict=True):
            item["source"] = badge
        payload.update(shopper=label, mode=mode, weight=recommender.artifacts.cooccurrence_weight)
        return payload

    return router
