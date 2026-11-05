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
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import duckdb
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse

from rankstack.data.encode import CodeIndex
from rankstack.retrieval.base import top_k
from rankstack.retrieval.fusion import DEFAULT_DEPTH
from rankstack.serving.recommender import Recommender

if TYPE_CHECKING:
    from rankstack.data.encode import Encoding

FIXTURE_PATH = Path(__file__).with_name("demo_shoppers.json")
PAGE_PATH = Path(__file__).with_name("demo.html")
DEMO_K = 10
#: Not a reviewer id: unknown to every encoding, so the request takes the cold-start path.
ANONYMOUS_USER_ID = "demo-anonymous-visitor"

Mode = Literal["popular", "blended"]


@dataclass(frozen=True, slots=True)
class ShopperSpec:
    """One curated demo user as committed: a label, a theme and an internal user code, nothing else.

    The code is an index into the locally built encoding. It identifies nobody outside this
    process, which is why the committed fixture holds it rather than a reviewer id or a history.
    """

    label: str
    theme: str
    user_code: int


@dataclass(frozen=True, slots=True)
class Shopper:
    """One demo user resolved on the server. `user_id` and `history` never leave the process."""

    label: str
    theme: str
    user_id: str
    history: tuple[dict[str, object], ...]


def load_shopper_specs(path: Path = FIXTURE_PATH) -> tuple[ShopperSpec, ...]:
    """Read the committed shopper fixture."""
    payload = json.loads(path.read_text())
    return tuple(
        ShopperSpec(
            label=str(entry["label"]),
            theme=str(entry["theme"]),
            user_code=int(entry["user_code"]),
        )
        for entry in payload["shoppers"]
    )


def _user_id(encoding: Encoding, code: int) -> str | None:
    users = encoding.user_to_code
    if isinstance(users, CodeIndex):
        return users.id_of(code)
    return encoding.user_codes.get(code)


def resolve_shoppers(
    specs: tuple[ShopperSpec, ...], encoding: Encoding, processed: Path
) -> tuple[Shopper, ...]:
    """Turn committed codes into shoppers, reading each history from the local train window.

    The history is what the page's left panel shows: each distinct train-window item with its
    catalogue title and category and the year of the user's latest interaction with it, newest
    first.
    """
    ids = [_user_id(encoding, spec.user_code) for spec in specs]
    unknown = [spec.label for spec, user in zip(specs, ids, strict=True) if user is None]
    if unknown:
        msg = f"demo shoppers missing from the encoding: {', '.join(unknown)}"
        raise ValueError(msg)
    train = processed / "interactions" / "train.parquet"
    items = processed / "items" / "items.parquet"
    rows = (
        duckdb.connect(database=":memory:")
        .execute(
            """
            WITH hist AS (
                SELECT user_id, parent_asin, max(timestamp) AS ts
                FROM read_parquet(?) WHERE list_contains(?, user_id) GROUP BY ALL
            )
            SELECT hist.user_id, items.title, items.main_category, hist.ts
            FROM hist LEFT JOIN read_parquet(?) AS items USING (parent_asin)
            ORDER BY hist.user_id, hist.ts DESC, hist.parent_asin
            """,
            [train.as_posix(), ids, items.as_posix()],
        )
        .fetchall()
    )
    histories: dict[str, list[dict[str, object]]] = {user: [] for user in ids if user}
    for user, title, category, ts in rows:
        histories[user].append(
            {
                "title": title or "Untitled item",
                "category": category or "Uncategorised",
                "year": datetime.fromtimestamp(ts / 1000, tz=UTC).year,
            }
        )
    return tuple(
        Shopper(label=spec.label, theme=spec.theme, user_id=user, history=tuple(histories[user]))
        for spec, user in zip(specs, ids, strict=True)
        if user is not None
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
    popular_artifacts = replace(served.artifacts, cooccurrence_weight=0.0)
    recommenders: dict[str, Recommender] = {
        "blended": served,
        "popular": Recommender(artifacts=popular_artifacts),
    }
    # Answers "would the other mode show the same list?" through recommenders whose metrics are
    # never exported, so `/metrics` still counts one served request per demo request.
    shadows: dict[str, Recommender] = {
        "blended": Recommender(artifacts=served.artifacts),
        "popular": Recommender(artifacts=popular_artifacts),
    }
    page = PAGE_PATH.read_text()
    router = APIRouter(prefix="/demo")

    @router.get("", response_class=HTMLResponse)
    def demo_page() -> str:
        """Serve the page: one file, no build step, no resource loaded from elsewhere."""
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
        other = shadows["popular" if mode == "blended" else "blended"]
        other_ids = [item.item_id for item in other.recommend(user_id, DEMO_K).items]
        payload.update(
            shopper=label,
            mode=mode,
            weight=recommender.artifacts.cooccurrence_weight,
            same_top_10_in_both_modes=other_ids == [item["item_id"] for item in items],
        )
        return payload

    return router
