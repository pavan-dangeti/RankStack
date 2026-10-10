"""The HTTP layer.

Thin on purpose: it parses, delegates to `Recommender`, and renders. Every policy decision —
degradation, explanation choice, which retriever serves — lives in the request path so it can be
tested without a server, and so the load test measures the same code the tests assert on.

`/metrics` exposes the Prometheus registry. `/healthz` reports load state without touching a model,
so an orchestrator can probe liveness cheaply and separately from readiness.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Annotated

from fastapi import FastAPI, Query, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from rankstack.config import Settings
from rankstack.config import settings as default_settings
from rankstack.logging import configure_logging, get_logger
from rankstack.serving.artifacts import load_artifacts
from rankstack.serving.recommender import DEFAULT_K, MAX_K, Recommender

if TYPE_CHECKING:
    from rankstack.serving.artifacts import ServingArtifacts

_logger = get_logger(__name__)

CORPUS = "electronics_0core"


def create_app(artifacts: ServingArtifacts) -> FastAPI:
    """Build the ASGI application around an already-loaded artefact set.

    Artefacts are passed in rather than loaded here so the failure-injection tests can start
    an app with a deliberately broken source and assert the degradation path, which is impossible if
    loading happens during startup and takes the whole process down with it.
    """
    configure_logging()
    recommender = Recommender(artifacts=artifacts)
    app = FastAPI(
        title="RankStack recommender",
        version="0.1.0",
        description="Recommendation service. Degrades rather than failing.",
    )
    app.state.recommender = recommender

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        """Liveness and readiness: artefacts loaded, catalogue size, wall clock."""
        return {
            "status": "ok",
            "corpus": artifacts.corpus,
            "n_items": artifacts.n_items,
            "cooccurrence": artifacts.cooccurrence is not None,
            "titles": artifacts.catalogue.titles is not None,
            "time_ms": int(time.time() * 1000),
        }

    @app.get("/recommend")
    def recommend(
        user_id: Annotated[str, Query(min_length=1, description="External user id.")],
        k: Annotated[int, Query(ge=1, le=MAX_K, description="Items to return.")] = DEFAULT_K,
    ) -> Response:
        """Return `k` ranked items for `user_id`.

        Always 200 with a non-empty list. Unknown users and failed sources degrade rather than 404
        or 500, which is the contract the design document's section 6.9 table specifies.
        """
        payload = recommender.recommend(user_id, k).as_dict()
        return Response(
            content=json.dumps(payload),
            media_type="application/json",
            headers={"X-Request-Id": str(payload["request_id"])},
        )

    @app.get("/metrics")
    def metrics() -> Response:
        """Prometheus exposition for the request path."""
        exposition = generate_latest(recommender.metrics.registry)
        return Response(content=exposition, media_type=CONTENT_TYPE_LATEST)

    return app


def build_default_app(config: Settings | None = None) -> FastAPI:
    """Load artefacts for the default corpus and build the app. Used by the container entrypoint."""
    resolved = config or default_settings
    artifacts = load_artifacts(resolved.artifact_root, CORPUS, with_titles=True)
    return create_app(artifacts)


_app: FastAPI | None = None


def __getattr__(name: str) -> object:
    """Build the app lazily on first attribute access.

    A module-level `app` would load hundreds of megabytes of artefacts at import time, which breaks
    `uvicorn --reload` and makes every test that imports this module pay for it.
    """
    global _app  # noqa: PLW0603 - one-time lazy construction, guarded by the module cache
    if name == "app":
        _app = build_default_app()
        return _app
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
