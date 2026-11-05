"""End-to-end: the demo page in a real browser, against the real app over synthetic artefacts."""

from __future__ import annotations

import re
import socket
import threading
import time
from typing import TYPE_CHECKING

import pytest
import uvicorn
from playwright.sync_api import Page, expect, sync_playwright

from rankstack.serving.app import create_app
from tests.conftest import demo_parts

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tests.conftest import Trained

pytestmark = pytest.mark.e2e


@pytest.fixture
def base_url(trained: Trained) -> Iterator[str]:
    """Serve the app on a free local port for the length of one test."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    app = create_app(*demo_parts(trained))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        if time.monotonic() > deadline:
            msg = "the demo server did not start"
            raise RuntimeError(msg)
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture
def page() -> Iterator[Page]:
    """Open a headless Chromium page."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        yield browser.new_page()
        browser.close()


def _served_ids(page: Page) -> list[str]:
    expect(page.locator("#recs")).not_to_have_attribute("aria-busy", "true")
    ids: list[str] = page.locator("#recs li").evaluate_all(
        "rows => rows.map(row => row.dataset.itemId)"
    )
    return ids


def test_a_visitor_can_pick_a_shopper_flip_the_ranking_and_see_a_cold_start(
    page: Page, base_url: str
) -> None:
    """The demo's whole story: history and ten results, a toggle that changes them, a cold start."""
    page.goto(f"{base_url}/demo")
    page.select_option("#shopper", "1")
    expect(page.locator("#recs li")).to_have_count(10)
    expect(page.locator("#history li").first).to_contain_text("Item")
    expect(page.locator("#fallback")).to_have_text("null")
    expect(page.locator("#request")).to_have_text(re.compile(r"^[0-9a-f-]{36}$"))
    blended = _served_ids(page)

    # Keyboard only: focus the radio and press Space, as a keyboard user would.
    page.focus("input[name=mode][value=popular]")
    page.keyboard.press("Space")
    expect(page.locator("input[name=mode][value=popular]")).to_be_checked()
    page.wait_for_function(
        "ids => [...document.querySelectorAll('#recs li')]"
        ".map(r => r.dataset.itemId).join() !== ids",
        arg=",".join(blended),
    )
    assert _served_ids(page) != blended

    page.select_option("#shopper", "anonymous")
    expect(page.locator("#fallback")).to_have_text("cold_start")
    expect(page.locator("#status")).to_contain_text("Cold start")
    expect(page.locator("#history li")).to_contain_text("No history")
    expect(page.locator("#recs li")).to_have_count(10)
