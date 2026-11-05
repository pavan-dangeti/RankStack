"""Record the demo page: the README GIF and the release walkthrough video.

    uv run python -m uvicorn rankstack.serving.app:app       # the real service, real artefacts
    uv run python scripts/record_demo.py gif                 # docs/media/demo.gif
    uv run python scripts/record_demo.py mp4 --out demo.mp4  # the two-minute release video

Playwright drives the page and records it; ffmpeg trims the blank lead-in and converts. The GIF is
the short story the README opens with: Shopper 5, the ranking toggle, the anonymous visitor. The
video walks through more shoppers, including one the toggle does not change, for about two minutes.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import Page, ViewportSize, expect, sync_playwright

REPO_ROOT = Path(__file__).resolve().parents[1]
GIF_PATH = REPO_ROOT / "docs" / "media" / "demo.gif"
# Mirrors MAX_ALLOWED_BYTES in scripts/check_no_dataset_files.py, which CI enforces on every file.
GIF_LIMIT_BYTES = 1_000_000
VIEWPORT: ViewportSize = {"width": 1040, "height": 720}


def _settle(page: Page) -> None:
    expect(page.locator("#recs li")).to_have_count(10)
    expect(page.locator("#recs")).not_to_have_attribute("aria-busy", "true")


def _choose_mode(page: Page, mode: str, hold: float) -> None:
    page.click(f"label:has(input[value={mode}])")
    _settle(page)
    page.wait_for_timeout(hold * 1000)


def _choose_shopper(page: Page, value: str, hold: float) -> None:
    page.select_option("#shopper", value)
    _settle(page)
    page.wait_for_timeout(hold * 1000)


def short_story(page: Page) -> None:
    """Shopper 5 in the served blend, the same shopper popular-only, then an anonymous visitor."""
    page.wait_for_timeout(2500)
    _choose_mode(page, "popular", 3)
    _choose_mode(page, "blended", 2)
    _choose_shopper(page, "anonymous", 3.5)


def long_story(page: Page) -> None:
    """About two minutes: several shoppers, both modes, the same-list note, a cold start."""
    page.wait_for_timeout(6000)
    _choose_mode(page, "popular", 7)
    _choose_mode(page, "blended", 6)
    for shopper in ("2", "1", "6", "8", "3"):
        _choose_shopper(page, shopper, 8)
        _choose_mode(page, "popular", 6)
        _choose_mode(page, "blended", 4)
    _choose_shopper(page, "anonymous", 8)
    page.mouse.wheel(0, 2000)
    page.wait_for_timeout(6000)


def record(url: str, story: str, workdir: Path) -> tuple[Path, float]:
    """Record one story; return the video and the seconds of blank lead-in before it is ready."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(
            viewport=VIEWPORT,
            record_video_dir=str(workdir),
            record_video_size=VIEWPORT,
            color_scheme="light",
        )
        page = context.new_page()
        opened = time.monotonic()
        page.goto(f"{url}/demo")
        _settle(page)
        lead_in = time.monotonic() - opened
        (short_story if story == "short" else long_story)(page)
        video = page.video
        assert video is not None  # noqa: S101 - the context records video
        context.close()  # the video file is complete only once its context closes
        path = Path(video.path())
        browser.close()
    return path, lead_in


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)  # noqa: S603, S607


def to_gif(video: Path, lead_in: float, out: Path) -> None:
    """Convert to a palette GIF small enough for the README."""
    # Few colours, few frames, and only the changed rectangle of each frame keep it under the limit.
    filters = (
        "fps=5,scale=880:-1:flags=lanczos,split[a][b];"
        "[a]palettegen=max_colors=48:stats_mode=diff[p];"
        "[b][p]paletteuse=dither=none:diff_mode=rectangle"
    )
    _ffmpeg("-ss", f"{lead_in:.2f}", "-i", str(video), "-vf", filters, str(out))


def to_mp4(video: Path, lead_in: float, out: Path) -> None:
    """Convert to an H.264 MP4 that plays anywhere."""
    _ffmpeg(
        "-ss", f"{lead_in:.2f}", "-i", str(video), "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "26", "-movflags", "+faststart", str(out),
    )  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    """Record the GIF or the video against a running service."""
    parser = argparse.ArgumentParser(description="Record the demo page.")
    parser.add_argument("kind", choices=("gif", "mp4"))
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if shutil.which("ffmpeg") is None:
        print("error: ffmpeg is required", file=sys.stderr)
        return 1
    out: Path = args.out or (GIF_PATH if args.kind == "gif" else Path("demo.mp4"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as workdir:
        story = "short" if args.kind == "gif" else "long"
        video, lead_in = record(args.url, story, Path(workdir))
        (to_gif if args.kind == "gif" else to_mp4)(video, lead_in, out)
    size = out.stat().st_size
    print(f"wrote {out} ({size / 1024 / 1024:.2f} MB)")
    if args.kind == "gif" and size > GIF_LIMIT_BYTES:
        print(f"error: the GIF is over {GIF_LIMIT_BYTES} bytes", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
