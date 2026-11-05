r"""Measure the serving process's memory: after loading, after requests, and at its peak.

    uv run python scripts/measure_memory.py                  # this machine

    docker run --rm -v "$PWD/data:/data:ro" -v "$PWD/artifacts:/artifacts:ro" \\
      -v "$PWD/scripts:/app/scripts:ro" -v "$PWD/results:/results" \\
      -e RANKSTACK_RESULTS_ROOT=/results --entrypoint python rankstack-serve \\
      /app/scripts/measure_memory.py                         # Linux, inside the serving image

The app is built exactly as the server builds it, then 120 requests go through its request path:
100 known users and 20 unknown ones. Resident memory is read from the process itself, from
/proc/self/statm on Linux and `ps` on macOS; the peak is the kernel's maximum resident set size,
which covers the transient cost of loading that a reading after the load would miss.
"""

from __future__ import annotations

import itertools
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

from rankstack.config import settings
from rankstack.serving.app import CORPUS, build_default_app

KNOWN_REQUESTS = 100
UNKNOWN_REQUESTS = 20
MIB = 1024 * 1024
# Detected at runtime, not through `sys.platform`, which type checkers narrow per platform and then
# report the other branch as unreachable.
STATM = Path("/proc/self/statm")


def resident_bytes() -> int:
    """Return this process's current resident set size."""
    if STATM.exists():
        resident_pages = int(STATM.read_text().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    out = subprocess.run(  # noqa: S603 - fixed argv, this process's own pid
        ["/bin/ps", "-o", "rss=", "-p", str(os.getpid())],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip()) * 1024


def peak_bytes() -> int:
    """Return this process's maximum resident set size so far."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is in bytes on macOS and in kilobytes on Linux.
    return int(peak) if platform.system() == "Darwin" else int(peak) * 1024


def main() -> int:
    """Load the app, serve requests, and write the memory figures."""
    started = time.perf_counter()
    app = build_default_app()
    load_seconds = time.perf_counter() - started
    after_load = resident_bytes()

    recommender = app.state.recommender
    user_ids = iter(recommender.artifacts.encoding.user_to_code)
    known = list(itertools.islice(user_ids, KNOWN_REQUESTS))
    unknown = [f"memory-probe-unknown-{index}" for index in range(UNKNOWN_REQUESTS)]
    for user in [*known, *unknown]:
        recommender.recommend(user, 20)
    after_requests = resident_bytes()

    container = Path("/.dockerenv").exists()
    name = f"{sys.platform}-{platform.machine()}{'-container' if container else ''}"
    payload = {
        "command": (
            "uv run python scripts/measure_memory.py"
            if not container
            else "docker run ... --entrypoint python rankstack-serve /app/scripts/measure_memory.py"
        ),
        "corpus": CORPUS,
        "platform": name,
        "python": platform.python_version(),
        "requests": {"known_users": len(known), "unknown_users": len(unknown), "k": 20},
        "artifact_load_seconds": round(load_seconds, 2),
        "rss_after_load_mib": round(after_load / MIB, 1),
        "rss_after_requests_mib": round(after_requests / MIB, 1),
        "peak_rss_mib": round(peak_bytes() / MIB, 1),
        "definitions": {
            "rss_after_load_mib": "resident set size once the app is built",
            "rss_after_requests_mib": "resident set size after the requests above",
            "peak_rss_mib": "maximum resident set size of the process, loading included",
        },
    }
    out = settings.results_root / f"memory_{CORPUS}_{name}.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: payload[k] for k in payload if k.endswith("_mib")}), f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
