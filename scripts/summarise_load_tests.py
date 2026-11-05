"""Summarise repeated load-test runs into medians, one set per variant.

    uv run python scripts/summarise_load_tests.py

Reads every `results/load_test_repeat_<variant>_<run>_<corpus>.json`, each the unchanged output of
`uv run python scripts/load_test.py --requests 400 --concurrency 8 --k 20` for one run of one code
variant, and writes `results/load_test_median_<corpus>.json`: per variant, the median of each
latency percentile, throughput and load time over its runs. Each figure's median is taken on its
own, so a variant's medians need not all come from the same run.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from typing import Any

from rankstack.config import settings
from rankstack.serving.app import CORPUS

#: What each variant is. The runs were interleaved on one machine: dict, arrays, dict, arrays, ...
VARIANTS = {
    "dict": "the user map as a Python dict: main at 544f891",
    "arrays": "the user map as sorted arrays searched by bisection (CodeIndex)",
}
RUN_FILE = re.compile(rf"^load_test_repeat_(?P<variant>[a-z]+)_(?P<run>\d+)_{CORPUS}\.json$")
PERCENTILES = ("p50_ms", "p95_ms", "p99_ms")


def summarise(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the per-figure medians over one variant's runs."""
    latency: dict[str, dict[str, float]] = {}
    for label in ("warm", "cold_start", "degraded"):
        rows = [next(row for row in run["latency"] if row["label"] == label) for run in runs]
        latency[label] = {
            p: round(statistics.median(float(row[p]) for row in rows), 3) for p in PERCENTILES
        }
    throughput = [float(run["throughput_requests_per_second_single_process"]) for run in runs]
    load = [float(run["artifact_load_seconds"]) for run in runs]
    return {
        "runs": len(runs),
        "latency": latency,
        "throughput_requests_per_second_single_process": round(statistics.median(throughput), 1),
        "artifact_load_seconds": round(statistics.median(load), 2),
        "all_slos_met_in_every_run": all(bool(run["all_slos_met"]) for run in runs),
    }


def main() -> int:
    """Group the run files by variant and write their medians."""
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for path in sorted(settings.results_root.glob(f"load_test_repeat_*_{CORPUS}.json")):
        match = RUN_FILE.match(path.name)
        if match is None or match["variant"] not in VARIANTS:
            msg = f"unexpected run file {path.name}"
            raise ValueError(msg)
        grouped[match["variant"]].append((int(match["run"]), json.loads(path.read_text())))
    payload: dict[str, Any] = {
        "command": "uv run python scripts/summarise_load_tests.py",
        "per_run_command": (
            "uv run python scripts/load_test.py --requests 400 --concurrency 8 --k 20"
        ),
        "variants": VARIANTS,
        "statistic": "median over runs, taken separately for each figure",
    }
    for variant in VARIANTS:
        runs = [run for _, run in sorted(grouped[variant], key=lambda pair: pair[0])]
        if not runs:
            msg = f"no runs for variant {variant}"
            raise ValueError(msg)
        payload[variant] = summarise(runs)
    out = settings.results_root / f"load_test_median_{CORPUS}.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
