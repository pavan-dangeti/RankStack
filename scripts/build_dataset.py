"""Build validated, time-split Parquet for one corpus and write its manifest."""

from __future__ import annotations

import argparse
import sys

from rankstack.config import settings
from rankstack.data.build import build_dataset, summarise, write_result_file
from rankstack.data.corpus import CORPORA, DEFAULT_CORPUS, get_corpus
from rankstack.data.leakage import (
    LeakageError,
    LeakageReport,
    check_partition_order,
    check_split_counts,
)
from rankstack.data.schema import ValidationError
from rankstack.logging import configure_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, choices=sorted(CORPORA))
    parser.add_argument(
        "--skip-download", action="store_true", help="assume raw files are already present"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the script and return a process exit code."""
    args = parse_args(argv)
    configure_logging()
    corpus = get_corpus(args.corpus)
    try:
        manifest = build_dataset(corpus, settings=settings, skip_download=args.skip_download)
    except (ValidationError, LeakageError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    report = LeakageReport()
    observed = {p.partition: p.rows for p in manifest.partitions}
    observed_ranges = {
        p.partition: (p.min_timestamp_ms, p.max_timestamp_ms) for p in manifest.partitions
    }
    check_partition_order(report, observed_ranges, corpus)
    check_split_counts(report, observed, corpus)
    if not report.clean:
        report.raise_on_leak()

    command = f"uv run python scripts/build_dataset.py --corpus {args.corpus}"
    if args.skip_download:
        command += " --skip-download"
    result_path = write_result_file(manifest, command, settings.results_root)

    print(summarise(manifest))
    print(f"leakage checks: clean\nresult file: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
