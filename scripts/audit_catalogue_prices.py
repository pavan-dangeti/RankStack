"""Record how implausible the published Amazon Devices prices are.

    uv run python scripts/audit_catalogue_prices.py

The catalogue's `price` is passed through as published. For Amazon's own devices it is not a
plausible retail price: this counts the priced items in that category, how many are listed under
one dollar, and keeps a few Echo examples, so the README can say so with figures it can point to.
"""

from __future__ import annotations

import argparse
import json

import duckdb

from rankstack.config import settings
from rankstack.data.corpus import CORPORA, DEFAULT_CORPUS, get_corpus

CATEGORY = "Amazon Devices"
THRESHOLD = 1.0
EXAMPLES = 3


def main(argv: list[str] | None = None) -> int:
    """Count implausible prices in one category and write the result file."""
    parser = argparse.ArgumentParser(description="Audit published Amazon Devices prices.")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, choices=sorted(CORPORA))
    args = parser.parse_args(argv)
    corpus = get_corpus(args.corpus).name
    items = (settings.processed(corpus) / "items" / "items.parquet").as_posix()

    connection = duckdb.connect(database=":memory:")
    counted = connection.execute(
        """
        SELECT count(*), count(price), count(*) FILTER (WHERE price < ?)
        FROM read_parquet(?) WHERE main_category = ?
        """,
        [THRESHOLD, items, CATEGORY],
    ).fetchone()
    examples = connection.execute(
        """
        SELECT parent_asin, title, price FROM read_parquet(?)
        WHERE main_category = ? AND price < ? AND title ILIKE 'echo%'
        ORDER BY parent_asin LIMIT ?
        """,
        [items, CATEGORY, THRESHOLD, EXAMPLES],
    ).fetchall()
    connection.close()
    assert counted is not None  # noqa: S101 - an aggregate always returns one row

    total, priced, below = counted
    payload = {
        "command": "uv run python scripts/audit_catalogue_prices.py",
        "corpus": corpus,
        "category": CATEGORY,
        "threshold": THRESHOLD,
        "items": int(total),
        "priced": int(priced),
        "priced_below_threshold": int(below),
        "examples": {asin: {"title": title, "price": price} for asin, title, price in examples},
    }
    out = settings.results_root / f"catalogue_prices_{corpus}.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"{below} of {priced} priced {CATEGORY} items are under {THRESHOLD}; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
