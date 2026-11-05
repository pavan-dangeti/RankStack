"""Build the demo page's shopper fixture: eight real warm users with readable, themed histories.

    uv run python scripts/build_demo_fixture.py

The selection is a rule, not a hand pick, so it can be re-run and audited. For each theme, in
order: users with 5 to 8 distinct train-window items, every item titled and categorised, and at
least 4 items whose title matches the theme; the first such user in a seeded hash order is taken,
and no user is taken twice. Only a label, the theme and each user's internal code from the local
encoding are written: no reviewer id and no history. The server resolves a code to its user and
reads the history from the local train window at startup (`resolve_shoppers`).
"""

from __future__ import annotations

import argparse
import json

import duckdb

from rankstack.config import settings
from rankstack.data.corpus import CORPORA, DEFAULT_CORPUS, get_corpus
from rankstack.data.encode import load_serving_encoding
from rankstack.serving.demo import FIXTURE_PATH

#: (theme, case-insensitive title pattern). The order is the shopper order on the page.
THEMES: tuple[tuple[str, str], ...] = (
    ("headphones", r"headphone|earbud|earphone"),
    ("photography", r"\b(dslr|mirrorless|canon|nikon|sony alpha|tripod|camera lens)\b"),
    ("gaming", r"gaming|controller|xbox|playstation|nintendo"),
    ("computing", r"\b(laptop|keyboard|monitor|mouse|ssd)\b"),
    ("smart home", r"\b(echo|alexa|smart plug|kasa|ring video)\b"),
    ("tv and streaming", r"\b(tv|soundbar|roku|fire tv|hdmi)\b"),
    # Word boundaries matter: without them "phone case" matches "Headphone Case".
    ("phone accessories", r"\b(iphone|galaxy|phone case|screen protector|charger)\b"),
    ("speakers and audio", r"\b(speaker|speakers|microphone|turntable|receiver)\b"),
)
MIN_ITEMS, MAX_ITEMS, MIN_THEME_HITS = 5, 8, 4


def select(connection: duckdb.DuckDBPyConnection, seed: int) -> list[dict[str, object]]:
    """Apply the selection rule and return the fixture's shoppers, in theme order."""
    # LEFT JOIN: an item without metadata must disqualify its user, not silently drop out.
    connection.execute(
        f"""
        CREATE TEMP TABLE candidate AS
        WITH hist AS (
            SELECT user_id, parent_asin, max(timestamp) AS ts FROM train GROUP BY ALL
        ),
        sized AS (
            SELECT user_id FROM hist GROUP BY user_id
            HAVING count(*) BETWEEN {MIN_ITEMS} AND {MAX_ITEMS}
        )
        SELECT hist.user_id, hist.ts, items.title, items.main_category
        FROM hist JOIN sized USING (user_id) LEFT JOIN items USING (parent_asin)
        """  # noqa: S608 - integer constants, not input
    )
    chosen: list[str] = []
    shoppers: list[dict[str, object]] = []
    for number, (theme, pattern) in enumerate(THEMES, start=1):
        row = connection.execute(
            """
            SELECT user_id FROM candidate GROUP BY user_id
            HAVING count(title) = count(*) AND count(main_category) = count(*)
               AND sum(CASE WHEN regexp_matches(title, ?, 'i') THEN 1 ELSE 0 END) >= ?
               AND NOT list_contains(?, user_id)
            ORDER BY hash(user_id || 'demo' || ?)
            LIMIT 1
            """,
            [pattern, MIN_THEME_HITS, chosen, seed],
        ).fetchone()
        if row is None:
            msg = f"no user satisfies the rule for theme {theme!r}"
            raise LookupError(msg)
        user_id = str(row[0])
        chosen.append(user_id)
        shoppers.append({"label": f"Shopper {number}", "theme": theme, "user_id": user_id})
    return shoppers


def main(argv: list[str] | None = None) -> int:
    """Select the shoppers and write the fixture the demo page serves."""
    parser = argparse.ArgumentParser(description="Build the demo shopper fixture.")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, choices=sorted(CORPORA))
    args = parser.parse_args(argv)
    processed = settings.processed(get_corpus(args.corpus).name)
    connection = duckdb.connect(database=":memory:")
    for relation, path in (
        ("train", processed / "interactions" / "train.parquet"),
        ("items", processed / "items" / "items.parquet"),
    ):
        connection.execute(
            f"CREATE VIEW {relation} AS SELECT * FROM read_parquet('{path.as_posix()}')"  # noqa: S608
        )
    selected = select(connection, settings.seed)
    connection.close()
    users = load_serving_encoding(settings.artifact_root, args.corpus).user_to_code
    shoppers = [
        {"label": s["label"], "theme": s["theme"], "user_code": users[str(s["user_id"])]}
        for s in selected
    ]
    payload = {
        "rule": "scripts/build_demo_fixture.py",
        "corpus": args.corpus,
        "seed": settings.seed,
        "shoppers": shoppers,
    }
    FIXTURE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {FIXTURE_PATH} with {len(shoppers)} shoppers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
