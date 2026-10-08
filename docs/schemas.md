# Data contracts

Everything downstream of `src/rankstack/data/build.py` reads Parquet, never the publisher's files.
This page is the contract. The declarations in `src/rankstack/data/schema.py` are the executable
version; if the two ever disagree, the code wins and this page is the bug.

## Inputs

### Interactions — `benchmark/{0,5}core/rating_only/{Category}.csv.gz`

| Column | Type | Notes |
|---|---|---|
| `user_id` | string | opaque, pseudonymous. Never logged, never returned. |
| `parent_asin` | string | product identifier |
| `rating` | float | 1.0–5.0 |
| `timestamp` | int64 | epoch **milliseconds** |

### Item metadata — `raw/meta_categories/meta_{Category}.jsonl.gz`

| Column | Type | Notes |
|---|---|---|
| `parent_asin` | string | join key to interactions |
| `title` | string | falls back to `main_category`, then to `parent_asin` |
| `main_category` | string | publisher's domain label |
| `categories` | string[] | joined with `" > "` into `category_path` |
| `price` | double | nullable; never coerced to zero |
| `store` | string | nullable |
| `description` | string[] | joined and truncated to 512 characters |

Descriptions are truncated because item text only ever feeds a local embedding model
(`docs/adr/0004-local-text-embeddings.md`); the full text is not needed and keeping the corpus small
keeps the embedding step fast.

## Outputs

Written to `data/processed/{corpus}/` and fingerprinted in `manifest.json`.

### `interactions/{train,valid,test}.parquet`

| Column | Type | Notes |
|---|---|---|
| `user_id` | string | |
| `parent_asin` | string | |
| `rating` | float32 | narrowed from float64; the scale has 21 distinct values |
| `timestamp` | int64 | authoritative |
| `event_time` | timestamp[us, UTC] | derived, for readability only |

Rows are ordered by `(timestamp, user_id, parent_asin)` so the Parquet bytes are reproducible and the
manifest hash is a meaningful fingerprint.

Partition rule — a global timeline, never a random split:

| Partition | Window | Electronics 0-core published rows |
|---|---|---:|
| `train` | `timestamp < t1` | 35,300,000 |
| `valid` | `t1 <= timestamp < t2` | 4,000,000 |
| `test` | `t2 <= timestamp` | 4,100,000 |

`t1 = 1628643414042` (2021-08-11T00:56:54.042Z) and `t2 = 1658002729837` (2022-07-16T20:18:49.837Z)
are the publisher's published boundaries, so results are comparable with external work.

### `items/items.parquet`

`parent_asin`, `title`, `main_category`, `category_path`, `price`, `store`, `description` — one row
per item, ordered by `parent_asin`.

### `features/item_as_of_features.parquet`

Long format, one row per (entity, feature, snapshot):

| Column | Type | Notes |
|---|---|---|
| `entity_type` | string | `item` at this stage |
| `entity_id` | string | `parent_asin` |
| `feature_name` | string | see below |
| `as_of_ms` | int64 | the snapshot boundary |
| `value_double` | double | |

| Feature | Definition |
|---|---|
| `item_interactions` | count of interactions with `timestamp < as_of_ms` |
| `item_days_since_first_interaction` | `(as_of_ms - min(timestamp)) / 86_400_000` over the same window |
| `item_days_since_last_interaction` | `(as_of_ms - max(timestamp)) / 86_400_000` over the same window |

Read rule: **latest snapshot with `as_of_ms <= prediction_time`.** An entity with no in-time snapshot
has no features, which is the cold-start case rather than an error — `rankstack.data.asof.read_as_of`
returns nothing for it.

The `<` is strict in both the feature SQL and its verification query, so an interaction at exactly
the boundary is excluded on both sides and the recomputation matches exactly.

## Validation

| Check | Kind | Failure |
|---|---|---|
| `null_user_id`, `null_parent_asin` | hard fail | build stops |
| `null_item_parent_asin` | hard fail | build stops |
| `rating_out_of_range` | hard fail | build stops |
| `timestamp_out_of_range` | hard fail | build stops |
| `interaction_count` | hard fail | build stops unless within 2% of the published count |
| `item_metadata_coverage` | hard fail | build stops below 90% of interacted items covered |
| `duplicate_events` | drop | one copy kept, count recorded |
| `duplicate_item_metadata` | drop | first occurrence kept, count recorded |
| `items_without_title`, `items_without_price` | report | recorded only |

Published counts are rounded to three significant figures, hence the 2% tolerance. The counts are
re-derived from the downloaded file, never read from the publisher's table.

## Known limitation at this stage

Snapshots exist only at `t1` and `t2`. An item whose first interaction is *after* `t2` therefore has
no collaborative features at any boundary — which is correct, and is exactly the cold-start
population the content path in Phase 4 exists to serve. User-level snapshots arrive with the
baselines that need them; the read rule is identical.