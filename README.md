# RankStack

A production-style recommendation system: given a user, it returns a ranked list of products they
are likely to want next, with a short explanation for each item and the model version that produced
it. Retrieval narrows a ~1.6M-item catalogue to a few hundred candidates, a learned ranker orders
them, and business rules enforce diversity and freshness. Evaluation is time-respecting: every
split, feature and metric is defined against a global timeline, and leakage is tested automatically.

**Status: Phase 1 — data foundation.** The system design is written up
([`docs/design.md`](docs/design.md)) and the data pipeline is implemented and tested: verified
download, schema validation that fails the build, the global-timeline split, point-in-time feature
snapshots, and a leakage suite with a negative control. Retrieval, ranking and serving land in later
phases; nothing here recommends anything yet.

- [Design document](docs/design.md) — requirements, SLOs, architecture, alternatives, risks, threat model
- [Architecture decisions](docs/adr/) — the decisions worth arguing about, and what was rejected
- [Data contracts](docs/schemas.md) — the Parquet schemas every later phase reads
- Data: Amazon Reviews 2023 (McAuley Lab), Electronics. Terms analysis and mitigations in
  `docs/design.md` §4.3.

## What the pipeline does

```bash
uv sync --all-groups                                     # locked environment, Python 3.12
uv run python scripts/download_data.py                   # fetch + verify raw files (~2.3 GB)
uv run python scripts/build_dataset.py                   # validate, split, write Parquet + manifest
uv run pytest --cov=rankstack --cov-fail-under=85        # unit, property, integration, e2e
```

Raw dataset files are never committed (`data/` is gitignored, and CI fails if one is tracked). Every
build writes `data/processed/{corpus}/manifest.json` with source hashes, partition row counts,
validation outcomes, output hashes, seed, tool versions and platform, so a run can be repeated and
audited.

## Measured dataset statistics

Amazon Reviews 2023, Electronics, 0-core. Every figure below comes from
`results/dataset_electronics_0core.json`, which CI re-reads and compares against these numbers
(`scripts/check_docs_drift.py`).

| Quantity | Value | Source |
|---|---:|---|
| Interactions after validation | 43,365,424 | `results/dataset_electronics_0core.json#/totals/interactions` |
| Distinct users | 18,286,190 | `results/dataset_electronics_0core.json#/totals/users` |
| Distinct items with metadata | 1,610,012 | `results/dataset_electronics_0core.json#/totals/items_with_metadata` |
| Train rows (`timestamp < t1`) | 35,335,971 | `results/dataset_electronics_0core.json#/partitions/train/rows` |
| Validation rows (`t1 <= ts < t2`) | 3,953,788 | `results/dataset_electronics_0core.json#/partitions/valid/rows` |
| Test rows (`t2 <= ts`) | 4,075,665 | `results/dataset_electronics_0core.json#/partitions/test/rows` |
| Items present in train | 1,334,914 | `results/dataset_electronics_0core.json#/partitions/train/items` |
| Items first seen after `t1` | 274,946 | `results/dataset_electronics_0core.json#/cold_start/new_items` |
| Users first seen after `t1` | 2,845,057 | `results/dataset_electronics_0core.json#/cold_start/new_users` |
| Rows dropped by validation | 2 | `results/dataset_electronics_0core.json#/validation/rows_dropped` |
| Build wall-clock | 125 s | `results/dataset_electronics_0core.json#/elapsed_seconds` |

Reproduce with `uv run python scripts/build_dataset.py --corpus electronics_0core` on the platform
recorded in the result file's `environment` block. Download is a separate step
(`scripts/download_data.py`); both verify file sizes and record SHA-256 hashes. The build time above
is a warm run: the first run on a cold file cache, parsing the 1.31 GB metadata file, took 233 s on the
same machine.

### What validation found in the published data

Three findings, all recorded in the result file rather than smoothed over:

- **2 interactions rated `0.0`**, which is outside the documented 1–5 scale. Dropped and counted
  (`validation.checks[rating_out_of_scale]`). Two rows out of 43.4M cannot move any metric.
- **`price` is heterogeneous**: 526,449 numeric, 1,083,247 null, 316 strings — 266 of which are the
  em-dash placeholder `—` and the rest look like `from 9.95`. The reader takes the first number in a
  string price and never coerces a missing price to zero, because "unknown price" and "free" are
  different facts for a price-affinity feature.
- **128,442 items have no category path** and 106,334 have no `main_category`. Category affinity
  treats those as unknown rather than as a bucket of their own.

The pipeline also cross-checks its split against the publisher's published split sizes, which it
re-derives rather than trusting: 35,335,971 measured against 35,300,000 published for train,
3,953,788 against 4,000,000 for validation, 4,075,665 against 4,100,000 for test — all inside the 2%
tolerance that three-significant-figure rounding requires.