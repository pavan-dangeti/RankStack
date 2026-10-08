# ADR-0002 — Corpus selection: Amazon Reviews 2023, Electronics, 0-core primary

- Status: Accepted
- Date: 2026-10-08

## Context

The dataset choice determines what can be measured. MovieLens-25M is the default in recommender
tutorials and is the least informative choice for a portfolio project: ~62K items and no product
text, price or category hierarchy, so ANN retrieval is a `np.argsort` and content-based cold start
is impossible.

Amazon Reviews 2023 (McAuley Lab) has 571.54M reviews, 54.51M users, 48.19M items across 33
categories, millisecond timestamps, and item metadata (title, description, price, hierarchical
categories). It also ships k-core (0-core and 5-core) and two benchmark split families, and it is
the dataset used by a substantial body of published work, which makes external comparison possible.

Two facts constrain the choice:

1. **The 5-core filter deletes the cold-start population by construction.** An item with fewer
   than 5 interactions cannot appear in a 5-core corpus. The publisher documents this trade-off
   ("Limited diversity, misalignment with the original data distribution") and still recommends
   5-core for benchmarking.
2. **Terms of use are not permissive.** The publisher makes the data available for research and
   states they cannot assign a licence; Amazon's Conditions of Use grant only personal,
   non-commercial use and exclude derivative use and data-mining tools (analysis in
   `docs/design.md` §4.3).

## Decision

1. **Primary corpus: Amazon Reviews 2023, Electronics, 0-core** — 18.3M users, 1.6M items, 43.4M
   interactions (published statistics, to be re-derived and asserted in Phase 1).
2. **Comparability corpus: Electronics, 5-core** — 1.6M users, 368.2K items, 15.5M interactions —
   used only to score under the publisher's benchmark protocol for comparison with published work.
3. **Transfer check: Musical Instruments, 5-core** — 57.4K users, 24.6K items, 511.8K interactions,
   8.4 MB download — a second domain trainable in minutes on the development Mac.
4. The corpus is selected by configuration, not by code. The pipeline reads one canonical schema
   (`docs/schemas.md`), so swapping datasets is a config change. Yelp Open Dataset (CC BY 4.0) is
   the designated permissively licensed fallback.
5. Raw dataset files are never committed. They are downloaded at runtime from the publisher's
   published URLs by a committed script, and CI fails if any large file appears under `data/`.

## Consequences

**Good.** 1.6M items makes ANN index construction and the recall/latency trade-off a real
experiment rather than a formality. 0-core preserves sparse items, so cold-start-item evaluation is
possible and meaningful. Price, category and text metadata exist, which is what makes the ranking
features and content path real. Published protocol and published split sizes allow a genuine
external comparison.

**Bad.** 970 MB compressed / ~2.5 GB uncompressed per interactions file plus 1.31 GB of metadata,
processed on a 16 GB laptop — memory and disk discipline are required. 0-core is sparse (about 2.4
interactions per user on average), so collaborative filtering is weaker than on 5-core and
cold-start users are the majority, which changes what the headline metrics mean. Free GPU time is
the constraint on the two-tower run, not the Mac.

**Risk.** The terms question. This is mitigated by non-commercial use, no redistribution,
enforced in CI, and a dataset-agnostic pipeline, but it cannot be eliminated. It is the first item
in the design doc's open questions.

## Alternatives considered

- **Home_and_Kitchen (66.6M interactions, 3.7M items)** — rejected: catalogue-dominated with long
  replacement cycles, so new-item cold start is rare.
- **5-core as the primary corpus** — rejected as primary: removes the cold-start population, which
  is a stated requirement.
- **Yelp Open Dataset (CC BY 4.0)** — retained as the licensed fallback; lacks price metadata and
  has a smaller item count.
- **MovieLens-25M** — rejected: the cliché, and it cannot support the required features.