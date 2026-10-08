# ADR-0005 — Online feature store is embedded by default, with a Redis adapter behind one interface

- Status: Accepted
- Date: 2026-10-08

## Context

Serving needs per-user and per-item features within the 8 ms feature budget. The brief allows
"Redis or an embedded store" and requires point-in-time correctness.

The scale makes the tradeoff easy to see. Electronics 0-core has 1.6M items and 18.3M users, but the
serving workload touches a few hundred users at a time, and the features that matter are small
vectors and scalars: ALS factors, the last-20 item IDs, category affinity, and ~25 scalar ranking
features. Per item that is a few hundred bytes; the item side is a few hundred megabytes as
float32, tens of megabytes as PQ-compressed vectors.

The free-tier option is Upstash Redis: 256 MB and **500K commands per month** (verified
2026-10-08). A demo that issued even a handful of feature reads per request would spend that budget
in days. That makes Redis the wrong default and the right demonstration.

## Decision

1. **Online store is embedded**: the serving container builds per-user and per-item feature arrays
   at load time and serves them from memory. It is the default and the primary path.
2. A **Redis adapter** implements the same interface for the distributed case. It is exercised by an
   integration test against a real Redis container, and by the adapter test against Upstash when
   credentials are present. It is not on the demo's critical path.
3. **One feature definition registry** generates both the offline builder and the online snapshot
   from the same source of truth, so training/serving skew is a type error rather than a silent
   divergence. Skew is additionally detected at load time by recomputing a random sample of online
   features offline and asserting equality within tolerance.
4. **Offline features are point-in-time correct by construction**: every feature snapshot carries an
   `as_of` timestamp, and the training builder selects the latest snapshot with
   `as_of <= prediction_time`. A feature that cannot be expressed with an `as_of` is not allowed
   into the registry.
5. `feature_age_seconds` is both a Prometheus gauge and a served feature. Data older than 24 hours
   raises an alert and switches the model to its degraded mode.

## Consequences

**Good.** No network hop in the latency budget, no quota ceiling, deterministic and reproducible
serving behaviour, and a working Redis path to demonstrate the distributed design.

**Bad.** State is per-container, so a multi-replica deployment needs the Redis path (or a shared
volume), and a rebuild of features requires a reload rather than a key write. Two implementations
of one interface is more code than one implementation.

**Risk.** Reload of the embedded snapshot takes memory proportional to catalogue size; measured and
reported in Phase 5, with PQ compression as the lever if it does not fit the free hosting shape.

## Alternatives considered

- **Upstash Redis as the primary store** — rejected: the 500K commands/month free tier cannot
  support a demo, and a demo that dies on quota is worse than one that is fast.
- **Self-hosted Redis in the serving container** — rejected: adds an operational dependency for no
  gain at this scale, and it is still an in-process store with the same failure modes.
- **Recompute features per request from DuckDB** — rejected: DuckDB is excellent for the offline
  pipeline and wrong for an 8 ms online budget; it also invites accidental non-point-in-time reads.
- **Feature store vendor (Feast)** — rejected: substantial operational weight for a project whose
  entire point is that it costs $0.