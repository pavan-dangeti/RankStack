# Serving runbook

Operating the recommendation API: what it serves, how it fails, what each alert means, and what to
do about it. Every latency number here comes from `results/load_test_electronics_0core.json` and
every quality number from a committed result file; the commands that produced them are given.

## What is actually being served

The request path serves **reciprocal-rank fusion of `popularity_recent` and item-item
co-occurrence**, over a 1,334,914-item catalogue.

That is not the most accurate configuration available and it is deliberate. Measured on the warm
cohort (`results/fusion_electronics_0core.json`):

| Source | HR@20 | Distinct top-1 items across 2,000 users |
|---|---|---|
| `popularity_recent` alone | 0.0200 | 3 |
| `cooccurrence` alone | 0.0015 | 1,911 |
| **fused (shipped)** | **0.0150** | **18** |

`popularity_recent` is the strongest single retriever on this corpus, and it is also a *global*
ranking: it returned only 3 distinct top items across 2,000 warm users. Serving it alone would hand
nearly every visitor the same list. Fusion gives up 25% of HR@20 and buys six times the
personalisation. If this service were measured on hit rate alone, the right answer would be to ship
`popularity_recent` alone; the choice here is documented rather than hidden.

The learned ranker and the two-tower model are **not** in the request path. Both were measured and
both lost (`results/two_stage_electronics_0core.json`, `results/two_tower_diagnosis_electronics_0core.json`).

## Start it

```bash
uv run python scripts/build_serving_manifest.py          # hash the artefacts
docker build -f infra/docker/Dockerfile.serve -t rankstack-serve .

docker run --rm -p 8000:8000 \
  -v "$PWD/data/processed:/data:ro" \
  -v "$PWD/artifacts:/artifacts:ro" \
  rankstack-serve
```

Artefacts are mounted read-only and verified against `serving_manifest.json` before the first
request. A hash mismatch **fails startup**. That is the intended behaviour: a serving process that
cannot vouch for its model should not answer.

First load takes about 9 s on the development machine and 683 MB resident (1.6M product titles).
For a 512 MB free tier, build the catalogue with `with_titles=False`; responses then carry
`title: null` and nothing else changes.

```bash
curl 'http://localhost:8000/recommend?user_id=AE2222C2STTMXNUHXTVDF5&k=5'
```

## The contract

`GET /recommend?user_id=<id>&k=<1..50>` always returns **HTTP 200 with a non-empty list**. Unknown
users, a corrupt index and a missing feature store all degrade; none of them produce a 404, a 500
or an empty page. `k` is clamped to 1..50 rather than rejected.

Every response carries `request_id`, `model_version`, `index_version` (the artefact checksum) and
`fallback_used`. Explanations come from a fixed template set keyed on the dominating signal; no text
is generated at request time (ADR-0003), so every explanation is traceable to a score.

| Endpoint | Purpose |
|---|---|
| `GET /recommend` | The recommendation path. |
| `GET /healthz` | Liveness. Artefact state, catalogue size, no model call. |
| `GET /metrics` | Prometheus exposition. |

## Latency

From `results/load_test_electronics_0core.json`, 400 requests at concurrency 8, k=20, single
process on the development machine:

| Path | Target | Measured p50 | p95 | p99 |
|---|---|---|---|---|
| Warm user | 60 / 150 / 300 ms | 25.1 ms | 33.0 ms | 59.7 ms |
| Cold start | p95 ≤ 250 ms | 24.1 ms | 29.7 ms | 30.8 ms |
| Degraded (index raising) | p95 ≤ 20 ms | 8.7 ms | 13.9 ms | 15.8 ms |

Throughput 310 req/s in one process. All five SLOs in the design document are met.

This measures `Recommender.recommend`, the path the ASGI layer calls directly, not the HTTP
round trip. The design allocates 10 ms separately for serialisation, logging and tracing.

```bash
uv run python scripts/load_test.py --requests 400 --concurrency 8
```

## Degradation

| Symptom in metrics | Cause | What happens | Action |
|---|---|---|---|
| `fallback_total{reason="cold_start"}` climbing | Users unseen at training time | Global ranking, personalised as far as the catalogue allows | Expected: 57% of test users have no train history. Investigate only if it jumps past that. |
| `fallback_total{reason="index"}` non-zero | Served source raised | Falls back to global popularity | Check artefact disk and the manifest. If it persists, redeploy. |
| `fallback_total{reason="retriever"}` non-zero | Ranking itself raised | Global popularity | A bug. Check logs for `serving.retrieval_failed`. |
| Process exits at startup, `does not match the serving manifest` | Artefact changed after hashing | Refuses to start | Rebuild the manifest and redeploy **together**. Never hash separately from deploying. |
| `recommend_duration_seconds` p95 > 150 ms | Retriever slowed | No automatic shedding | Scale out. The path is CPU-bound on a 1.3M-item score vector. |
| `recommend_requests_total` flat while traffic rises | Wrong port, or health check only | — | Check routing first. |

`GET /metrics` counters are per-process and reset on restart. That is inherent to Prometheus
counters and not a bug in this service.

## Logs

One structured JSON line per request, with `request_id` matching the `X-Request-Id` response header.
Degradations log at `error` with a traceback; everything else logs at `info`. Request ids are not
carried between processes — there is only one process in the current deployment shape.

## Common tasks

**Rebuild artefacts and restart.** Artefacts are not baked into the image, so the image is unchanged.

```bash
uv run python scripts/train_baselines.py
uv run python scripts/build_serving_manifest.py
docker restart <container>
```

**Prove the degradation paths still work.** Every row above has a test; run them.

```bash
uv run pytest tests/unit/test_serving.py -q
```

**Check a response before shipping.** The contract test asserts HTTP 200, a non-empty list and a
matching request id. To eyeball a payload:

```bash
curl -s 'http://localhost:8000/recommend?user_id=NO_SUCH_USER&k=2' | python3 -m json.tool
```

## Known limits

- **OpenTelemetry export is not wired.** The request path emits in-process spans only. There is
  single-process deployment shape here, so there is no process boundary to export across yet.
- **The serving image still carries torch.** torch is a main dependency of the package, so the image
  is larger than the request path needs. Making it an optional extra would shrink it by roughly 2 GB
  and is not done.
- **No load shedding.** The `shed_requests_total` counter exists and is always zero. There is no
  concurrency ceiling on the request path, because measured p95 leaves enough headroom that
  shedding would be untested speculation.
- **The catalogue is loaded, not memory-mapped.** `np.load` without `mmap_mode`, because the arrays
  are read at full length on every request anyway.