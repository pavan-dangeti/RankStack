# Serving runbook

Operating the recommendation API: what it serves, how it fails, what each alert means, and what to
do about it. Every latency number here comes from `results/load_test_electronics_0core.json` and
every quality number from a committed result file; the commands that produced them are given.

## What is actually being served

For a known user, the request path serves **reciprocal-rank fusion of `popularity_recent` and
item-item co-occurrence**, over a 1,334,914-item catalogue. An unknown user has no history for
co-occurrence to use, so they get `popularity_recent` alone: the same list for every unknown user.

That is not the most accurate configuration available and it is deliberate. Measured on the warm
cohort (`results/fusion_electronics_0core.json`):

| Source | HR@20 | Distinct top-1 items across 2,000 users | Figure |
|---|---|---|---|
| `popularity_recent` alone | 0.0200 | 3 | results/fusion_electronics_0core.json#/sources/popularity_recent/ranking/hr@20 |
| `cooccurrence` alone | 0.0015 | 1,911 | results/fusion_electronics_0core.json#/sources/cooccurrence/ranking/hr@20 |
| **fused (shipped)** | **0.0130** | **29** | results/fusion_electronics_0core.json#/sources/fused/ranking/hr@20 |

`popularity_recent` is the strongest single retriever on this corpus, and it is also a *global*
ranking: it returned only
3 (`results/fusion_electronics_0core.json#/personalisation/distinct_top_1_global`) distinct top items
across 2,000 warm users. Serving it alone would hand nearly every visitor the same list. Fusion
returns
29 (`results/fusion_electronics_0core.json#/personalisation/distinct_top_1_fused`) distinct top items
at a paired HR@20 cost of
-0.007 (`results/fusion_electronics_0core.json#/paired_fused_vs_popularity_recent/hr/hr@20/difference`).
If this service were measured on hit rate alone, the right answer would be to ship
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

A known user, with four train-window purchases (a charger and three sets of headphones):

```bash
curl -s 'http://localhost:8000/recommend?user_id=AE222N3VUKMF3GO6D4LHTELE7UWA&k=5' | python3 -m json.tool
```

The response has `"fallback_used": null`. For this user the fused list holds globally trending items
reordered where co-occurrence links them to the user's history, so its order differs from the
cold-start list below even where the items are the same.

An unknown user takes the cold-start path:

```bash
curl -s 'http://localhost:8000/recommend?user_id=NO_SUCH_USER&k=5' | python3 -m json.tool
```

The response has `"fallback_used": "cold_start"`, and the list is exactly the top of
`popularity_recent`, the same for every unknown user. If two different unknown ids ever return
different lists, or a cold-start list contains an item that is not globally trending, the cold-start
path is reading someone's history: treat it as a privacy incident.

## The contract

`GET /recommend?user_id=<id>&k=<1..50>` always returns **HTTP 200 with a non-empty list**. Unknown
users, a corrupt index and a missing feature store all degrade; none of them produce a 404, a 500
or an empty page. Out-of-range `k` (below 1 or above 50) is rejected with **HTTP 422**: that is a
malformed request from the caller, not a degradation, so it is not answered with a list.

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

| Path | Target | Measured p50 | p95 | p99 | Figure |
|---|---|---|---|---|---|
| Warm user | 60 / 150 / 300 ms | 25.9 ms | 32.1 ms | 58.5 ms | results/load_test_electronics_0core.json#/latency/0/p50_ms |
| Cold start | p95 ≤ 250 ms | 8.6 ms | 11.3 ms | 11.8 ms | results/load_test_electronics_0core.json#/latency/1/p50_ms |
| Degraded (index raising) | p95 ≤ 20 ms | 9.2 ms | 16.5 ms | 16.8 ms | results/load_test_electronics_0core.json#/latency/2/p50_ms |

Throughput 305.7 req/s (`results/load_test_electronics_0core.json#/throughput_requests_per_second_single_process`)
in one process. All five SLOs in the design document are met.

This measures `Recommender.recommend`, the path the ASGI layer calls directly, not the HTTP
round trip. The design allocates 10 ms separately for serialisation, logging and tracing.

```bash
uv run python scripts/load_test.py --requests 400 --concurrency 8
```

## Degradation

| Symptom in metrics | Cause | What happens | Action |
|---|---|---|---|
| `fallback_total{reason="cold_start"}` climbing | Users unseen at training time | `popularity_recent` alone, the same list for every unknown user | Expected: 57% of test users have no train history. Investigate only if it jumps past that. |
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