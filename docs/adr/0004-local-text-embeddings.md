# ADR-0004 — Item text is embedded locally; dataset content never leaves the machine

- Status: Accepted
- Date: 2026-10-08

## Context

The item tower needs a text embedding of each item's title and description, and the cold-start path
depends on it for items with no interaction history. On Electronics 0-core that is ~1.6M items.

Three ways to get those embeddings: a local sentence-transformer, a hosted embedding API, or an LLM
API. Two free API keys are available to this project (`NVIDIA_API_KEY`, `GEMINI_API_KEY`).

The constraint that decides it is not cost, it is the dataset's terms. Amazon Reviews 2023 carries
no explicit licence and Amazon's Conditions of Use do not clearly license redistribution or
derivative use of product listings and descriptions (analysis in `docs/design.md` §4.3). Sending
1.6M product titles and descriptions to a third-party inference API is a disclosure of dataset
content to another party that this project has no licence basis for and no need to make.

## Decision

1. All text embedding is computed locally with a pinned, permissively licensed sentence-transformer
   (~22M parameters, Apache-2.0), on the Mac's MPS/CPU or on a free GPU in Colab. The encoder is
   identified in configuration so it can be swapped.
2. Embeddings are precomputed once per item into float16 arrays under `artifacts/` (gitignored), with
   a content hash of the encoder plus the input text recorded in the artefact manifest.
3. **No dataset content is sent to any external API by any code path**, training, evaluation,
   serving or demo. This is enforced by a test that blocks outbound network access from the
   pipeline test suite, and by the absence of any HTTP client in the pipeline and serving modules.
4. API keys are used, if at all, only for offline analysis whose inputs are aggregate numbers or
   text authored in this repository — never raw catalogue or user strings.

## Consequences

**Good.** No licence question attached to data disclosure. No per-item API cost or rate limit.
Deterministic and reproducible embeddings, so an index rebuild is a function of the corpus hash and
the encoder version. No third party in the critical path.

**Bad.** Precomputing 1.6M embeddings takes real wall-clock time on a laptop (order of an hour on
MPS) and is the longest single offline step. A hosted API would be faster. Embedding quality is
bounded by a 22M-parameter model, so it will be weaker than a large text encoder — which is a
finding to report, not a problem to hide, and the encoder is swappable by configuration.

**Risk.** Wall-clock time for the embedding step, and the free-GPU dependency for the fast path.
Mitigated by making the step resumable per shard and by treating the Mac CPU path as the guaranteed
one.

## Alternatives considered

- **Hosted embedding API (Gemini, NVIDIA)** — rejected: discloses dataset content, adds quota and
  cost, and makes the artefact hash depend on a provider's model version.
- **Large local encoder (1B+ parameters)** — rejected for v1: not needed to make the content path
  work, too slow to embed 1.6M items on free hardware, and the value of the comparison is better
  demonstrated as an ablation on a subsample.
- **Bag-of-words / TF-IDF item vectors** — rejected as the only representation: cheap and a useful
  baseline, but it will not make new items reachable through semantics. Kept as a comparison point
  in Phase 3.