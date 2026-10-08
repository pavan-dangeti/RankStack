# ADR-0006 — One free-tier container serves both the demo UI and the API

- Status: Accepted
- Date: 2026-10-08

## Context

Phase 6 requires a live demo on free hosting that shows a chosen user's history, their
recommendations with explanations, a model switch, and a cold-start user.

Free tiers available (limits verified 2026-10-08):

| Option | Shape | Catch |
|---|---|---|
| Hugging Face Space, CPU Basic | 2 vCPU, 16 GB RAM, 50 GB non-persistent disk | Docs note creating a Space that *runs* compute requires a paid plan while the pricing page lists CPU Basic as free; free accounts get 8 concurrent CPU runtime units; sleeps after 48 h idle |
| Render free web service | 512 MB RAM | Spins down after 15 min idle, ~1 min cold start, 750 instance-hours/month, 512 MB cannot hold a float32 index |
| Netlify / Vercel | static hosting, free | Static only — cannot serve a model or an index |
| Oracle Cloud Always Free VM | up to 4 OCPUs / 24 GB | Capacity is scarce; instances are frequently unavailable in a given region |

## Decision

1. **One Docker image, one Hugging Face Space (CPU Basic)** serving both the Gradio demo UI and the
   FastAPI API from the same process, with the index loaded once at startup.
2. **The serving index is PQ-compressed** so that it fits the smallest plausible memory ceiling.
   At 1.6M items × 128 dims, float32 is ~819 MB; IVF-PQ at 32 bytes per vector is ~51 MB. PQ is
   chosen on measured recall-versus-latency evidence from Phase 3, not on memory alone.
3. The Space serves a **static shell** so that a sleeping or restarting Space shows a page that
   explains the delay instead of a browser error.
4. Fallback plan if HF's paid-plan requirement blocks the free tier: a **static Space** (free for
   everyone, no compute) calling a **Render free API**, accepting the 15-minute spin-down. The
   latency SLOs in `docs/design.md` §3.3 are stated for the 4 vCPU shape and are measured
   separately from the free-hosting shape; the README reports both and does not present free-tier
   cold-start behaviour as a latency result.

## Consequences

**Good.** One deployment, one image, one place to look. 16 GB of RAM is enough for the item
embedding matrix, the PQ index and the ranker with room to spare. No cross-service networking to
instrument. Free.

**Bad.** Cold start on first visit after sleep. The Space is a single point of failure for the demo.
Coupling the demo and the API means a UI bug can take down the API.

**Risk.** HF may change free-tier policy, in which case the demo URL breaks. Mitigated by the
fallback plan, by verifying the Space is live immediately before the `v1.0.0` tag, and by the
README stating the expected wake-up behaviour.

## Alternatives considered

- **Render API + static UI on Netlify** — rejected: 512 MB and a 15-minute spin-down make a poor
  first impression, and it splits one service across two providers.
- **Oracle Always Free VM** — rejected as the primary: 24 GB of RAM would solve every memory
  problem, but instance availability in most regions is unreliable and it needs manual lifecycle
  management. Noted as a known alternative.
- **GitHub Pages + client-side inference** — rejected: shipping model weights and index files to a
  browser would expose artefacts and would not measure real serving latency.
- **ZeroGPU Space** — rejected: GPU runtime is for inference-heavy demo use; this system needs CPU
  and memory, not GPU, at serving time.