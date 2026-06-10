# Project Plan

This plan reflects the current implementation direction: a PoUW CAPTCHA built
around dense model shard execution, cheap server-side verification, and a
human-verified golden dataset.

## Current Baseline

- Model: `mnist-tiny`
- Architecture: dense `784 -> 128 -> 64 -> 10`
- Task type: MNIST image classification
- Client work: assigned dense layer segment
- Server verification: commitments, secret projections, and spot audits
- Pipeline: multiple CAPTCHA sessions can contribute consecutive segments to a
  single inference run

## Core Goals

1. Keep the browser challenge useful: every solve should contribute verified ML
   inference work.
2. Keep verification cheaper than computation: dense layer verification should
   avoid routine full recomputation.
3. Preserve user experience: normal-risk sessions receive small layer segments.
4. Use human verification selectively to improve the golden dataset.
5. Keep model rotation safe with checksum-pinned in-flight tasks.

## Near-Term Work

- Harden site key and domain validation.
- Add clearer admin visibility for pipeline runs, verification outcomes, and
  golden dataset consensus.
- Improve production configuration defaults for secrets, Redis, CORS, and rate
  limits.
- Expand tests around full API flows, not only verifier internals.
- Document deployment with Redis and a production database.

## Longer-Term Work

- Support additional dense models through the existing manifest-plus-weights
  model store.
- Add model lifecycle tooling for retraining, publishing, and rollback.
- Improve client performance instrumentation across browsers.
- Add stronger abuse monitoring around repeated failed proofs and anomalous
  timing.

## Architecture References

- [docs/CORE_WORKFLOW.md](docs/CORE_WORKFLOW.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/ARCHITECTURE_SIMPLE.md](docs/ARCHITECTURE_SIMPLE.md)
