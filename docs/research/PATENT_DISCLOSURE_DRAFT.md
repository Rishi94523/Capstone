# Patent Disclosure Draft

This is an engineering disclosure draft, not legal advice or a patent filing.

## Invention Name

Verifiable Useful-Work CAPTCHA and Rate-Limiting System

## Problem

Traditional CAPTCHA systems rely on puzzles that are increasingly solvable by
AI systems and CAPTCHA-solving farms. Conventional proof-of-work rate limiting
can impose cost on bots, but the computation is usually wasted.

## Core Idea

Require browser clients to perform assigned useful ML inference work as the
access toll. Verify that work server-side using task-bound probabilistic proofs.
Use completed work and selective human verification to produce useful machine
labels and golden dataset entries.

## System Components

- Site registration with public site keys and private validation secrets.
- Browser widget that requests a challenge and computes assigned model shards.
- Risk scorer that adjusts workload based on request frequency, proof failures,
  behavioral signals, timing, and reputation.
- Distributed inference pipeline that hands verified activations between
  independent CAPTCHA sessions.
- Proof verifier using vector commitments, secret projection checks, and spot
  audits.
- Human verification flow for completed predictions.
- Golden dataset and economics metrics pipeline.
- One-time token validation for server-to-server form submission checks.

## Potentially Novel Combination

The projection equation itself is prior art-adjacent (Freivalds-style
verification). The inventive combination is the end-to-end use of verified
browser ML shards as a CAPTCHA/rate-limit primitive that converts automation
pressure into useful inference and labeling work, with a verifier that is
generic over affine layer operators: dense layers verify via `s = W·r`,
convolutional layers via `s = conv_transpose(r, kernels)` — both precomputed
once per model version — so heterogeneous architectures (MLPs, CNNs,
attention/matmul blocks) plug into the same distributed pipeline without
changing the verification protocol.

## Important Variants To Claim

- Affine-operator-generic verification: dense, conv2d (transposed-convolution
  projections), attention/matmul projections, embeddings-as-matmul.
- Server-replayed post-op chains (activation, pooling, flatten) between
  provable layers, preserving chain of custody at O(n) cost.
- Multi-architecture model store with per-run model rotation; runs are
  version-isolated so retraining never corrupts in-flight pipelines.
- Different model types and layer sizes.
- Risk-based shard sizing.
- Segment handoff through verified activation storage.
- Per-layer checksum-pinned shard delivery (wire-byte checksums valid for any
  layer type's canonical flattening).
- Proof binding to task id, sample id, model checksum, and segment index.
- One-time token validation with private site secrets.
- Human verification triggering based on risk tier or model completion.
- Human-feedback retraining loop: golden-label consensus → fine-tune →
  version bump rotates checksums and projections.
- Economic dashboard measuring bot work converted into useful value.

## Evidence And Prototype

- Working FastAPI backend.
- Browser TypeScript widget and SDK with dense and conv2d execution.
- `mnist-tiny` (dense MLP, 98.09%) and `mnist-cnn` (conv net, 98.46%) trained
  models with real layer checksums, both served through the same pipeline.
- Local multi-architecture evaluator in `scripts/evaluate_pouw.py`.
- Latest evidence report in `docs/evaluation/latest.md`.

## Public Disclosure Caution

File a provisional patent application before public paper submission or broad
commercial disclosure if patent protection is a priority.
