# PoUW CAPTCHA System - Current Status

## Status

The project currently implements a working PoUW CAPTCHA demo around a real
trained MNIST model and a distributed dense-layer inference pipeline.

## What Is Implemented

- FastAPI backend with CAPTCHA init, submit, verify, validate, demo, dashboard,
  and pipeline inspection endpoints.
- Browser widget that executes assigned dense model shards.
- `mnist-tiny` model with checksum-pinned dense layers.
- Server-side proof verification using commitments, secret projections, and
  probabilistic spot audits.
- Distributed pipeline runs where multiple solvers can piece together one full
  inference.
- Optional human verification that feeds the golden dataset flow.
- E2E client simulator for exercising the live system.

## Current Model

```text
models/mnist-tiny/
├── manifest.json
└── weights.npz
```

The active model is a dense MNIST classifier:

```text
784 -> 128 -> 64 -> 10
```

The manifest includes per-layer SHA-256 checksums and a model checksum. Browser
clients verify shard integrity before computation.

## Local Endpoints

When the backend is running on port `8000`:

- Health: `http://localhost:8000/health`
- Readiness: `http://localhost:8000/ready`
- API docs: `http://localhost:8000/docs`
- Demo: `http://localhost:8000/demo`
- Dashboard: `http://localhost:8000/dashboard`
- Pipeline runs: `http://localhost:8000/api/v1/pipeline/runs`

## Quick Checks

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
curl http://localhost:8000/api/v1/pipeline/runs
```

Run the solver simulator:

```bash
cd server
.venv\Scripts\python ..\scripts\e2e_client.py --solves 12
```

## Production Gaps To Treat Carefully

- The current model is intentionally small for a capstone/demo environment.
- Redis should be available for verification request storage and risk tracking.
- Production deployment should use strong secrets, constrained CORS, real site
  key/domain validation, rate limits, monitoring, and hardened replay controls.
- The proof system gives strong probabilistic assurance, not mathematical
  impossibility.
