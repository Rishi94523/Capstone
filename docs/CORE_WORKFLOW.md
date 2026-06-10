# Core Workflow: Distributed PoUW CAPTCHA

End-to-end description of the production core built around a **real trained
model**, **verifiable computation**, and a **distributed labeling pipeline**.

## 1. The model (plug-and-play)

Models live in `models/<name>/`:

```
models/mnist-tiny/
├── manifest.json   # layers, labels, preprocessing, REAL SHA-256 checksums
└── weights.npz     # trained float32 weights
```

- `mnist-tiny` is a 784→128→64→10 MLP trained on real MNIST in pure NumPy
  (`scripts/train_mnist_numpy.py`), ~97.5% test accuracy in ~15s of training.
- **Checksums are real and layered**: each layer's checksum is SHA-256 over
  the exact float32 bytes a browser receives; the model checksum is a hash of
  the layer checksums. A client holding only a *segment* of the model can
  therefore still verify integrity (`crypto.subtle.digest` over the
  `Float32Array` bytes) before executing anything.
- Adding a new dataset/model = dropping a new manifest + weights directory
  into `models/`. The server (`app/ml/model_store.py`) loads and
  integrity-verifies everything at startup. No code changes needed for new
  dense models.

## 2. Distributed inference pipeline (the "piecing together")

Each unlabeled sample flows through a **pipeline run** (`pipeline_runs`
table). Solvers compute *segments* of layers, sized by risk tier:

| Risk tier  | Layers per CAPTCHA | Rationale                       |
|------------|--------------------|---------------------------------|
| normal     | 1                  | fastest UX, few hundred ms      |
| suspicious | 2                  | more work for risky traffic     |
| bot_like   | all remaining      | maximum cost for bots           |

The server stores the verified **post-activation** of each completed segment
and hands it to the next solver as their input. When the final layer
completes, the softmax output becomes the sample's machine label — one
inference pieced together from multiple users' partial computations
(`app/core/pipeline.py`). Claims have a 90s TTL so abandoned segments get
reassigned.

## 3. Verifiable computation (verify WITHOUT recomputing)

The crucial mechanism (`app/ml/proof_verifier.py`). The client submits the
**pre-activation vector** of every computed layer plus commitment hashes.
The server verifies three ways, cheapest first:

1. **Commitment hashes** — each vector is hashed (canonical 4-decimal form)
   and bound with the task id, sample id, and segment position into a single
   proof hash. Proofs can't be replayed across tasks or detached from data.

2. **Freivalds-style secret projections** — for each dense layer
   `z = W·x + b`, the server holds K=4 secret random vectors `r` and the
   precomputed `s = W·r` (once per model load, never per request). It checks

   ```
   r · z  ≈  s · x  +  r · b
   ```

   which costs **O(in + out)** multiplications versus the **O(in × out)**
   the client had to spend. `r` is derived from the server secret key (so
   workers agree and clients can't reconstruct it). A fabricated `z` passes
   4 independent secret projections with negligible probability. The layer
   input `x` is always known server-side: the sample input at segment 0, or
   the previous solver's verified activation afterward. Activations are
   recomputed server-side from submitted pre-activations (elementwise, cheap).

3. **Probabilistic spot audits** — ~8% of submissions get a full segment
   recompute, bounding any adaptive attack on the projection checks.

The asymmetry grows with model size: for the 784×128 layer verification is
already ~27× cheaper than recomputation; for production-scale layers it's
100×+. Tests in `server/tests/test_proof_verifier.py` cover honest clients
(including multi-user piecing), fabricated outputs, single-value tampering,
wrong-input precompute attacks, replays, hash mismatches, and audit catches.

## 4. Human verification → golden dataset → retraining

- When a pipeline run completes, the solver is sometimes asked to verify the
  pieced-together label (always for bot_like, 50% suspicious, ~20% normal).
- Verified labels accumulate; 3-way consensus (reputation-weighted) promotes
  a label into the **golden dataset** (`app/services/golden_dataset.py`).
- `scripts/retrain_from_golden.py` fine-tunes the model on verified labels
  (mixed with replay data to prevent forgetting), bumps the version, and
  regenerates all checksums. In-flight tasks pin the model checksum at
  assignment, so version rotation is race-free; pipeline runs are isolated
  per model version.

## 5. Quality controls

- **Honeypots**: ~10% of seeded samples keep their true label hidden
  (`known_label`); completed runs are checked against it.
- **Timing plausibility**: submissions faster than 10% of the expected
  compute time are rejected.
- **Server-derived labels are authoritative**: the final label comes from
  the verified final-layer output, not from anything the client claims.

## Running it

```bash
# 1. Train the model (downloads MNIST, ~30s total)
server/venv/Scripts/python scripts/train_mnist_numpy.py

# 2. Seed real samples
cd server && venv/Scripts/python ../scripts/seed_data.py --count 200

# 3. Start the server
venv/Scripts/python -m uvicorn app.main:app --port 8000

# 4. Demo at http://localhost:8000/demo, dashboard at /dashboard
#    Pipeline state: GET /api/v1/pipeline/runs

# 5. Simulate solvers (watch runs get pieced together)
venv/Scripts/python ../scripts/e2e_client.py --solves 12

# 6. Retrain from human-verified labels
venv/Scripts/python ../scripts/retrain_from_golden.py --min-verifications 1
```

Measured on this machine: 30 distributed solves → 10 completed runs →
**21/21 pieced-together labels matched the true MNIST labels**; retraining on
human-verified labels took accuracy 97.53% → 98.09% (v2.0.0 → v2.0.1).
