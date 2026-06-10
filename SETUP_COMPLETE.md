# Setup Notes

This project is set up as a local PoUW CAPTCHA demo with a real `mnist-tiny`
model and browser-side dense shard execution.

## Components

| Component | Status |
| --- | --- |
| FastAPI backend | Implemented |
| SQLite local database | Default local setup |
| Redis integration | Used for verification/risk state |
| Browser widget | Implemented |
| SDK wrappers | Vanilla, React, and Vue entry points |
| Model store | Loads checksum-pinned manifests from `models/` |
| Proof verifier | Commitment, projection, and audit checks |
| Golden dataset flow | Implemented service path |

## Recommended Local Setup

```bash
npm install

cd server
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Seed samples:

```bash
cd server
.venv\Scripts\python ..\scripts\seed_data.py --count 200
```

Run the backend:

```bash
cd server
.venv\Scripts\python -m uvicorn app.main:app --reload --port 8000
```

Run the widget dev server:

```bash
npm run dev --workspace=packages/widget
```

Exercise the full flow:

```bash
cd server
.venv\Scripts\python ..\scripts\e2e_client.py --solves 12
```

## Current Demo URLs

- Backend: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- Demo: `http://localhost:8000/demo`
- Dashboard: `http://localhost:8000/dashboard`

## Notes

Older roadmap material described alternative datasets and runtimes. The current
checked-in implementation is the dense MNIST shard pipeline described in
[docs/CORE_WORKFLOW.md](docs/CORE_WORKFLOW.md).
