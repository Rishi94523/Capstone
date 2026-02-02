# PoUW CAPTCHA System - Implementation Plan

> **Proof-of-Useful-Work CAPTCHA**: A web-based CAPTCHA system that replaces traditional puzzle-based CAPTCHAs with productive machine learning computation executed inside the user's browser.

## Executive Summary

This system transforms CAPTCHA from wasteful computation into productive ML work by:
- Running lightweight ML inference/training tasks in the browser
- Collecting human-verified labels to build high-quality datasets
- Using computational cost as the security barrier (expensive for bots, cheap for humans)
- Preserving user privacy through federated learning and local-only data processing

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Python 3.11+ / FastAPI |
| **Database** | PostgreSQL 15+ / Redis 7+ |
| **Browser ML** | TensorFlow.js (primary) + ONNX Runtime Web (fallback) |
| **Widget** | Vanilla JS with Shadow DOM |
| **Distribution** | Embeddable `<script>` tag + NPM package |
| **ML Tasks** | CIFAR-10 (MobileNetV2) + IMDB (DistilBERT) |
| **Package Manager** | npm (workspaces) |
| **Cloud** | AWS (ECS, RDS, ElastiCache) |
| **Testing** | pytest, Jest, Playwright |
| **Demo** | Included |

---

## Project Structure

```
Capstone/
├── packages/
│   ├── widget/                    # Client-side CAPTCHA widget
│   │   ├── src/
│   │   │   ├── core/              # Core widget logic
│   │   │   │   ├── index.ts       # Main entry point
│   │   │   │   ├── session.ts     # Session management
│   │   │   │   ├── api-client.ts  # Server communication
│   │   │   │   └── config.ts      # Configuration
│   │   │   ├── ml/                # ML engine (TF.js + ONNX.js)
│   │   │   │   ├── engine.ts      # Unified ML interface
│   │   │   │   ├── tfjs-runtime.ts
│   │   │   │   ├── onnx-runtime.ts
│   │   │   │   └── model-loader.ts
│   │   │   ├── ui/                # Accessible UI components
│   │   │   │   ├── widget.ts      # Main widget component
│   │   │   │   ├── verification.ts # Human verification UI
│   │   │   │   ├── progress.ts    # Progress indicator
│   │   │   │   └── styles.ts      # Shadow DOM styles
│   │   │   └── utils/             # Utilities
│   │   │       ├── crypto.ts      # PoW hash generation
│   │   │       ├── timing.ts      # Performance timing
│   │   │       └── accessibility.ts
│   │   ├── tests/
│   │   └── package.json
│   │
│   └── sdk/                       # NPM SDK for framework integration
│       ├── src/
│       │   ├── react/             # React hooks/components
│       │   ├── vue/               # Vue composables
│       │   └── vanilla/           # Plain JS wrapper
│       └── package.json
│
├── server/                        # FastAPI backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                # FastAPI application
│   │   ├── config.py              # Settings management
│   │   ├── api/                   # REST API routes
│   │   │   ├── __init__.py
│   │   │   ├── captcha.py         # CAPTCHA endpoints
│   │   │   ├── verification.py    # Human verification endpoints
│   │   │   └── federated.py       # Federated learning endpoints
│   │   ├── core/                  # Core business logic
│   │   │   ├── __init__.py
│   │   │   ├── task_coordinator.py
│   │   │   ├── risk_scorer.py
│   │   │   └── difficulty_adapter.py
│   │   ├── ml/                    # ML backend services
│   │   │   ├── __init__.py
│   │   │   ├── inference_validator.py
│   │   │   ├── federated_aggregator.py
│   │   │   └── model_manager.py
│   │   ├── models/                # SQLAlchemy models
│   │   │   ├── __init__.py
│   │   │   ├── sample.py
│   │   │   ├── task.py
│   │   │   ├── session.py
│   │   │   ├── prediction.py
│   │   │   ├── verification.py
│   │   │   ├── golden_dataset.py
│   │   │   └── reputation.py
│   │   ├── schemas/               # Pydantic schemas
│   │   │   ├── __init__.py
│   │   │   ├── captcha.py
│   │   │   ├── verification.py
│   │   │   └── common.py
│   │   ├── services/              # Business services
│   │   │   ├── __init__.py
│   │   │   ├── golden_dataset.py
│   │   │   ├── reputation.py
│   │   │   └── privacy.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── security.py
│   │       └── hashing.py
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── test_captcha.py
│   │   ├── test_verification.py
│   │   └── test_coordinator.py
│   ├── alembic/                   # Database migrations
│   │   ├── env.py
│   │   ├── script.py.mako
│   │   └── versions/
│   ├── alembic.ini
│   └── requirements.txt
│
├── models/                        # Pre-trained ML models
│   ├── cifar10-mobilenet/
│   │   ├── model.json
│   │   └── weights.bin
│   └── imdb-distilbert/
│       ├── model.json
│       └── weights.bin
│
├── demo/                          # Demo application
│   ├── frontend/                  # Simple HTML/JS demo site
│   │   ├── index.html
│   │   ├── login.html
│   │   ├── contact.html
│   │   └── admin.html
│   └── backend/                   # Demo site backend
│       └── server.py
│
├── scripts/                       # Utility scripts
│   ├── model_converter.py         # Convert models to TF.js/ONNX
│   ├── seed_data.py               # Seed database with samples
│   └── generate_samples.py        # Generate test samples
│
├── docker/                        # Docker configs
│   ├── Dockerfile.server
│   ├── Dockerfile.widget
│   └── docker-compose.yml
│
├── docs/                          # Documentation
│   ├── API.md
│   ├── INTEGRATION.md
│   ├── ARCHITECTURE.md
│   └── SECURITY.md
│
├── .github/                       # GitHub Actions
│   └── workflows/
│       ├── test.yml
│       └── deploy.yml
│
├── package.json                   # Root package.json (workspaces)
├── tsconfig.json                  # TypeScript config
├── .eslintrc.js                   # ESLint config
├── .prettierrc                    # Prettier config
├── .env.example                   # Environment variables template
├── PLAN.md                        # This file
└── README.md
```

---

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER'S BROWSER                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        PoUW CAPTCHA Widget                           │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │    │
│  │  │   ML Engine  │  │  Verification │  │     Session Manager      │   │    │
│  │  │  (TF.js/ONNX)│  │      UI       │  │                          │   │    │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ HTTPS
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           APPLICATION SERVER                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         FastAPI Backend                              │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │    │
│  │  │  CAPTCHA API │  │ Verification │  │    Federated Learning    │   │    │
│  │  │   Endpoints  │  │   Endpoints  │  │       Endpoints          │   │    │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
        ┌───────────────────┐ ┌─────────────┐ ┌─────────────────────┐
        │  Task Coordinator │ │   Redis     │ │     PostgreSQL      │
        │                   │ │   Cache     │ │     Database        │
        │  - Risk Scoring   │ │             │ │                     │
        │  - Difficulty     │ │  - Sessions │ │  - Samples          │
        │  - Task Assignment│ │  - Rate     │ │  - Predictions      │
        └───────────────────┘ │    Limits   │ │  - Golden Dataset   │
                              └─────────────┘ │  - Reputation       │
                                              └─────────────────────┘
                                                        │
                    ┌───────────────────────────────────┤
                    ▼                                   ▼
        ┌─────────────────────────┐     ┌─────────────────────────────┐
        │   Inference Validator   │     │  Federated Learning         │
        │                         │     │  Aggregator                 │
        │  - Known-sample checks  │     │                             │
        │  - Prediction validation│     │  - Gradient collection      │
        │  - PoW verification     │     │  - Differential privacy     │
        └─────────────────────────┘     │  - Model updates            │
                                        └─────────────────────────────┘
```

### Data Flow Diagrams

#### 1. Inference Flow (Normal User)
```
┌──────┐     ┌────────┐     ┌────────────┐     ┌───────────┐
│Client│────▶│ Server │────▶│Coordinator │────▶│  Sample   │
│      │     │        │     │            │     │   Store   │
└──────┘     └────────┘     └────────────┘     └───────────┘
    │                              │
    │  1. Init CAPTCHA             │ 2. Assign task
    │                              │    (risk-based)
    ▼                              ▼
┌──────┐     ┌────────┐     ┌────────────┐
│ Load │────▶│  Run   │────▶│  Submit    │
│Model │     │Inference│    │ Prediction │
└──────┘     └────────┘     └────────────┘
    │                              │
    │  3. TF.js/ONNX               │ 4. Prediction + PoW hash
    │     (300-800ms)              │
    ▼                              ▼
┌──────┐     ┌────────┐     ┌────────────┐
│Result│◀────│Validate│◀────│  Inference │
│      │     │        │     │  Validator │
└──────┘     └────────┘     └────────────┘
                                   │
                                   │ 5. Verify against known samples
                                   ▼
                            ┌────────────┐
                            │  Success   │
                            │  Token     │
                            └────────────┘
```

#### 2. Verification Flow (20% of Sessions)
```
┌──────────────────────────────────────────────────────────┐
│                    After Inference                        │
└──────────────────────────────────────────────────────────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │  Show Prediction to User  │
              │                          │
              │  "Is this a CAT?"        │
              │  [Yes] [No] [Correct →]  │
              └──────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
       ┌────────┐     ┌──────────┐    ┌──────────┐
       │  Yes   │     │    No    │    │ Correct  │
       │(Confirm)│    │ (Wrong)  │    │ (Relabel)│
       └────────┘     └──────────┘    └──────────┘
            │               │               │
            └───────────────┴───────────────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │  Store Verification       │
              │  Update Reputation        │
              │  Check Consensus          │
              └──────────────────────────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │  ≥3 verifications?        │
              │  ≥80% agreement?          │
              └──────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
      ┌──────────────┐           ┌──────────────┐
      │   Promote    │           │   Discard    │
      │   to Golden  │           │   if <60%    │
      │   Dataset    │           │   agreement  │
      └──────────────┘           └──────────────┘
```

---

## Database Schema

### Tables

#### `samples` - Unlabeled ML Samples
```sql
CREATE TABLE samples (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    data_type VARCHAR(50) NOT NULL,          -- 'image', 'text'
    model_type VARCHAR(100) NOT NULL,        -- 'cifar10', 'imdb'
    data_hash VARCHAR(64) NOT NULL UNIQUE,   -- SHA-256 of data
    data_url TEXT,                           -- CDN URL for sample
    data_blob BYTEA,                         -- Or stored directly
    metadata JSONB DEFAULT '{}',
    times_served INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

#### `sessions` - CAPTCHA Sessions
```sql
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    domain VARCHAR(255) NOT NULL,
    session_token VARCHAR(255) NOT NULL UNIQUE,
    risk_score FLOAT DEFAULT 0.0,
    difficulty_tier VARCHAR(20) DEFAULT 'normal',  -- normal, suspicious, bot_like
    status VARCHAR(20) DEFAULT 'pending',          -- pending, processing, completed, failed
    client_fingerprint VARCHAR(64),                -- Anonymous hash
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    expires_at TIMESTAMP NOT NULL
);
```

#### `tasks` - Assigned ML Tasks
```sql
CREATE TABLE tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id),
    sample_id UUID REFERENCES samples(id),
    task_type VARCHAR(50) NOT NULL,          -- 'inference', 'gradient', 'training'
    expected_time_ms INTEGER NOT NULL,
    is_known_sample BOOLEAN DEFAULT FALSE,   -- Honeypot for validation
    known_label VARCHAR(100),                -- Expected answer for known samples
    status VARCHAR(20) DEFAULT 'assigned',
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### `predictions` - Client Predictions
```sql
CREATE TABLE predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES tasks(id),
    session_id UUID REFERENCES sessions(id),
    sample_id UUID REFERENCES samples(id),
    predicted_label VARCHAR(100) NOT NULL,
    confidence FLOAT NOT NULL,
    inference_time_ms INTEGER NOT NULL,
    pow_hash VARCHAR(128) NOT NULL,
    is_valid BOOLEAN,                        -- Passed validation?
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### `verifications` - Human Verification Responses
```sql
CREATE TABLE verifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_id UUID REFERENCES predictions(id),
    sample_id UUID REFERENCES samples(id),
    session_id UUID REFERENCES sessions(id),
    response_type VARCHAR(20) NOT NULL,      -- 'confirm', 'reject', 'correct'
    original_label VARCHAR(100) NOT NULL,
    verified_label VARCHAR(100),             -- NULL if confirmed, new label if corrected
    response_time_ms INTEGER,
    reputation_score FLOAT,                  -- User's reputation at time of verification
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### `golden_dataset` - Verified, Consensus-Reached Labels
```sql
CREATE TABLE golden_dataset (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sample_id UUID REFERENCES samples(id) UNIQUE,
    data_type VARCHAR(50) NOT NULL,
    verified_label VARCHAR(100) NOT NULL,
    confidence_score FLOAT NOT NULL,
    verification_count INTEGER NOT NULL,
    agreement_score FLOAT NOT NULL,          -- % agreement
    weighted_agreement FLOAT NOT NULL,       -- Reputation-weighted
    domain_attribution VARCHAR(255),         -- Which domain contributed most
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

#### `reputation_scores` - Anonymous User Reputation
```sql
CREATE TABLE reputation_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fingerprint_hash VARCHAR(64) NOT NULL UNIQUE,  -- Anonymous identifier
    score FLOAT DEFAULT 1.0,                       -- Range: 0.0 - 5.0
    correct_verifications INTEGER DEFAULT 0,
    incorrect_verifications INTEGER DEFAULT 0,
    total_sessions INTEGER DEFAULT 0,
    last_activity TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### `domain_config` - Per-Domain Settings
```sql
CREATE TABLE domain_config (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    domain VARCHAR(255) NOT NULL UNIQUE,
    api_key_hash VARCHAR(64) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    verification_rate FLOAT DEFAULT 0.2,     -- 20% default
    difficulty_multiplier FLOAT DEFAULT 1.0,
    allowed_origins TEXT[],
    webhook_url TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## API Specification

### CAPTCHA Endpoints

#### `POST /api/v1/captcha/init`
Initialize a new CAPTCHA session.

**Request:**
```json
{
    "domain": "example.com",
    "api_key": "pk_live_xxxxx",
    "client_metadata": {
        "user_agent": "Mozilla/5.0...",
        "language": "en-US",
        "timezone": "America/New_York"
    }
}
```

**Response:**
```json
{
    "session_id": "uuid",
    "challenge_token": "jwt_token",
    "task": {
        "task_id": "uuid",
        "model_url": "https://cdn.pouw.dev/models/cifar10/model.json",
        "sample_url": "https://cdn.pouw.dev/samples/abc123.jpg",
        "sample_data": "base64_encoded_data",
        "task_type": "inference",
        "expected_time_ms": 500
    },
    "difficulty": "normal",
    "expires_at": "2024-01-01T12:00:00Z"
}
```

#### `POST /api/v1/captcha/submit`
Submit inference result.

**Request:**
```json
{
    "session_id": "uuid",
    "task_id": "uuid",
    "prediction": {
        "label": "cat",
        "confidence": 0.94,
        "top_k": [
            {"label": "cat", "confidence": 0.94},
            {"label": "dog", "confidence": 0.04},
            {"label": "bird", "confidence": 0.02}
        ]
    },
    "proof_of_work": {
        "hash": "sha256_hash",
        "nonce": 12345,
        "model_checksum": "abc123"
    },
    "timing": {
        "model_load_ms": 150,
        "inference_ms": 320,
        "total_ms": 470
    }
}
```

**Response (No Verification Needed):**
```json
{
    "success": true,
    "requires_verification": false,
    "captcha_token": "eyJhbGciOiJIUzI1NiIs...",
    "expires_at": "2024-01-01T12:05:00Z"
}
```

**Response (Verification Required):**
```json
{
    "success": true,
    "requires_verification": true,
    "verification": {
        "verification_id": "uuid",
        "display_data": {
            "type": "image",
            "url": "https://cdn.pouw.dev/samples/abc123.jpg"
        },
        "predicted_label": "cat",
        "prompt": "Is this image a cat?",
        "options": ["Yes, it's correct", "No, it's wrong"]
    }
}
```

#### `POST /api/v1/captcha/verify`
Submit human verification response.

**Request:**
```json
{
    "session_id": "uuid",
    "verification_id": "uuid",
    "response": "confirm",
    "corrected_label": null
}
```

**Response:**
```json
{
    "success": true,
    "captcha_token": "eyJhbGciOiJIUzI1NiIs...",
    "expires_at": "2024-01-01T12:05:00Z"
}
```

#### `GET /api/v1/captcha/validate/{token}`
Validate a CAPTCHA token (server-to-server).

**Response:**
```json
{
    "valid": true,
    "session_id": "uuid",
    "domain": "example.com",
    "completed_at": "2024-01-01T12:00:30Z",
    "difficulty": "normal",
    "verification_performed": true
}
```

---

## Security Model

### Threat Vectors and Mitigations

| Threat | Description | Mitigation |
|--------|-------------|------------|
| **Bot Farms** | Automated solving at scale | Computational cost scales with volume; 10s tasks for suspicious traffic |
| **Sybil Attacks** | Multiple fake identities | Reputation system with slow accumulation; new users have low weight |
| **Model Poisoning** | Malicious gradient updates | Gradient clipping, norm validation, known-sample injection |
| **Replay Attacks** | Reusing valid responses | Session tokens with short expiry; one-time use PoW hashes |
| **Timing Attacks** | Faking inference time | Server-side timing validation; cryptographic timing proofs |
| **Headless Browsers** | Automated Chrome/Puppeteer | Behavioral analysis; WebGL fingerprinting (privacy-preserving) |

### Adaptive Difficulty Tiers

```python
DIFFICULTY_TIERS = {
    "normal": {
        "risk_score_max": 0.3,
        "inference_time_ms": 500,
        "task_type": "inference_only",
        "verification_probability": 0.2,
        "model_complexity": "standard"
    },
    "suspicious": {
        "risk_score_max": 0.7,
        "inference_time_ms": 3000,
        "task_type": "inference_with_gradient",
        "verification_probability": 0.5,
        "model_complexity": "enhanced"
    },
    "bot_like": {
        "risk_score_max": 1.0,
        "inference_time_ms": 10000,
        "task_type": "training_batch",
        "verification_probability": 1.0,
        "model_complexity": "maximum"
    }
}
```

### Risk Scoring Factors

```python
RISK_FACTORS = {
    "request_frequency": 0.3,      # Requests per minute
    "session_velocity": 0.2,       # How fast sessions complete
    "behavioral_signals": 0.2,     # Mouse/keyboard patterns
    "reputation_history": 0.15,    # Past verification accuracy
    "known_sample_accuracy": 0.15  # Performance on honeypots
}
```

---

## Privacy Model

### Data Minimization Principles

1. **No PII Collection**: No names, emails, or identifying information
2. **No IP Tracking**: IPs used only for rate limiting, not stored
3. **No Fingerprinting**: Only anonymous session hashes
4. **Local Processing**: All ML inference happens in browser
5. **Aggregated Metrics**: Statistics reported at domain level only

### Federated Learning Privacy

```python
PRIVACY_CONFIG = {
    "gradient_clipping_norm": 1.0,
    "differential_privacy": {
        "enabled": True,
        "epsilon": 1.0,
        "delta": 1e-5,
        "noise_multiplier": 1.1
    },
    "secure_aggregation": {
        "min_clients": 10,
        "threshold": 0.6
    }
}
```

### Compliance Targets

- **GDPR**: No personal data processing; anonymous aggregation only
- **CCPA**: No sale of personal information; no tracking
- **WCAG 2.2**: Full accessibility support (AA compliance)

---

## Golden Dataset Pipeline

### Consensus Algorithm

```python
def calculate_consensus(sample_id: str) -> ConsensusResult:
    verifications = get_verifications(sample_id)
    
    if len(verifications) < MIN_VERIFICATIONS:  # 3
        return ConsensusResult(status="pending")
    
    # Weight by reputation
    weighted_votes = {}
    total_weight = 0
    
    for v in verifications:
        weight = v.reputation_score
        label = v.verified_label or v.original_label
        weighted_votes[label] = weighted_votes.get(label, 0) + weight
        total_weight += weight
    
    # Find majority
    top_label = max(weighted_votes, key=weighted_votes.get)
    agreement = weighted_votes[top_label] / total_weight
    
    if agreement >= 0.8:
        return ConsensusResult(
            status="golden",
            label=top_label,
            agreement=agreement
        )
    elif agreement < 0.6:
        return ConsensusResult(status="discarded")
    else:
        return ConsensusResult(status="needs_more_verification")
```

### Reputation System

```python
def update_reputation(user_hash: str, was_correct: bool):
    current = get_reputation(user_hash)
    
    if was_correct:
        delta = +0.1
    else:
        delta = -0.2  # Asymmetric penalty
    
    new_score = max(0.0, min(5.0, current.score + delta))
    save_reputation(user_hash, new_score)
```

---

## Accessibility (WCAG 2.2)

### Requirements

1. **Keyboard Navigation**
   - All interactive elements focusable via Tab
   - Enter/Space to activate buttons
   - Escape to dismiss dialogs

2. **Screen Reader Support**
   - ARIA labels on all elements
   - Live regions for status updates
   - Descriptive button text

3. **Visual Accessibility**
   - Minimum 4.5:1 contrast ratio
   - No reliance on color alone
   - Visible focus indicators

4. **Motion Sensitivity**
   - Respect `prefers-reduced-motion`
   - No auto-playing animations
   - Static alternatives available

### Implementation

```html
<div role="region" 
     aria-label="Security verification"
     aria-live="polite">
  
  <div class="pouw-status" aria-atomic="true">
    Running security check...
  </div>
  
  <button class="pouw-confirm"
          aria-describedby="pouw-instruction">
    Yes, this is correct
  </button>
  
  <p id="pouw-instruction" class="sr-only">
    Confirm if the displayed image matches the label "cat"
  </p>
</div>
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
- [x] Project structure and tooling
- [ ] Database schema and migrations
- [ ] Core FastAPI application
- [ ] Basic CAPTCHA API endpoints
- [ ] Redis session management
- [ ] Docker Compose for local dev

### Phase 2: ML Engine (Week 2-3)
- [ ] TensorFlow.js integration
- [ ] ONNX Runtime Web fallback
- [ ] Model loading and caching
- [ ] Inference execution
- [ ] Proof-of-work generation
- [ ] Performance timing

### Phase 3: Widget UI (Week 3)
- [ ] Shadow DOM encapsulation
- [ ] Loading/processing states
- [ ] Verification UI
- [ ] Accessibility implementation
- [ ] Keyboard navigation
- [ ] Screen reader support

### Phase 4: Validation & Dataset (Week 4)
- [ ] Inference validator
- [ ] Known-sample honeypots
- [ ] Verification flow
- [ ] Consensus algorithm
- [ ] Reputation system
- [ ] Golden dataset storage

### Phase 5: Security (Week 4-5)
- [ ] Risk scoring engine
- [ ] Adaptive difficulty
- [ ] Rate limiting
- [ ] Behavioral analysis
- [ ] PoW verification

### Phase 6: Demo & SDK (Week 5)
- [ ] Demo application
- [ ] NPM package
- [ ] React hooks
- [ ] Vue composables
- [ ] Integration docs

### Phase 7: Testing (Ongoing)
- [ ] Unit tests (80%+ coverage)
- [ ] Integration tests
- [ ] E2E tests (Playwright)
- [ ] Performance benchmarks
- [ ] Accessibility audits

### Phase 8: Deployment (Week 6)
- [ ] AWS infrastructure
- [ ] CI/CD pipeline
- [ ] Monitoring setup
- [ ] Documentation

---

## ML Models

### CIFAR-10 (Image Classification)

| Property | Value |
|----------|-------|
| Architecture | MobileNetV2 (quantized INT8) |
| Input Size | 32x32x3 |
| Output | 10 classes |
| Model Size | ~3MB (TF.js) |
| Inference Time | 200-400ms |
| Use Case | Normal difficulty |

### IMDB Sentiment (Text Classification)

| Property | Value |
|----------|-------|
| Architecture | DistilBERT (distilled) |
| Input Size | 256 tokens max |
| Output | 2 classes (pos/neg) |
| Model Size | ~25MB (with tokenizer) |
| Inference Time | 400-600ms |
| Use Case | Text verification tasks |

---

## Economic Model

### Value Generation

```
┌─────────────────────────────────────────────────────────────┐
│                    TRADITIONAL CAPTCHA                       │
│  User Time → Puzzle Solving → Zero Value → Discarded        │
└─────────────────────────────────────────────────────────────┘

                            ▼

┌─────────────────────────────────────────────────────────────┐
│                      PoUW CAPTCHA                            │
│  User Time → ML Inference → Verified Labels → ML Training   │
│                              ↓                               │
│                       Golden Dataset                         │
│                              ↓                               │
│                    Dataset Marketplace                       │
└─────────────────────────────────────────────────────────────┘
```

### Stakeholders

| Stakeholder | Value Received |
|-------------|----------------|
| **Websites** | Bot protection + share of dataset revenue |
| **Users** | Productive computation instead of puzzles |
| **AI Developers** | High-quality labeled training data |
| **ML Researchers** | Federated learning insights |

---

## Success Metrics

### Security Metrics
- Bot detection rate: >95%
- False positive rate: <1%
- Attack cost amplification: 100x for bots vs humans

### Performance Metrics
- P50 inference time: <500ms
- P95 inference time: <800ms
- API response time: <100ms

### Quality Metrics
- Golden dataset precision: >95%
- Human verification accuracy: >90%
- Model improvement per 1M samples: measurable

### UX Metrics
- Completion rate: >98%
- User friction score: <2s average interaction
- Accessibility audit: WCAG 2.2 AA

---

## References

- [TensorFlow.js Documentation](https://www.tensorflow.org/js)
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)
- [Federated Learning](https://federated.withgoogle.com/)
- [WCAG 2.2 Guidelines](https://www.w3.org/TR/WCAG22/)
- [Differential Privacy](https://www.microsoft.com/en-us/research/publication/differential-privacy/)

---

*Last Updated: January 2025*
