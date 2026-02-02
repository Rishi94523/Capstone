# PoUW CAPTCHA - Architecture Documentation

## Overview

**Proof-of-Useful-Work (PoUW) CAPTCHA** replaces traditional puzzles with actual ML inference tasks. Users' browsers execute neural network forward passes as proof-of-work, making bot automation computationally expensive while contributing useful ML computation.

---

## Answers to Your Questions

### 1. Is the Dataset/Image Sent with the Request?

**YES** - The sample data (image) is sent embedded in the `/captcha/init` response as Base64-encoded data in the `sample_data` field.

```json
{
  "session_id": "uuid",
  "task": {
    "sample_data": "BASE64_ENCODED_IMAGE_BYTES",
    "sample_url": null,
    "sample_type": "image"
  }
}
```

**Two modes are supported:**

- **Inline Data**: `sample_data` contains Base64-encoded image bytes (current implementation)
- **CDN URL**: `sample_url` points to an external CDN for larger samples

The sample is a 28x28 grayscale image (784 bytes) for MNIST, or 32x32x3 (3072 bytes) for CIFAR-10.

### 2. Can Training Be Done?

**YES** - The system supports training, but it's currently configured for inference tasks only.

**Training capabilities:**

- `scripts/train_mnist_model.py` - Trains a tiny MNIST CNN (~50KB model)
- Model is already trained: `models/mnist-tiny/model_full.keras` (98.22% accuracy)
- Shard-based training is architecturally supported but not fully implemented

**To train a new model:**

```bash
# Use Anaconda Python (TensorFlow doesn't support Python 3.14)
"F:\programfiles\Anaconda\python.exe" scripts/train_mnist_model.py --output models/mnist-tiny --epochs 10
```

**Future training modes:**

- `gradient_contribution` - Clients compute gradients on their samples
- `federated_training` - Aggregated model updates from multiple clients

---

## System Architecture

```mermaid
flowchart TB
    subgraph Client["Browser (Client)"]
        UI[CAPTCHA Widget UI]
        ML[ML Engine<br/>TensorFlow.js / ONNX]
        Shard[Shard Executor<br/>Conv2D, Dense, MaxPool]
        Crypto[Crypto Utils<br/>Hash Generation]
    end

    subgraph Server["Backend Server"]
        API[FastAPI Endpoints]
        Risk[Risk Scorer]
        Coord[Task Coordinator]
        Valid[Inference Validator]

        subgraph Storage
            DB[(SQLite/PostgreSQL)]
            Redis[(Redis/In-Memory)]
        end
    end

    subgraph External["External Resources"]
        CDN[Model CDN]
        Samples[Sample Storage]
    end

    UI -->|1. Click Verify| API
    API -->|2. Compute Risk| Risk
    Risk -->|3. Get Difficulty| Coord
    Coord -->|4. Assign Task| API
    API -->|5. Return Challenge| UI

    UI -->|6. Load Model| CDN
    UI -->|7. Run Inference| ML
    ML --> Shard
    Shard --> Crypto

    UI -->|8. Submit Proof| API
    API -->|9. Validate| Valid
    Valid -->|10. Issue Token| UI
```

---

## Data Flow Sequence

```mermaid
sequenceDiagram
    participant B as Browser
    participant S as Server
    participant DB as Database
    participant R as Redis

    Note over B,R: Phase 1: Initialization
    B->>S: POST /captcha/init<br/>{site_key, client_metadata}
    S->>R: Check rate limit
    S->>S: Compute risk score
    S->>DB: Create session
    S->>DB: Select sample + Create task
    S-->>B: {session_id, challenge_token, task}

    Note over B,R: Phase 2: ML Inference (Client-Side)
    B->>B: Decode sample_data (Base64)
    B->>B: Load model shards
    B->>B: Execute neural network layers
    B->>B: Generate prediction + proof hash

    Note over B,R: Phase 3: Verification
    B->>S: POST /captcha/submit<br/>{session_id, prediction, proof_of_work}
    S->>S: Validate timing
    S->>S: Verify proof hash
    S->>DB: Store prediction
    alt Suspicious Activity
        S-->>B: {requires_verification: true}
        B->>B: Show human verification UI
        B->>S: POST /captcha/verify
    end
    S-->>B: {captcha_token}

    Note over B,R: Phase 4: Token Validation (Server-to-Server)
    B->>B: Submit form with token
    Note right of B: Website Backend
    B->>S: GET /captcha/validate/{token}
    S->>DB: Verify session completed
    S-->>B: {valid: true}
```

---

## Component Architecture

```mermaid
classDiagram
    class CaptchaWidget {
        +init(siteKey, container)
        +verify(): Promise~Token~
        +reset()
        -runInference(task)
        -generateProof(output)
    }

    class MLEngine {
        +loadModel(url)
        +runInference(input): Prediction
        +executeShards(shards, input)
        -tfjsRuntime: TFJSRuntime
        -onnxRuntime: ONNXRuntime
        -shardEngine: ShardInferenceEngine
    }

    class ShardInferenceEngine {
        +executeShard(shard, input)
        +conv2dForward(input)
        +denseForward(input)
        +maxPoolForward(input)
        -generateProof(outputs)
    }

    class TaskCoordinator {
        +assignTask(sessionId, difficulty)
        +getDifficultyTier(riskScore)
        -selectSample(useKnown)
        -DIFFICULTY_TIERS
    }

    class RiskScorer {
        +computeRiskScore(ip, ua, siteKey)
        -checkRateLimit(ip)
        -analyzeUserAgent(ua)
        -getReputation(ip)
    }

    class InferenceValidator {
        +validatePrediction(task, prediction, proof)
        +shouldRequireVerification(session)
        -verifyProofOfWork(proof)
        -checkTiming(timing)
    }

    CaptchaWidget --> MLEngine
    MLEngine --> ShardInferenceEngine
    TaskCoordinator --> RiskScorer
    InferenceValidator --> TaskCoordinator
```

---

## Database Schema

```mermaid
erDiagram
    SESSIONS ||--o{ TASKS : has
    SESSIONS ||--o{ PREDICTIONS : has
    SESSIONS ||--o{ VERIFICATIONS : has
    TASKS ||--|| SAMPLES : uses
    TASKS ||--o| PREDICTIONS : generates

    SESSIONS {
        uuid id PK
        string domain
        string session_token
        float risk_score
        string difficulty_tier
        string status
        datetime expires_at
        datetime completed_at
    }

    TASKS {
        uuid id PK
        uuid session_id FK
        uuid sample_id FK
        string task_type
        int expected_time_ms
        bool is_known_sample
        string known_label
        string status
        json metadata
    }

    SAMPLES {
        uuid id PK
        string data_type
        string model_type
        string data_hash
        string data_url
        blob data_blob
        json metadata
        int times_served
    }

    PREDICTIONS {
        uuid id PK
        uuid task_id FK
        uuid session_id FK
        string predicted_label
        float confidence
        int inference_time_ms
        string pow_hash
        bool is_valid
    }

    VERIFICATIONS {
        uuid id PK
        uuid session_id FK
        string response
        bool is_correct
        datetime responded_at
    }
```

---

## Difficulty Tiers

```mermaid
flowchart LR
    subgraph Risk["Risk Score"]
        R1[0.0 - 0.3]
        R2[0.3 - 0.7]
        R3[0.7 - 1.0]
    end

    subgraph Difficulty["Difficulty Tier"]
        D1[Normal<br/>1 layer, 20ms]
        D2[Suspicious<br/>3 layers, 100ms]
        D3[Bot-Like<br/>6 layers, 200ms]
    end

    subgraph Verification["Verification"]
        V1[20% probability]
        V2[50% probability]
        V3[100% required]
    end

    R1 --> D1 --> V1
    R2 --> D2 --> V2
    R3 --> D3 --> V3
```

---

## Neural Network Shard Execution

```mermaid
flowchart LR
    subgraph Input
        I[28x28x1<br/>Image]
    end

    subgraph Shard1["Shard 1 (Easy)"]
        C1[Conv2D 8 filters]
        P1[MaxPool2D]
    end

    subgraph Shard2["Shard 2 (Medium)"]
        C2[Conv2D 16 filters]
        P2[MaxPool2D]
    end

    subgraph Shard3["Shard 3 (Hard)"]
        F[Flatten]
        D1[Dense 32]
        D2[Dense 10]
    end

    subgraph Output
        O[Softmax<br/>Prediction]
    end

    I --> C1 --> P1 --> C2 --> P2 --> F --> D1 --> D2 --> O

    style Shard1 fill:#90EE90
    style Shard2 fill:#FFD700
    style Shard3 fill:#FF6B6B
```

---

## Deployment Architecture

```mermaid
flowchart TB
    subgraph CDN["CDN (CloudFlare/Vercel)"]
        Widget[widget.js<br/>~50KB]
        Models[Model Shards<br/>~50KB each]
    end

    subgraph LB["Load Balancer"]
        Nginx[Nginx/Traefik]
    end

    subgraph API["API Servers"]
        API1[FastAPI Instance 1]
        API2[FastAPI Instance 2]
        API3[FastAPI Instance N]
    end

    subgraph Data["Data Layer"]
        PG[(PostgreSQL<br/>Primary)]
        PGR[(PostgreSQL<br/>Replica)]
        RD[(Redis Cluster)]
    end

    User[User Browser] --> CDN
    User --> Nginx
    Nginx --> API1 & API2 & API3
    API1 & API2 & API3 --> PG
    API1 & API2 & API3 --> RD
    PG -.-> PGR
```

---

## File Structure

```
Capstone/
├── server/                     # Python FastAPI Backend
│   ├── app/
│   │   ├── api/captcha.py     # REST endpoints
│   │   ├── core/
│   │   │   ├── task_coordinator.py
│   │   │   └── risk_scorer.py
│   │   ├── ml/
│   │   │   ├── inference_validator.py
│   │   │   ├── shard_manager.py
│   │   │   └── ground_truth_cache.py
│   │   ├── models/            # SQLAlchemy models
│   │   ├── schemas/           # Pydantic schemas
│   │   └── utils/
│   └── tests/
│
├── packages/widget/           # TypeScript CAPTCHA Widget
│   └── src/
│       ├── core/
│       │   ├── captcha.ts     # Main widget class
│       │   └── api-client.ts
│       ├── ml/
│       │   ├── engine.ts      # Unified ML engine
│       │   ├── shard-engine.ts # Shard executor
│       │   ├── tfjs-runtime.ts
│       │   └── onnx-runtime.ts
│       └── ui/
│
├── scripts/
│   └── train_mnist_model.py   # Model training
│
├── models/
│   └── mnist-tiny/            # Trained model files
│
└── demo/frontend/             # Demo application
```

---

## API Endpoints

| Endpoint                           | Method | Description                     |
| ---------------------------------- | ------ | ------------------------------- |
| `/api/v1/captcha/init`             | POST   | Initialize session, get ML task |
| `/api/v1/captcha/submit`           | POST   | Submit prediction + proof       |
| `/api/v1/captcha/validate/{token}` | GET    | Server-to-server validation     |
| `/health`                          | GET    | Health check                    |
| `/ready`                           | GET    | Readiness check                 |

---

## Security Considerations

1. **Proof-of-Work Verification**: Server validates hash of inference outputs
2. **Timing Analysis**: Suspiciously fast responses trigger harder challenges
3. **Honeypot Samples**: Known-label samples detect automated responses
4. **Rate Limiting**: Per-IP request limits via Redis
5. **Token Expiry**: Challenge tokens expire in 5 minutes
6. **JWT Signing**: All tokens are cryptographically signed

---

## Future Enhancements

- [ ] Federated learning with gradient aggregation
- [ ] Model sharding across multiple clients
- [ ] WebGPU acceleration for faster inference
- [ ] Distributed training contributions
- [ ] Real-time model updates
