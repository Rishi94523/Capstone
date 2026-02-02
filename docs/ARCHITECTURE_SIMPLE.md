# PoUW CAPTCHA - High Level Architecture

## Core Flow Diagram

```mermaid
flowchart LR
    subgraph Client["Browser (Client)"]
        UI[User Clicks Verify]
        Decode[Decode Base64 Image]
        ML[Run Neural Network<br/>Conv2D → MaxPool → Dense]
        Proof[Generate Proof Hash]
    end

    subgraph Server["Backend Server"]
        Init[Create Session<br/>Select Sample]
        Validate[Validate Proof<br/>Check Timing]
        Token[Issue CAPTCHA Token]
    end

    subgraph Data["Sample Data"]
        MNIST[MNIST 28x28<br/>Digit Images]
        CIFAR[CIFAR-10 32x32<br/>Object Images]
    end

    UI -->|1. POST /init| Init
    MNIST & CIFAR -->|Stored in DB| Init
    Init -->|2. Return session_id +<br/>Base64 encoded image| Decode
    Decode -->|3. Run locally| ML
    ML -->|4. Prediction + Hash| Proof
    Proof -->|5. POST /submit| Validate
    Validate -->|6. Return token| Token
```

## Simplified Sequence

```mermaid
sequenceDiagram
    participant U as User Browser
    participant S as Server
    participant DB as Database

    U->>S: POST /captcha/init {site_key, metadata}
    S->>DB: Get random MNIST/CIFAR image
    S-->>U: {session_id, sample_data: "BASE64_IMAGE", model_url}

    Note over U: Decode Base64 → Image pixels
    Note over U: Load model weights
    Note over U: Run inference layers:<br/>Conv2D → MaxPool → Dense → Softmax
    Note over U: prediction = "cat" (0.92 confidence)
    Note over U: proof_hash = SHA256(layer_outputs)

    U->>S: POST /captcha/submit {session_id, prediction, proof_hash, timing}
    S->>S: Verify hash matches expected output
    S->>S: Check timing is realistic (not too fast)
    S-->>U: {success: true, captcha_token}

    Note over U: Form submission with token
    U->>S: GET /captcha/validate/{token}
    S-->>U: {valid: true}
```

## Component Overview

```mermaid
flowchart TB
    subgraph Request["1. Client Request"]
        A[User visits website]
        B[Clicks 'Verify']
    end

    subgraph ServerProcess["2. Server Processing"]
        C[Compute Risk Score<br/>based on IP, User Agent]
        D{Risk Level?}
        E1[Low Risk<br/>1 layer, 20ms]
        E2[Medium Risk<br/>3 layers, 100ms]
        E3[High Risk<br/>6 layers, 200ms]
        F[Select Image Sample<br/>from MNIST/CIFAR dataset]
        G[Create Task Record]
    end

    subgraph Response["3. Server Response"]
        H[Return JSON:<br/>• session_id<br/>• Base64 image<br/>• model_url<br/>• difficulty]
    end

    subgraph ClientML["4. Client ML Inference"]
        I[Decode Base64 → Float32Array]
        J[Load Model Weights]
        K[Execute Layers:<br/>Conv2D 8 filters<br/>MaxPool 2x2<br/>Conv2D 16 filters<br/>MaxPool 2x2<br/>Dense 32<br/>Dense 10 softmax]
        L[Get Prediction + Confidence]
        M[Hash Layer Outputs]
    end

    subgraph Submit["5. Submit & Validate"]
        N[Send prediction + proof]
        O[Server validates hash]
        P[Issue signed token]
    end

    A --> B --> C --> D
    D -->|0-30%| E1
    D -->|30-70%| E2
    D -->|70-100%| E3
    E1 & E2 & E3 --> F --> G --> H
    H --> I --> J --> K --> L --> M --> N --> O --> P
```

## Data Flow Summary

```mermaid
flowchart LR
    subgraph S[Server]
        DB[(Image DB<br/>MNIST/CIFAR)]
        API[FastAPI]
    end

    subgraph C[Client Browser]
        Widget[CAPTCHA Widget]
        TF[TensorFlow.js<br/>or ONNX]
    end

    DB -->|1. Base64 Image| API
    API -->|2. JSON Response| Widget
    Widget -->|3. Pixels| TF
    TF -->|4. Prediction| Widget
    Widget -->|5. Proof Hash| API
    API -->|6. Token| Widget

    style DB fill:#4ade80
    style TF fill:#60a5fa
```

## What Gets Sent Where

| Step | Direction       | Data                                                |
| ---- | --------------- | --------------------------------------------------- |
| 1    | Client → Server | `{site_key, user_agent, timezone}`                  |
| 2    | Server → Client | `{session_id, sample_data: "BASE64...", model_url}` |
| 3    | Client (local)  | Decode image, run Conv2D, MaxPool, Dense layers     |
| 4    | Client → Server | `{prediction: "cat", confidence: 0.92, proof_hash}` |
| 5    | Server → Client | `{valid: true, captcha_token}`                      |

## Key Insight

```
Traditional CAPTCHA:  Human solves puzzle → Server validates answer
PoUW CAPTCHA:         Browser runs ML inference → Server validates computation proof
```

The "useful work" is actual neural network computation that could contribute to:

- Model inference on unlabeled data
- Gradient computation for training
- Distributed federated learning
