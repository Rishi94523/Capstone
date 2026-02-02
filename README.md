# PoUW CAPTCHA

**Proof-of-Useful-Work CAPTCHA System**

A web-based CAPTCHA system that replaces traditional puzzle-based CAPTCHAs with productive machine learning computation executed inside the user's browser.

## Features

- **Useful Computation**: Browser performs ML inference instead of solving puzzles
- **Security**: Computational cost makes bot attacks economically expensive
- **Golden Dataset**: Human-verified labels improve ML model training
- **Privacy-First**: All computation happens locally in the browser
- **Accessible**: No distorted images or audio challenges; WCAG 2.2 compliant

## Quick Start

### Prerequisites

- Node.js 18+
- Python 3.11+
- Docker & Docker Compose
- PostgreSQL 15+ (or use Docker)
- Redis 7+ (or use Docker)

### Development Setup

1. **Clone and install dependencies**

```bash
# Install frontend dependencies
npm install

# Install Python dependencies
cd server
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Start infrastructure with Docker**

```bash
docker-compose -f docker/docker-compose.yml up -d postgres redis
```

3. **Configure environment**

```bash
cp .env.example .env
# Edit .env with your settings
```

4. **Run database migrations**

```bash
cd server
alembic upgrade head
```

5. **Start the development servers**

```bash
# Terminal 1: Backend API
cd server
uvicorn app.main:app --reload

# Terminal 2: Widget development
cd packages/widget
npm run dev

# Terminal 3: Demo site (optional)
# Open demo/frontend/index.html in browser
```

6. **Access the application**

- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Widget Dev: http://localhost:5173
- Demo: http://localhost:3000

## Project Structure

```
├── packages/
│   ├── widget/          # Client-side CAPTCHA widget (TF.js + ONNX)
│   └── sdk/             # NPM SDK for framework integration
├── server/              # FastAPI backend
│   ├── app/
│   │   ├── api/         # REST API endpoints
│   │   ├── core/        # Task coordination, risk scoring
│   │   ├── ml/          # Inference validation
│   │   ├── models/      # SQLAlchemy models
│   │   ├── schemas/     # Pydantic schemas
│   │   └── services/    # Golden dataset, reputation
│   └── alembic/         # Database migrations
├── models/              # Pre-trained ML models
├── demo/                # Demo application
├── docker/              # Docker configuration
└── docs/                # Documentation
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   User's Browser                     │
│  ┌───────────────────────────────────────────────┐  │
│  │             PoUW CAPTCHA Widget                │  │
│  │  • TensorFlow.js / ONNX Runtime               │  │
│  │  • ML Inference (300-800ms)                   │  │
│  │  • Human Verification UI                      │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                   FastAPI Server                     │
│  • Task Coordinator (risk-based difficulty)         │
│  • Inference Validator                              │
│  • Golden Dataset Pipeline                          │
│  • Reputation System                                │
└─────────────────────────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
       ┌────────────┐          ┌────────────┐
       │ PostgreSQL │          │   Redis    │
       │ (data)     │          │ (sessions) │
       └────────────┘          └────────────┘
```

## Integration

### Script Tag (Recommended)

```html
<script src="https://cdn.pouw.dev/widget/v1/pouw-captcha.js"></script>
<script>
  const captcha = new PoUWCaptcha({
    siteKey: 'your-site-key',
    container: '#captcha-container',
    onSuccess: (token) => {
      // Send token to your backend for validation
      console.log('CAPTCHA solved:', token);
    },
  });
</script>
<div id="captcha-container"></div>
```

### React

```jsx
import { usePoUWCaptcha } from '@pouw/sdk/react';

function MyForm() {
  const { token, isVerified, CaptchaWidget } = usePoUWCaptcha({
    siteKey: 'your-site-key',
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!isVerified) return;

    // Validate token on your backend
    await submitForm({ captchaToken: token });
  };

  return (
    <form onSubmit={handleSubmit}>
      <CaptchaWidget />
      <button type="submit" disabled={!isVerified}>
        Submit
      </button>
    </form>
  );
}
```

## API Reference

### Initialize CAPTCHA

```
POST /api/v1/captcha/init
```

### Submit Prediction

```
POST /api/v1/captcha/submit
```

### Submit Verification

```
POST /api/v1/captcha/verify
```

### Validate Token (Server-to-Server)

```
GET /api/v1/captcha/validate/{token}
```

See [API Documentation](docs/API.md) for full details.

## Configuration

Key environment variables:

| Variable                    | Description                                   | Default |
| --------------------------- | --------------------------------------------- | ------- |
| `DATABASE_URL`              | PostgreSQL connection URL                     | -       |
| `REDIS_URL`                 | Redis connection URL                          | -       |
| `SECRET_KEY`                | JWT signing key                               | -       |
| `VERIFICATION_RATE`         | Rate of sessions requiring human verification | 0.2     |
| `NORMAL_DIFFICULTY_TIME_MS` | Expected inference time for normal users      | 500     |
| `CONSENSUS_THRESHOLD`       | Agreement required for golden dataset         | 0.8     |

See [.env.example](.env.example) for all options.

## Testing

```bash
# Backend tests
cd server
pytest

# Widget tests
cd packages/widget
npm test

# E2E tests
npm run test:e2e
```

## Deployment

### Docker

```bash
docker-compose -f docker/docker-compose.yml up -d
```

### AWS

See [deployment documentation](docs/DEPLOYMENT.md) for AWS ECS/Fargate setup.

## Security

- Adaptive difficulty based on risk scoring
- Known-sample injection for bot detection
- Rate limiting and behavioral analysis
- No raw data leaves the browser
- GDPR/CCPA compliant

See [Security Model](docs/SECURITY.md) for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- TensorFlow.js team for browser ML
- ONNX Runtime for cross-platform inference
- The open source community
