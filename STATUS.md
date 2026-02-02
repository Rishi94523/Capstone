# PoUW CAPTCHA System - Status

## Current Status: RUNNING ✓

The PoUW CAPTCHA system is now running in demo mode!

### What's Running

**Backend API Server:**

- URL: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- Stats: http://localhost:8000/stats

**Demo Application:**

- Located at: `demo/frontend/index.html`
- Open directly in your browser to test the CAPTCHA

### About Datasets & Models

**Current Setup:**
The system is running in **DEMO MODE** which means:

❌ **No Real ML Models Downloaded**

- The system doesn't require actual TensorFlow.js or ONNX models to demonstrate the flow
- Model URLs point to placeholder endpoints
- Inference is simulated server-side

❌ **No Real Datasets**

- No CIFAR-10 or IMDB datasets are needed for the demo
- The server generates dummy samples on the fly
- Labels are randomized from the CIFAR-10 label set

**What It's Running On:**

1. **SQLite Database** - Lightweight, in-memory storage
2. **FastAPI Demo Server** - Simplified version without heavy ML dependencies
3. **In-Memory Session Storage** - No Redis required for demo
4. **Mock ML Tasks** - Simulates inference without actual model execution

### To Use Real ML Models:

If you want to use actual machine learning models, you would need to:

1. **Download Pre-trained Models:**

   ```bash
   # MobileNetV2 for CIFAR-10 (TensorFlow.js format)
   # ~3MB
   wget https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json

   # DistilBERT for IMDB sentiment (ONNX format)
   # ~25MB
   # Download from Hugging Face
   ```

2. **Convert Models:**

   ```bash
   # TensorFlow → TensorFlow.js
   tensorflowjs_converter --input_format keras model.h5 tfjs_model/

   # PyTorch → ONNX
   # (use PyTorch's torch.onnx.export)
   ```

3. **Host Models:**
   - Put model files in `models/` directory
   - Serve via CDN or local file server
   - Update `MODEL_CDN_URL` in `.env`

4. **Add Real Datasets:**
   ```python
   # scripts/load_cifar10.py
   from tensorflow.keras.datasets import cifar10
   (x_train, y_train), (x_test, y_test) = cifar10.load_data()
   # Store in database
   ```

### Quick Test

Try the API:

```bash
# Health check
curl http://localhost:8000/health

# Get stats
curl http://localhost:8000/stats

# Initialize CAPTCHA
curl -X POST http://localhost:8000/api/v1/captcha/init \
  -H "Content-Type: application/json" \
  -d '{
    "site_key": "demo_key",
    "client_metadata": {
      "user_agent": "curl/7.0",
      "language": "en",
      "timezone": "UTC"
    }
  }'
```

### Next Steps

1. ✅ Backend server running (demo mode)
2. ⏳ Install widget dependencies (optional for demo)
3. ⏳ Download real ML models (for production)
4. ⏳ Set up PostgreSQL + Redis (for production)
5. ⏳ Configure cloud deployment (AWS)

**For now, you can test the complete CAPTCHA flow using the demo application!**

Open `demo/frontend/index.html` in your browser to see it in action.
