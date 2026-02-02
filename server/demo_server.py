"""
Simplified PoUW CAPTCHA Server for demo purposes.

This version uses SQLite and runs without Redis for easy local development.
"""

import logging
import uuid
import random
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="PoUW CAPTCHA API (Demo)",
    description="Simplified demo server",
    version="0.1.0-demo",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for demo
sessions = {}
tasks = {}

# Models
class ClientMetadata(BaseModel):
    user_agent: str
    language: str
    timezone: str
    screen_width: Optional[int] = None
    screen_height: Optional[int] = None

class CaptchaInitRequest(BaseModel):
    site_key: str
    client_metadata: ClientMetadata

class TopKPrediction(BaseModel):
    label: str
    confidence: float

class PredictionData(BaseModel):
    label: str
    confidence: float
    top_k: list[TopKPrediction]

class ProofOfWorkData(BaseModel):
    hash: str
    nonce: int
    model_checksum: str
    input_hash: str
    output_hash: str

class TimingData(BaseModel):
    model_load_ms: int
    inference_ms: int
    total_ms: int
    started_at: int
    completed_at: int

class CaptchaSubmitRequest(BaseModel):
    session_id: str
    task_id: str
    prediction: PredictionData
    proof_of_work: ProofOfWorkData
    timing: TimingData

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "0.1.0-demo"}

@app.get("/ready")
async def ready_check():
    return {"status": "ready"}

@app.get("/")
async def root():
    return {
        "name": "PoUW CAPTCHA API (Demo)",
        "version": "0.1.0-demo",
        "docs": "/docs",
        "note": "This is a simplified demo server without database",
    }

@app.post("/api/v1/captcha/init")
async def init_captcha(request: CaptchaInitRequest):
    """Initialize CAPTCHA session."""
    session_id = str(uuid.uuid4())
    task_id = str(uuid.uuid4())
    
    # Create session
    sessions[session_id] = {
        "created_at": datetime.utcnow(),
        "status": "pending",
        "domain": "demo",
    }
    
    # CIFAR-10 labels
    labels = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]
    
    # Create task
    tasks[task_id] = {
        "session_id": session_id,
        "sample_id": str(uuid.uuid4()),
        "expected_label": random.choice(labels),
    }
    
    logger.info(f"Session initialized: {session_id}")
    
    return {
        "session_id": session_id,
        "challenge_token": f"challenge_{session_id}",
        "task": {
            "task_id": task_id,
            "model_url": "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json",
            "sample_data": None,
            "sample_url": "https://via.placeholder.com/32x32",
            "sample_type": "image",
            "task_type": "inference",
            "expected_time_ms": 500,
            "model_meta": {
                "name": "cifar10-mobilenet",
                "version": "1.0.0",
                "input_shape": [1, 32, 32, 3],
                "labels": labels,
                "checksum": "demo123",
            },
        },
        "difficulty": "normal",
        "expires_at": (datetime.utcnow() + timedelta(minutes=5)).isoformat(),
    }

@app.post("/api/v1/captcha/submit")
async def submit_captcha(request: CaptchaSubmitRequest):
    """Submit CAPTCHA prediction."""
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if request.task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Update session
    sessions[request.session_id]["status"] = "completed"
    
    # Generate token
    captcha_token = f"captcha_token_{request.session_id}"
    
    logger.info(f"CAPTCHA completed: {request.session_id}")
    
    # 20% chance of requiring verification
    requires_verification = random.random() < 0.2
    
    if requires_verification:
        return {
            "success": True,
            "requires_verification": True,
            "verification": {
                "verification_id": str(uuid.uuid4()),
                "display_data": {
                    "type": "image",
                    "url": "https://via.placeholder.com/150",
                },
                "predicted_label": request.prediction.label,
                "prompt": f"Is this a {request.prediction.label}?",
                "options": [
                    {"id": "confirm", "label": "Yes, correct"},
                    {"id": "reject", "label": "No, wrong"},
                ],
            },
        }
    else:
        return {
            "success": True,
            "requires_verification": False,
            "captcha_token": captcha_token,
            "expires_at": (datetime.utcnow() + timedelta(minutes=5)).isoformat(),
        }

@app.post("/api/v1/captcha/verify")
async def submit_verification(request: dict):
    """Submit verification response."""
    session_id = request.get("session_id")
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    captcha_token = f"captcha_token_{session_id}"
    
    logger.info(f"Verification completed: {session_id}")
    
    return {
        "success": True,
        "captcha_token": captcha_token,
        "expires_at": (datetime.utcnow() + timedelta(minutes=5)).isoformat(),
    }

@app.get("/api/v1/captcha/validate/{token}")
async def validate_captcha(token: str):
    """Validate CAPTCHA token."""
    # Simple validation for demo
    is_valid = token.startswith("captcha_token_")
    
    if is_valid:
        return {
            "valid": True,
            "session_id": token.replace("captcha_token_", ""),
            "domain": "demo",
            "completed_at": datetime.utcnow().isoformat(),
            "difficulty": "normal",
            "verification_performed": False,
        }
    else:
        return {"valid": False}

@app.get("/stats")
async def get_stats():
    """Get server statistics."""
    return {
        "total_sessions": len(sessions),
        "total_tasks": len(tasks),
        "completed_sessions": len([s for s in sessions.values() if s["status"] == "completed"]),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
