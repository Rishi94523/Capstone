"""
Simplified PoUW CAPTCHA Server for demo purposes.

This version uses SQLite and runs without Redis for easy local development.
"""

import logging
import uuid
import random
import json
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
from starlette.middleware.base import BaseHTTPMiddleware

# Configure logging - more verbose
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="PoUW CAPTCHA API (Demo)",
    description="Simplified demo server",
    version="0.1.0-demo",
)

# Middleware to log all request bodies
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Only log POST requests to /submit
        if request.method == "POST" and "submit" in request.url.path:
            # We can't read body here as it would consume it
            # Instead, log what we can
            logger.info(f"=" * 60)
            logger.info(f"INCOMING REQUEST: {request.method} {request.url.path}")
            logger.info(f"Content-Type: {request.headers.get('content-type')}")
            logger.info(f"Content-Length: {request.headers.get('content-length')}")
            logger.info(f"=" * 60)
        
        response = await call_next(request)
        return response

# Add middleware BEFORE CORS
app.add_middleware(RequestLoggingMiddleware)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add validation error handler to see exact errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # The body was already consumed by Pydantic, so we get it from the exception
    logger.error(f"=" * 60)
    logger.error(f"VALIDATION ERROR on {request.url.path}")
    logger.error(f"Errors: {json.dumps(exc.errors(), indent=2, default=str)}")
    
    # Log each error in detail
    for error in exc.errors():
        logger.error(f"  Field: {' -> '.join(str(x) for x in error.get('loc', []))}")
        logger.error(f"  Type: {error.get('type')}")
        logger.error(f"  Message: {error.get('msg')}")
        logger.error(f"  Input: {error.get('input')}")
    
    logger.error(f"=" * 60)
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "message": "Validation failed - check server logs for details"
        },
    )

# Debug endpoint to see raw request
@app.post("/api/v1/captcha/debug-submit")
async def debug_submit(request: Request):
    """Debug endpoint - accepts any JSON and logs it."""
    body = await request.body()
    body_str = body.decode('utf-8')
    
    logger.info(f"=" * 60)
    logger.info(f"DEBUG SUBMIT - Raw body received:")
    logger.info(body_str)
    
    try:
        body_json = json.loads(body_str)
        logger.info(f"Parsed JSON structure:")
        
        def log_structure(obj, prefix=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, (dict, list)):
                        logger.info(f"{prefix}{k}: {type(v).__name__}")
                        log_structure(v, prefix + "  ")
                    else:
                        logger.info(f"{prefix}{k}: {type(v).__name__} = {v}")
            elif isinstance(obj, list):
                logger.info(f"{prefix}[list of {len(obj)} items]")
                if obj:
                    log_structure(obj[0], prefix + "  ")
        
        log_structure(body_json)
        
        # Check what's expected vs what's received
        expected_keys = ["session_id", "task_id", "prediction", "proof_of_work", "timing"]
        received_keys = list(body_json.keys())
        logger.info(f"Expected keys: {expected_keys}")
        logger.info(f"Received keys: {received_keys}")
        logger.info(f"Missing keys: {set(expected_keys) - set(received_keys)}")
        logger.info(f"Extra keys: {set(received_keys) - set(expected_keys)}")
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
    
    logger.info(f"=" * 60)
    
    return {"status": "logged", "body_length": len(body_str)}

# In-memory storage for demo
sessions = {}
tasks = {}

# Models - Using Field aliases to accept snake_case from client
class ClientMetadata(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    user_agent: str = Field(alias="user_agent")
    language: str
    timezone: str
    screen_width: Optional[int] = Field(default=None, alias="screen_width")
    screen_height: Optional[int] = Field(default=None, alias="screen_height")

class CaptchaInitRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    site_key: str = Field(alias="site_key")
    client_metadata: ClientMetadata

class TopKPrediction(BaseModel):
    label: str
    confidence: float

class PredictionData(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    label: str
    confidence: float
    top_k: list[TopKPrediction] = Field(alias="top_k")

class ProofOfWorkData(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    hash: str
    nonce: int
    model_checksum: str = Field(alias="model_checksum")
    input_hash: str = Field(alias="input_hash")
    output_hash: str = Field(alias="output_hash")

class TimingData(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    model_load_ms: int = Field(alias="model_load_ms")
    inference_ms: int = Field(alias="inference_ms")
    total_ms: int = Field(alias="total_ms")
    started_at: int = Field(alias="started_at")
    completed_at: int = Field(alias="completed_at")

class CaptchaSubmitRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    session_id: str = Field(alias="session_id")
    task_id: str = Field(alias="task_id")
    prediction: PredictionData
    proof_of_work: ProofOfWorkData = Field(alias="proof_of_work")
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
    
    # Return camelCase to match TypeScript client expectations
    return {
        "sessionId": session_id,
        "challengeToken": f"challenge_{session_id}",
        "task": {
            "taskId": task_id,
            "modelUrl": "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json",
            "sampleData": None,
            "sampleUrl": "https://via.placeholder.com/224x224",
            "sampleType": "image",
            "taskType": "inference",
            "expectedTimeMs": 500,
            "modelMeta": {
                "name": "cifar10-mobilenet",
                "version": "1.0.0",
                "inputShape": [1, 224, 224, 3],
                "labels": labels,
                "checksum": "demo123",
            },
        },
        "difficulty": "normal",
        "expiresAt": (datetime.utcnow() + timedelta(minutes=5)).isoformat(),
    }

@app.post("/api/v1/captcha/submit")
async def submit_captcha(request: CaptchaSubmitRequest):
    """Submit CAPTCHA prediction."""
    logger.info(f"Submit request received: session_id={request.session_id}, task_id={request.task_id}")
    logger.info(f"Prediction: {request.prediction}")
    logger.info(f"Proof of work: {request.proof_of_work}")
    logger.info(f"Timing: {request.timing}")
    
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
            "requiresVerification": True,
            "verification": {
                "verificationId": str(uuid.uuid4()),
                "displayType": "image",
                "displayContent": "https://via.placeholder.com/150",
                "predictedLabel": request.prediction.label,
                "prompt": f"Is this a {request.prediction.label}?",
                "options": [
                    {"id": "confirm", "label": "Yes, correct", "type": "confirm"},
                    {"id": "reject", "label": "No, wrong", "type": "reject"},
                ],
            },
        }
    else:
        return {
            "success": True,
            "requiresVerification": False,
            "captchaToken": captcha_token,
            "expiresAt": (datetime.utcnow() + timedelta(minutes=5)).isoformat(),
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
        "captchaToken": captcha_token,
        "expiresAt": (datetime.utcnow() + timedelta(minutes=5)).isoformat(),
    }

@app.get("/api/v1/captcha/validate/{token}")
async def validate_captcha(token: str):
    """Validate CAPTCHA token."""
    # Simple validation for demo
    is_valid = token.startswith("captcha_token_")
    
    if is_valid:
        return {
            "valid": True,
            "sessionId": token.replace("captcha_token_", ""),
            "domain": "demo",
            "completedAt": datetime.utcnow().isoformat(),
            "difficulty": "normal",
            "verificationPerformed": False,
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
