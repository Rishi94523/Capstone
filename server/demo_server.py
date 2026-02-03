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
from fastapi.responses import JSONResponse, HTMLResponse
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
inference_log = []  # Store all inference results for dashboard

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
    # Use CIFAR-10 sample images from a public dataset for variety
    sample_images = [
        "https://via.placeholder.com/224x224/FF6B6B/ffffff?text=Sample+1",
        "https://via.placeholder.com/224x224/4ECDC4/ffffff?text=Sample+2",
        "https://via.placeholder.com/224x224/45B7D1/ffffff?text=Sample+3",
        "https://via.placeholder.com/224x224/96CEB4/ffffff?text=Sample+4",
        "https://via.placeholder.com/224x224/FFEAA7/333333?text=Sample+5",
        "https://via.placeholder.com/224x224/DDA0DD/333333?text=Sample+6",
        "https://via.placeholder.com/224x224/98D8C8/333333?text=Sample+7",
        "https://via.placeholder.com/224x224/F7DC6F/333333?text=Sample+8",
    ]
    sample_url = random.choice(sample_images)
    expected_label = random.choice(labels)
    
    tasks[task_id] = {
        "session_id": session_id,
        "sample_id": str(uuid.uuid4()),
        "sample_url": sample_url,
        "expected_label": expected_label,
        "created_at": datetime.utcnow().isoformat(),
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
            "sampleUrl": sample_url,
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
    
    # Get task info for logging
    task_info = tasks[request.task_id]
    
    # Log the inference result for dashboard
    inference_record = {
        "id": str(uuid.uuid4()),
        "session_id": request.session_id,
        "task_id": request.task_id,
        "image_url": task_info.get("sample_url", "unknown"),
        "expected_label": task_info.get("expected_label", "unknown"),
        "predicted_label": request.prediction.label,
        "confidence": request.prediction.confidence,
        "top_k": [{"label": p.label, "confidence": p.confidence} for p in request.prediction.top_k],
        "inference_ms": request.timing.inference_ms,
        "total_ms": request.timing.total_ms,
        "model_load_ms": request.timing.model_load_ms,
        "timestamp": datetime.utcnow().isoformat(),
        "correct": request.prediction.label == task_info.get("expected_label"),
        "proof_hash": request.proof_of_work.hash[:16] + "...",
    }
    inference_log.append(inference_record)
    logger.info(f"Logged inference: {inference_record['predicted_label']} (conf: {inference_record['confidence']:.2f})")
    
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
        "total_inferences": len(inference_log),
    }


@app.get("/api/v1/inferences")
async def get_inferences(limit: int = 100):
    """Get inference log data as JSON."""
    return {
        "inferences": inference_log[-limit:][::-1],  # Most recent first
        "total": len(inference_log),
    }


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """ML Inference Dashboard - Shows all classifications done."""
    
    # Calculate stats
    total = len(inference_log)
    correct = sum(1 for i in inference_log if i.get("correct", False))
    accuracy = (correct / total * 100) if total > 0 else 0
    avg_inference_ms = sum(i.get("inference_ms", 0) for i in inference_log) / total if total > 0 else 0
    avg_confidence = sum(i.get("confidence", 0) for i in inference_log) / total if total > 0 else 0
    
    # Build inference rows HTML
    inference_rows = ""
    for i, inf in enumerate(reversed(inference_log[-50:])):  # Show last 50
        correct_badge = '<span class="badge correct">Correct</span>' if inf.get("correct") else '<span class="badge incorrect">Incorrect</span>'
        top_k_html = ""
        for pred in inf.get("top_k", [])[:3]:
            top_k_html += f'<div class="topk-item"><span>{pred["label"]}</span><span>{pred["confidence"]:.1%}</span></div>'
        
        inference_rows += f'''
        <tr>
            <td>#{total - i}</td>
            <td><img src="{inf.get("image_url", "")}" alt="sample" class="sample-img" onerror="this.src='https://via.placeholder.com/60x60?text=?'"></td>
            <td><strong>{inf.get("predicted_label", "?")}</strong></td>
            <td>{inf.get("expected_label", "?")}</td>
            <td>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {inf.get("confidence", 0) * 100}%"></div>
                    <span>{inf.get("confidence", 0):.1%}</span>
                </div>
            </td>
            <td>{correct_badge}</td>
            <td>{inf.get("inference_ms", 0)}ms</td>
            <td class="topk-cell">{top_k_html}</td>
            <td class="timestamp">{inf.get("timestamp", "")[:19]}</td>
        </tr>
        '''
    
    if not inference_rows:
        inference_rows = '<tr><td colspan="9" class="empty">No inferences yet. Use the frontend to submit some CAPTCHA challenges!</td></tr>'
    
    html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML Inference Dashboard - PoUW CAPTCHA</title>
        <meta charset="UTF-8">
        <meta http-equiv="refresh" content="5">
        <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0f172a;
                color: #e2e8f0;
                min-height: 100vh;
                padding: 20px;
            }}
            .container {{ max-width: 1400px; margin: 0 auto; }}
            h1 {{
                font-size: 28px;
                margin-bottom: 8px;
                background: linear-gradient(135deg, #6366f1, #06b6d4);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}
            .subtitle {{ color: #64748b; margin-bottom: 24px; }}
            
            /* Stats Cards */
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 16px;
                margin-bottom: 24px;
            }}
            .stat-card {{
                background: #1e293b;
                border-radius: 12px;
                padding: 20px;
                border: 1px solid #334155;
            }}
            .stat-label {{ color: #64748b; font-size: 13px; margin-bottom: 4px; }}
            .stat-value {{ font-size: 32px; font-weight: 700; }}
            .stat-value.green {{ color: #10b981; }}
            .stat-value.blue {{ color: #3b82f6; }}
            .stat-value.purple {{ color: #8b5cf6; }}
            .stat-value.orange {{ color: #f59e0b; }}
            
            /* Table */
            .table-container {{
                background: #1e293b;
                border-radius: 12px;
                border: 1px solid #334155;
                overflow: hidden;
            }}
            .table-header {{
                padding: 16px 20px;
                border-bottom: 1px solid #334155;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .table-header h2 {{ font-size: 18px; }}
            .refresh-note {{ color: #64748b; font-size: 12px; }}
            
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #334155; }}
            th {{ background: #0f172a; color: #94a3b8; font-weight: 500; font-size: 12px; text-transform: uppercase; }}
            tr:hover {{ background: #334155; }}
            
            .sample-img {{ width: 50px; height: 50px; border-radius: 6px; object-fit: cover; }}
            
            .confidence-bar {{
                width: 100px;
                height: 20px;
                background: #334155;
                border-radius: 4px;
                position: relative;
                overflow: hidden;
            }}
            .confidence-fill {{
                height: 100%;
                background: linear-gradient(90deg, #10b981, #34d399);
                border-radius: 4px;
            }}
            .confidence-bar span {{
                position: absolute;
                left: 50%;
                top: 50%;
                transform: translate(-50%, -50%);
                font-size: 11px;
                font-weight: 600;
            }}
            
            .badge {{
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 11px;
                font-weight: 600;
            }}
            .badge.correct {{ background: rgba(16, 185, 129, 0.2); color: #10b981; }}
            .badge.incorrect {{ background: rgba(239, 68, 68, 0.2); color: #ef4444; }}
            
            .topk-cell {{ min-width: 150px; }}
            .topk-item {{
                display: flex;
                justify-content: space-between;
                font-size: 11px;
                color: #94a3b8;
                padding: 2px 0;
            }}
            
            .timestamp {{ font-size: 11px; color: #64748b; font-family: monospace; }}
            .empty {{ text-align: center; padding: 40px; color: #64748b; }}
            
            .header-row {{
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                margin-bottom: 24px;
            }}
            .api-note {{
                background: #1e293b;
                padding: 12px 16px;
                border-radius: 8px;
                font-size: 12px;
                color: #94a3b8;
                border: 1px solid #334155;
            }}
            .api-note code {{
                background: #0f172a;
                padding: 2px 6px;
                border-radius: 4px;
                color: #06b6d4;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header-row">
                <div>
                    <h1>ML Inference Dashboard</h1>
                    <p class="subtitle">Real-time visibility into CAPTCHA ML classifications</p>
                </div>
                <div class="api-note">
                    API: <code>GET /api/v1/inferences</code> | Auto-refresh: 5s
                </div>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Total Inferences</div>
                    <div class="stat-value blue">{total}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Accuracy Rate</div>
                    <div class="stat-value green">{accuracy:.1f}%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Avg Inference Time</div>
                    <div class="stat-value purple">{avg_inference_ms:.0f}ms</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Avg Confidence</div>
                    <div class="stat-value orange">{avg_confidence:.1%}</div>
                </div>
            </div>
            
            <div class="table-container">
                <div class="table-header">
                    <h2>Recent Inferences</h2>
                    <span class="refresh-note">Showing last 50 | Page auto-refreshes every 5 seconds</span>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Image</th>
                            <th>Predicted</th>
                            <th>Expected</th>
                            <th>Confidence</th>
                            <th>Result</th>
                            <th>Time</th>
                            <th>Top-K Predictions</th>
                            <th>Timestamp</th>
                        </tr>
                    </thead>
                    <tbody>
                        {inference_rows}
                    </tbody>
                </table>
            </div>
        </div>
    </body>
    </html>
    '''
    return html

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
