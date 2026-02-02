"""
CAPTCHA API schemas.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class ClientMetadata(BaseModel):
    """Client metadata for risk scoring."""

    user_agent: str = Field(..., description="Browser user agent")
    language: str = Field(..., description="Browser language")
    timezone: str = Field(..., description="Client timezone")
    screen_width: Optional[int] = Field(default=None, description="Screen width")
    screen_height: Optional[int] = Field(default=None, description="Screen height")


class CaptchaInitRequest(BaseModel):
    """Request to initialize a CAPTCHA session."""

    site_key: str = Field(..., min_length=10, description="Site API key")
    client_metadata: ClientMetadata = Field(..., description="Client information")


class ModelMeta(BaseModel):
    """ML model metadata."""

    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    input_shape: List[int] = Field(..., description="Expected input shape")
    labels: List[str] = Field(..., description="Output labels")
    checksum: str = Field(..., description="Model checksum for integrity")


class TaskInfo(BaseModel):
    """ML task information."""

    task_id: str = Field(..., description="Unique task ID")
    model_url: str = Field(..., description="URL to load model")
    sample_data: Optional[str] = Field(default=None, description="Base64 sample data")
    sample_url: Optional[str] = Field(default=None, description="Sample URL")
    sample_type: str = Field(..., description="Sample type: image or text")
    task_type: str = Field(..., description="Task type: inference, gradient, training")
    expected_time_ms: int = Field(..., description="Expected completion time")
    model_meta: ModelMeta = Field(..., description="Model metadata")


class CaptchaInitResponse(BaseModel):
    """Response from CAPTCHA initialization."""

    session_id: str = Field(..., description="Session ID")
    challenge_token: str = Field(..., description="JWT challenge token")
    task: TaskInfo = Field(..., description="Assigned ML task")
    difficulty: str = Field(..., description="Difficulty tier")
    expires_at: datetime = Field(..., description="Session expiry time")


class TopKPrediction(BaseModel):
    """Top-K prediction entry."""

    label: str = Field(..., description="Predicted label")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")


class PredictionData(BaseModel):
    """Prediction result from client."""

    label: str = Field(..., description="Top predicted label")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    top_k: List[TopKPrediction] = Field(..., description="Top-K predictions")


class ProofOfWorkData(BaseModel):
    """Proof-of-work data from client."""

    hash: str = Field(..., min_length=64, description="PoW hash")
    nonce: int = Field(..., ge=0, description="Nonce used")
    model_checksum: str = Field(..., description="Model checksum")
    input_hash: str = Field(..., description="Input data hash")
    output_hash: str = Field(..., description="Output data hash")


class TimingData(BaseModel):
    """Timing information from client."""

    model_load_ms: int = Field(..., ge=0, description="Model load time")
    inference_ms: int = Field(..., ge=0, description="Inference time")
    total_ms: int = Field(..., ge=0, description="Total time")
    started_at: int = Field(..., description="Start timestamp")
    completed_at: int = Field(..., description="Completion timestamp")


class CaptchaSubmitRequest(BaseModel):
    """Request to submit CAPTCHA prediction."""

    session_id: str = Field(..., description="Session ID")
    task_id: str = Field(..., description="Task ID")
    prediction: PredictionData = Field(..., description="Prediction result")
    proof_of_work: ProofOfWorkData = Field(..., description="Proof of work")
    timing: TimingData = Field(..., description="Timing data")


class VerificationDisplayData(BaseModel):
    """Data for verification display."""

    type: str = Field(..., description="Display type: image or text")
    url: Optional[str] = Field(default=None, description="Content URL")
    content: Optional[str] = Field(default=None, description="Inline content")


class VerificationInfo(BaseModel):
    """Verification information for human verification."""

    verification_id: str = Field(..., description="Verification ID")
    display_data: VerificationDisplayData = Field(..., description="Display data")
    predicted_label: str = Field(..., description="Predicted label")
    prompt: str = Field(..., description="User prompt")
    options: List[Dict[str, str]] = Field(..., description="Available options")


class CaptchaSubmitResponse(BaseModel):
    """Response from CAPTCHA submission."""

    success: bool = Field(..., description="Submission success")
    requires_verification: bool = Field(..., description="Whether verification needed")
    verification: Optional[VerificationInfo] = Field(
        default=None, description="Verification data if needed"
    )
    captcha_token: Optional[str] = Field(
        default=None, description="CAPTCHA token if no verification needed"
    )
    expires_at: Optional[datetime] = Field(
        default=None, description="Token expiry time"
    )


class CaptchaValidateResponse(BaseModel):
    """Response from token validation."""

    valid: bool = Field(..., description="Token validity")
    session_id: Optional[str] = Field(default=None, description="Session ID")
    domain: Optional[str] = Field(default=None, description="Domain")
    completed_at: Optional[datetime] = Field(
        default=None, description="Completion time"
    )
    difficulty: Optional[str] = Field(default=None, description="Difficulty tier")
    verification_performed: Optional[bool] = Field(
        default=None, description="Whether verification was performed"
    )
