"""CAPTCHA API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


def to_camel(value: str) -> str:
    parts = value.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


class APIModel(BaseModel):
    """Base schema with snake_case input and camelCase serialization support."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )


class ClientMetadata(APIModel):
    user_agent: str = Field(..., description="Browser user agent")
    language: str = Field(..., description="Browser language")
    timezone: str = Field(..., description="Client timezone")
    screen_width: Optional[int] = Field(default=None, description="Screen width")
    screen_height: Optional[int] = Field(default=None, description="Screen height")


class CaptchaInitRequest(APIModel):
    site_key: str = Field(..., min_length=10, description="Site API key")
    client_metadata: ClientMetadata = Field(..., description="Client information")


class ModelMeta(APIModel):
    name: str
    version: str
    input_shape: List[int]
    labels: List[str]
    checksum: str


class NeuralLayerConfig(APIModel):
    name: str
    type: str
    weights: List[float]
    biases: List[float]
    input_shape: List[int]
    output_shape: List[int]
    activation: str


class ModelShardInfo(APIModel):
    index: int
    name: str
    layer_type: str
    input_shape: List[int]
    output_shape: List[int]
    activation: Optional[str] = None
    layers: List[NeuralLayerConfig] = Field(default_factory=list)


class TaskInfo(APIModel):
    task_id: str
    model_url: str
    sample_data: Optional[str] = None
    sample_url: Optional[str] = None
    sample_type: str
    task_type: str
    expected_time_ms: int
    model_meta: ModelMeta


class ShardTaskInfo(APIModel):
    task_id: str
    sample_id: str
    model_name: str
    model_version: str
    shards: List[ModelShardInfo]
    input_data: str
    input_shape: List[int]
    expected_layers: int
    difficulty: str
    expected_time_ms: int
    ground_truth_key: str
    labels: List[str]
    model_checksum: str


class CaptchaInitResponse(APIModel):
    session_id: str
    challenge_token: str
    task: TaskInfo | ShardTaskInfo
    difficulty: str
    expires_at: datetime


class TopKPrediction(APIModel):
    label: str
    confidence: float = Field(..., ge=0, le=1)


class PredictionData(APIModel):
    label: str
    confidence: float = Field(..., ge=0, le=1)
    top_k: List[TopKPrediction]


class ProofOfWorkData(APIModel):
    hash: str = Field(..., min_length=64)
    nonce: int = Field(..., ge=0)
    model_checksum: str
    input_hash: str
    output_hash: str


class InferenceProofData(APIModel):
    task_id: str
    sample_id: str
    layer_count: int = Field(..., ge=1)
    output_hashes: List[str] = Field(..., min_length=1)
    prediction_hash: str
    proof_hash: str
    timestamp: int


class TimingData(APIModel):
    model_load_ms: int = Field(..., ge=0)
    inference_ms: int = Field(..., ge=0)
    total_ms: int = Field(..., ge=0)
    started_at: int
    completed_at: int


class CaptchaSubmitRequest(APIModel):
    session_id: str
    task_id: str
    prediction: PredictionData
    proof_of_work: Optional[ProofOfWorkData] = None
    proof: Optional[InferenceProofData] = None
    timing: TimingData

    @model_validator(mode="after")
    def validate_proof_variant(self) -> "CaptchaSubmitRequest":
        if self.proof_of_work is None and self.proof is None:
            raise ValueError("Either proof_of_work or proof must be provided")
        return self


class VerificationDisplayData(APIModel):
    type: str
    url: Optional[str] = None
    content: Optional[str] = None


class VerificationOption(APIModel):
    id: str
    label: str
    type: str


class VerificationInfo(APIModel):
    verification_id: str
    display_data: VerificationDisplayData
    predicted_label: str
    prompt: str
    options: List[VerificationOption]


class CaptchaSubmitResponse(APIModel):
    success: bool
    requires_verification: bool
    verification: Optional[VerificationInfo] = None
    captcha_token: Optional[str] = None
    expires_at: Optional[datetime] = None


class CaptchaValidateResponse(APIModel):
    valid: bool
    session_id: Optional[str] = None
    domain: Optional[str] = None
    completed_at: Optional[datetime] = None
    difficulty: Optional[str] = None
    verification_performed: Optional[bool] = None
