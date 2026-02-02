"""
Verification API schemas.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class VerificationSubmitRequest(BaseModel):
    """Request to submit human verification."""

    session_id: str = Field(..., description="Session ID")
    verification_id: str = Field(..., description="Verification ID")
    response: str = Field(
        ..., description="Response type: confirm, reject, correct"
    )
    corrected_label: Optional[str] = Field(
        default=None, description="Corrected label if response is 'correct'"
    )
    response_time_ms: Optional[int] = Field(
        default=None, ge=0, description="Time taken to respond"
    )


class VerificationSubmitResponse(BaseModel):
    """Response from verification submission."""

    success: bool = Field(..., description="Verification success")
    captcha_token: str = Field(..., description="CAPTCHA token")
    expires_at: datetime = Field(..., description="Token expiry time")


class VerificationData(BaseModel):
    """Data for creating a verification request."""

    verification_id: str
    display_type: str
    display_content: str
    predicted_label: str
    prompt: str
