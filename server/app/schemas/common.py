"""
Common schemas used across the API.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional error details"
    )


class SuccessResponse(BaseModel):
    """Standard success response."""

    success: bool = Field(default=True, description="Operation success status")
    message: Optional[str] = Field(default=None, description="Optional message")
