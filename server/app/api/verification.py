"""
Verification API endpoints.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import get_settings
from app.models import get_db, Session, Prediction, Verification
from app.schemas import VerificationSubmitRequest, VerificationSubmitResponse
from app.services.golden_dataset import GoldenDatasetService
from app.services.reputation import ReputationService
from app.utils.security import generate_captcha_token
from app.utils.redis_client import get_redis

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


@router.post("/captcha/verify", response_model=VerificationSubmitResponse)
async def submit_verification(
    request: VerificationSubmitRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Submit human verification response.

    1. Validate verification request
    2. Store verification response
    3. Update reputation
    4. Check for consensus
    5. Return CAPTCHA token
    """
    try:
        redis = await get_redis()

        # Validate verification request
        verification_data = await redis.get(f"verification:{request.verification_id}")
        if not verification_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Verification request not found or expired",
            )

        session_id_str, prediction_id_str = verification_data.decode().split(":")
        session_id = uuid.UUID(session_id_str)
        prediction_id = uuid.UUID(prediction_id_str)

        # Validate session matches
        if str(session_id) != request.session_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Session mismatch",
            )

        # Get session and prediction
        session = await _get_session(db, session_id)
        prediction = await _get_prediction(db, prediction_id)

        if not session or not prediction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session or prediction not found",
            )

        if session.is_expired:
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail="Session expired",
            )

        # Determine verified label
        verified_label = None
        if request.response == "correct" and request.corrected_label:
            verified_label = request.corrected_label
        elif request.response == "confirm":
            verified_label = prediction.predicted_label

        # Get user reputation
        reputation_service = ReputationService(db)
        reputation = await reputation_service.get_reputation(
            session.client_fingerprint or "anonymous"
        )

        # Store verification
        verification = Verification(
            prediction_id=prediction.id,
            sample_id=prediction.sample_id,
            session_id=session.id,
            response_type=request.response,
            original_label=prediction.predicted_label,
            verified_label=verified_label,
            response_time_ms=request.response_time_ms,
            reputation_score=reputation.score if reputation else 1.0,
        )
        db.add(verification)

        # Update golden dataset
        if verified_label:
            golden_service = GoldenDatasetService(db)
            await golden_service.process_verification(
                sample_id=prediction.sample_id,
                verified_label=verified_label,
                reputation_score=reputation.score if reputation else 1.0,
                domain=session.domain,
            )

        # Generate CAPTCHA token
        captcha_token = generate_captcha_token(
            session_id=str(session.id),
            domain=session.domain,
        )
        expires_at = datetime.utcnow() + timedelta(
            seconds=settings.captcha_token_expiry_seconds
        )

        # Update session status
        session.status = "completed"
        session.completed_at = datetime.utcnow()

        # Delete verification request from Redis
        await redis.delete(f"verification:{request.verification_id}")

        await db.commit()

        logger.info(f"Verification completed: {session.id}")

        return VerificationSubmitResponse(
            success=True,
            captcha_token=captcha_token,
            expires_at=expires_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error submitting verification: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit verification",
        )


async def _get_session(db: AsyncSession, session_id: uuid.UUID) -> Optional[Session]:
    """Get session by UUID."""
    result = await db.execute(
        select(Session).where(Session.id == session_id)
    )
    return result.scalar_one_or_none()


async def _get_prediction(
    db: AsyncSession, prediction_id: uuid.UUID
) -> Optional[Prediction]:
    """Get prediction by UUID."""
    result = await db.execute(
        select(Prediction).where(Prediction.id == prediction_id)
    )
    return result.scalar_one_or_none()
