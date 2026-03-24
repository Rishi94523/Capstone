"""
CAPTCHA API endpoints.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Header, status, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import get_settings
from app.models import get_db, Session, Task, Sample, Prediction
from app.schemas import (
    CaptchaInitRequest,
    CaptchaInitResponse,
    CaptchaSubmitRequest,
    CaptchaSubmitResponse,
    CaptchaValidateResponse,
    TaskInfo,
    ShardTaskInfo,
    ModelMeta,
    VerificationInfo,
    VerificationDisplayData,
    VerificationOption,
)
from app.core.task_coordinator import TaskCoordinator
from app.core.risk_scorer import RiskScorer
from app.ml.inference_validator import InferenceValidator
from app.utils.security import create_jwt_token, verify_jwt_token, generate_captcha_token
from app.utils.redis_client import get_redis

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()

# In-memory inference log for dashboard visibility
inference_log = []


@router.post("/captcha/init", response_model=CaptchaInitResponse)
async def init_captcha(
    request: CaptchaInitRequest,
    db: AsyncSession = Depends(get_db),
    x_forwarded_for: Optional[str] = Header(default=None),
):
    """
    Initialize a new CAPTCHA session.

    1. Validate site key
    2. Compute risk score
    3. Assign difficulty tier
    4. Select ML task
    5. Return session with task
    """
    try:
        # Get client IP for rate limiting
        client_ip = x_forwarded_for.split(",")[0].strip() if x_forwarded_for else "unknown"

        # Initialize services
        redis = await get_redis()
        risk_scorer = RiskScorer(redis)
        task_coordinator = TaskCoordinator(db, redis)

        # Compute risk score
        risk_score = await risk_scorer.compute_risk_score(
            client_ip=client_ip,
            user_agent=request.client_metadata.user_agent,
            site_key=request.site_key,
        )

        # Determine difficulty tier
        difficulty = task_coordinator.get_difficulty_tier(risk_score)

        # Create session
        session_token = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(seconds=settings.captcha_token_expiry_seconds)

        session = Session(
            domain=_extract_domain(request.site_key),
            session_token=session_token,
            risk_score=risk_score,
            difficulty_tier=difficulty,
            status="pending",
            expires_at=expires_at,
        )
        db.add(session)
        await db.flush()

        # Assign task
        task, sample, shard_task = await task_coordinator.assign_task(
            session_id=session.id,
            difficulty=difficulty,
        )

        # Create challenge token
        challenge_token = create_jwt_token(
            data={
                "session_id": str(session.id),
                "task_id": str(task.id),
                "difficulty": difficulty,
            },
            expires_delta=timedelta(seconds=settings.captcha_token_expiry_seconds),
        )

        # Build response
        if task.task_type == "shard_inference":
            task_info = ShardTaskInfo(
                task_id=str(task.id),
                sample_id=shard_task.sample_id,
                model_name=shard_task.model_name,
                model_version=shard_task.model_version,
                shards=shard_task.shards,
                input_data=shard_task.input_data,
                input_shape=shard_task.input_shape,
                expected_layers=shard_task.expected_layers,
                difficulty=shard_task.difficulty,
                expected_time_ms=shard_task.expected_time_ms,
                ground_truth_key=shard_task.ground_truth_key,
                labels=shard_task.labels,
                model_checksum=shard_task.model_checksum,
            )
        else:
            task_info = TaskInfo(
                task_id=str(task.id),
                model_url=f"{settings.model_cdn_url}/{settings.default_model}/model.json",
                sample_data=_encode_sample_data(sample),
                sample_url=sample.data_url,
                sample_type=sample.data_type,
                task_type=task.task_type,
                expected_time_ms=task.expected_time_ms,
                model_meta=ModelMeta(
                    name=settings.default_model,
                    version="1.0.0",
                    input_shape=[1, 32, 32, 3],
                    labels=_get_model_labels(settings.default_model),
                    checksum="abc123",
                ),
            )

        await db.commit()

        logger.info(f"Session initialized: {session.id}, difficulty: {difficulty}")

        return CaptchaInitResponse(
            session_id=str(session.id),
            challenge_token=challenge_token,
            task=task_info,
            difficulty=difficulty,
            expires_at=expires_at,
        )

    except Exception as e:
        logger.exception(f"Error initializing CAPTCHA: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialize CAPTCHA",
        )


@router.post("/captcha/submit", response_model=CaptchaSubmitResponse)
async def submit_captcha(
    request: CaptchaSubmitRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Submit CAPTCHA prediction result.

    1. Validate session
    2. Validate prediction
    3. Verify proof of work
    4. Determine if verification needed
    5. Return result or verification request
    """
    try:
        redis = await get_redis()
        validator = InferenceValidator(db, redis)

        # Validate session
        session = await _get_session(db, request.session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found",
            )

        if session.is_expired:
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail="Session expired",
            )

        # Validate task
        task = await _get_task(db, request.task_id, session.id)
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found",
            )

        # Validate prediction
        is_valid = await validator.validate_prediction(
            task=task,
            prediction=request.prediction,
            proof_of_work=request.proof_of_work,
            timing=request.timing,
            inference_proof=request.proof,
        )

        # Store prediction
        prediction = Prediction(
            task_id=task.id,
            session_id=session.id,
            sample_id=task.sample_id,
            predicted_label=request.prediction.label,
            confidence=request.prediction.confidence,
            inference_time_ms=request.timing.inference_ms,
            pow_hash=(request.proof.proof_hash if request.proof else request.proof_of_work.hash),
            is_valid=is_valid,
        )
        db.add(prediction)
        await db.flush()

        # Log inference for dashboard visibility
        # Query sample separately to avoid async lazy-load issue
        sample_result = await db.execute(select(Sample).where(Sample.id == task.sample_id))
        sample_for_log = sample_result.scalar_one_or_none()
        
        # Build image URL - use our endpoint if blob exists, otherwise use data_url
        image_url = None
        if sample_for_log:
            if sample_for_log.data_blob:
                image_url = f"/api/v1/sample/{task.sample_id}/image"
            elif sample_for_log.data_url:
                image_url = sample_for_log.data_url
        
        inference_record = {
            "id": str(prediction.id),
            "session_id": str(session.id),
            "task_id": str(task.id),
            "sample_id": str(task.sample_id),
            "image_url": image_url,
            "predicted_label": request.prediction.label,
            "confidence": request.prediction.confidence,
            "top_k": [{"label": p.label, "confidence": p.confidence} for p in (request.prediction.top_k or [])],
            "inference_ms": request.timing.inference_ms,
            "total_ms": request.timing.total_ms,
            "timestamp": datetime.utcnow().isoformat(),
            "is_valid": is_valid,
        }
        inference_log.append(inference_record)
        # Keep only last 500 records
        if len(inference_log) > 500:
            inference_log.pop(0)
        logger.info(f"Logged inference: {inference_record['predicted_label']} (conf: {inference_record['confidence']:.2f})")

        if not is_valid:
            session.status = "failed"
            task.status = "failed"
            await db.commit()

            logger.warning(f"CAPTCHA validation failed: {session.id}")

            return CaptchaSubmitResponse(
                success=False,
                requires_verification=False,
            )

        # Determine if verification is needed
        requires_verification = await validator.should_require_verification(
            session=session,
            prediction=prediction,
        )

        if requires_verification:
            # Create verification request
            verification_id = str(uuid.uuid4())
            # Reuse the sample we already queried
            sample = sample_for_log

            # Store verification request in Redis
            await redis.setex(
                f"verification:{verification_id}",
                settings.captcha_token_expiry_seconds,
                f"{session.id}:{prediction.id}",
            )

            session.status = "verifying"
            await db.commit()

            return CaptchaSubmitResponse(
                success=True,
                requires_verification=True,
                verification=VerificationInfo(
                    verification_id=verification_id,
                    display_data=VerificationDisplayData(
                        type=sample.data_type,
                        url=sample.data_url,
                        content=_encode_sample_data(sample) if not sample.data_url else None,
                    ),
                    predicted_label=request.prediction.label,
                    prompt=f"Is this a {request.prediction.label}?",
                    options=[
                        VerificationOption(id="confirm", label="Yes, correct", type="confirm"),
                        VerificationOption(id="correct", label="Choose correct label", type="correct"),
                    ],
                ),
            )
        else:
            # Generate CAPTCHA token
            captcha_token = generate_captcha_token(
                session_id=str(session.id),
                domain=session.domain,
            )
            expires_at = datetime.utcnow() + timedelta(
                seconds=settings.captcha_token_expiry_seconds
            )

            session.status = "completed"
            session.completed_at = datetime.utcnow()
            task.status = "completed"
            await db.commit()

            logger.info(f"CAPTCHA completed: {session.id}")

            return CaptchaSubmitResponse(
                success=True,
                requires_verification=False,
                captcha_token=captcha_token,
                expires_at=expires_at,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error submitting CAPTCHA: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit CAPTCHA",
        )


@router.get("/captcha/validate/{token}", response_model=CaptchaValidateResponse)
async def validate_captcha(
    token: str,
    db: AsyncSession = Depends(get_db),
    x_pouw_site_key: Optional[str] = Header(default=None),
):
    """
    Validate a CAPTCHA token (server-to-server).

    This endpoint is called by the website's backend to verify
    that a CAPTCHA token is valid.
    """
    try:
        # Verify token
        payload = verify_jwt_token(token)
        if not payload:
            return CaptchaValidateResponse(valid=False)

        session_id = payload.get("session_id")
        if not session_id:
            return CaptchaValidateResponse(valid=False)

        # Get session
        session = await _get_session(db, session_id)
        if not session or session.status != "completed":
            return CaptchaValidateResponse(valid=False)

        # Check expiry
        token_exp = payload.get("exp")
        if token_exp and datetime.fromtimestamp(token_exp) < datetime.utcnow():
            return CaptchaValidateResponse(valid=False)

        return CaptchaValidateResponse(
            valid=True,
            session_id=str(session.id),
            domain=session.domain,
            completed_at=session.completed_at,
            difficulty=session.difficulty_tier,
            verification_performed=len(session.verifications) > 0,
        )

    except Exception as e:
        logger.exception(f"Error validating CAPTCHA: {e}")
        return CaptchaValidateResponse(valid=False)


# Helper functions

def _extract_domain(site_key: str) -> str:
    """Extract domain from site key (simplified)."""
    # In production, this would validate against DomainConfig
    return "example.com"


def _encode_sample_data(sample: Sample) -> Optional[str]:
    """Encode sample data to base64."""
    import base64

    if sample.data_blob:
        return base64.b64encode(sample.data_blob).decode("utf-8")
    return None


def _get_model_labels(model_name: str) -> list:
    """Get labels for a model."""
    labels = {
        "cifar10-mobilenet": [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ],
        "imdb-distilbert": ["negative", "positive"],
        "mnist-tiny": [str(i) for i in range(10)],
    }
    return labels.get(model_name, [])


async def _get_session(db: AsyncSession, session_id: str) -> Optional[Session]:
    """Get session by ID."""
    try:
        session_uuid = uuid.UUID(session_id)
        result = await db.execute(
            select(Session).where(Session.id == session_uuid)
        )
        return result.scalar_one_or_none()
    except ValueError:
        return None


async def _get_task(
    db: AsyncSession, task_id: str, session_id: uuid.UUID
) -> Optional[Task]:
    """Get task by ID and session."""
    try:
        task_uuid = uuid.UUID(task_id)
        result = await db.execute(
            select(Task).where(
                Task.id == task_uuid,
                Task.session_id == session_id,
            )
        )
        return result.scalar_one_or_none()
    except ValueError:
        return None


@router.get("/sample/{sample_id}/image")
async def get_sample_image(
    sample_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Serve sample image from database blob."""
    try:
        sample_uuid = uuid.UUID(sample_id)
        result = await db.execute(
            select(Sample).where(Sample.id == sample_uuid)
        )
        sample = result.scalar_one_or_none()
        
        if not sample:
            raise HTTPException(status_code=404, detail="Sample not found")
        
        if sample.data_blob:
            # Determine content type
            content_type = "image/png"
            if sample.data_type == "image":
                # Check magic bytes for format
                if sample.data_blob[:3] == b'\xff\xd8\xff':
                    content_type = "image/jpeg"
                elif sample.data_blob[:8] == b'\x89PNG\r\n\x1a\n':
                    content_type = "image/png"
            
            return Response(content=sample.data_blob, media_type=content_type)
        elif sample.data_url:
            # Redirect to URL
            from fastapi.responses import RedirectResponse
            return RedirectResponse(url=sample.data_url)
        else:
            raise HTTPException(status_code=404, detail="No image data available")
            
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid sample ID")
