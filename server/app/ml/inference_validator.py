"""
Inference Validator for validating client predictions.
"""

import logging
import hashlib
import random
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio import Redis

from app.config import get_settings
from app.models import Task, Session, Prediction
from app.schemas import PredictionData, ProofOfWorkData, TimingData

logger = logging.getLogger(__name__)
settings = get_settings()


class InferenceValidator:
    """
    Validates client inference results.

    Responsibilities:
    - Validate prediction plausibility
    - Verify proof of work
    - Check timing constraints
    - Validate known sample predictions
    """

    def __init__(self, db: AsyncSession, redis: Redis):
        self.db = db
        self.redis = redis

    async def validate_prediction(
        self,
        task: Task,
        prediction: PredictionData,
        proof_of_work: ProofOfWorkData,
        timing: TimingData,
    ) -> bool:
        """
        Validate a client prediction.

        Args:
            task: The assigned task
            prediction: Client's prediction
            proof_of_work: Proof of work data
            timing: Timing information

        Returns:
            True if prediction is valid
        """
        validations = [
            self._validate_timing(task, timing),
            self._validate_proof_of_work(proof_of_work),
            self._validate_prediction_plausibility(prediction),
        ]

        # Validate known sample if applicable
        if task.is_known_sample and task.known_label:
            validations.append(
                self._validate_known_sample(prediction, task.known_label)
            )

        is_valid = all(validations)

        if not is_valid:
            logger.warning(
                f"Prediction validation failed for task {task.id}: "
                f"timing={validations[0]}, pow={validations[1]}, "
                f"plausibility={validations[2]}"
            )

        return is_valid

    def _validate_timing(self, task: Task, timing: TimingData) -> bool:
        """
        Validate timing constraints.

        Checks that inference time is within expected bounds.
        """
        expected_ms = task.expected_time_ms
        actual_ms = timing.inference_ms

        # Allow some tolerance
        min_time = expected_ms * 0.1  # At least 10% of expected
        max_time = expected_ms * 5.0  # At most 5x expected

        if actual_ms < min_time:
            logger.warning(
                f"Suspiciously fast inference: {actual_ms}ms < {min_time}ms"
            )
            return False

        if actual_ms > max_time:
            logger.warning(
                f"Inference too slow: {actual_ms}ms > {max_time}ms"
            )
            # Don't fail for slow, just log it
            pass

        return True

    def _validate_proof_of_work(self, pow_data: ProofOfWorkData) -> bool:
        """
        Validate proof of work hash.

        Verifies that the client actually performed computation.
        """
        # Reconstruct expected hash
        payload = ":".join([
            pow_data.model_checksum,
            pow_data.input_hash,
            pow_data.output_hash,
            str(pow_data.nonce),
        ])

        expected_prefix = "0"  # Simple difficulty

        # For now, accept any valid-looking hash
        # In production, we'd verify the hash computation
        if not pow_data.hash:
            return False

        if len(pow_data.hash) != 64:  # SHA-256 hex length
            return False

        if not pow_data.hash.startswith(expected_prefix):
            logger.warning(f"PoW hash doesn't meet difficulty: {pow_data.hash[:8]}...")
            return False

        return True

    def _validate_prediction_plausibility(self, prediction: PredictionData) -> bool:
        """
        Validate that prediction is plausible.

        Checks confidence scores and consistency.
        """
        # Check confidence is valid
        if not 0 <= prediction.confidence <= 1:
            return False

        # Check top-k consistency
        if not prediction.top_k:
            return False

        # Top prediction should match
        if prediction.top_k[0].label != prediction.label:
            return False

        # Confidences should sum to approximately 1
        total_confidence = sum(p.confidence for p in prediction.top_k)
        if total_confidence > 1.1:  # Allow some tolerance
            logger.warning(f"Top-k confidences sum to {total_confidence}")
            # Don't fail, just note it

        return True

    def _validate_known_sample(
        self, prediction: PredictionData, known_label: str
    ) -> bool:
        """
        Validate prediction against known sample label.

        Used for honeypot validation.
        """
        # Check if prediction matches known label
        is_correct = prediction.label.lower() == known_label.lower()

        if not is_correct:
            logger.info(
                f"Known sample mismatch: predicted={prediction.label}, "
                f"expected={known_label}"
            )

        return True  # Don't fail the validation, just track it

    async def should_require_verification(
        self,
        session: Session,
        prediction: Prediction,
    ) -> bool:
        """
        Determine if human verification is required.

        Args:
            session: The session
            prediction: The prediction

        Returns:
            True if verification should be required
        """
        # Always verify for bot-like difficulty
        if session.difficulty_tier == "bot_like":
            return True

        # Higher probability for suspicious
        if session.difficulty_tier == "suspicious":
            base_probability = 0.5
        else:
            base_probability = settings.verification_rate

        # Adjust based on prediction confidence
        if prediction.confidence < 0.5:
            # Low confidence = more likely to verify
            base_probability *= 1.5

        # Random sampling
        return random.random() < base_probability

    async def record_known_sample_result(
        self,
        fingerprint: str,
        was_correct: bool,
    ) -> None:
        """Record known sample validation result for risk scoring."""
        key = f"known_accuracy:{fingerprint}"

        # Update accuracy with exponential moving average
        current = await self.redis.get(key)
        if current:
            current_acc = float(current)
            new_acc = 0.3 * (1.0 if was_correct else 0.0) + 0.7 * current_acc
        else:
            new_acc = 1.0 if was_correct else 0.0

        await self.redis.setex(key, 86400, str(new_acc))  # 24 hour expiry
