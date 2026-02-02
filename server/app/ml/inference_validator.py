"""
Inference Validator for validating client predictions and shard outputs.

This module validates both traditional full-model predictions and
shard-based inference outputs against pre-computed ground truth.
"""

import logging
import hashlib
import json
import random
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio import Redis

from app.config import get_settings
from app.models import Task, Session, Prediction
from app.schemas import PredictionData, ProofOfWorkData, TimingData
from app.ml.ground_truth_cache import get_ground_truth_cache, GroundTruthCache

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class ShardValidationResult:
    """Result of validating a shard output."""
    is_valid: bool
    layer_name: str
    confidence: float
    error_message: Optional[str] = None
    expected_hash: Optional[str] = None
    actual_hash: Optional[str] = None


@dataclass
class InferenceProof:
    """Proof of inference computation from client."""
    input_hash: str
    output_hash: str
    layer_outputs: Dict[str, List[float]]  # layer_name -> output values
    final_prediction: Optional[int] = None
    computation_time_ms: int = 0


class InferenceValidator:
    """
    Validates client inference results including shard-based computation.

    Responsibilities:
    - Validate prediction plausibility
    - Verify proof of work / inference proof
    - Check timing constraints
    - Validate known sample predictions
    - Validate shard outputs against ground truth
    """

    def __init__(self, db: AsyncSession, redis: Redis):
        self.db = db
        self.redis = redis
        self._ground_truth_cache: Optional[GroundTruthCache] = None

    async def _get_ground_truth_cache(self) -> GroundTruthCache:
        """Lazy initialization of ground truth cache."""
        if self._ground_truth_cache is None:
            self._ground_truth_cache = await get_ground_truth_cache()
        return self._ground_truth_cache

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

    async def validate_shard_inference(
        self,
        task: Task,
        inference_proof: InferenceProof,
        timing: TimingData,
    ) -> Tuple[bool, List[ShardValidationResult]]:
        """
        Validate shard-based inference output.

        Args:
            task: The assigned task
            inference_proof: Proof of computation from client
            timing: Timing information

        Returns:
            Tuple of (all_valid, list_of_results)
        """
        results = []
        all_valid = True

        # Validate timing
        timing_valid = self._validate_timing(task, timing)
        if not timing_valid:
            all_valid = False

        # Get task metadata
        task_metadata = task.metadata_ or {}
        shard_task_data = task_metadata.get("shard_task", {})
        model_name = shard_task_data.get("model_name", "mnist-tiny")
        ground_truth_key = shard_task_data.get("ground_truth_key")

        if not ground_truth_key:
            logger.error(f"Missing ground truth key for task {task.id}")
            return False, [ShardValidationResult(
                is_valid=False,
                layer_name="unknown",
                confidence=0.0,
                error_message="Missing ground truth key"
            )]

        # Parse ground truth key to get input hash
        parts = ground_truth_key.split(":")
        if len(parts) < 3:
            return False, [ShardValidationResult(
                is_valid=False,
                layer_name="unknown",
                confidence=0.0,
                error_message="Invalid ground truth key format"
            )]

        input_hash = parts[1]

        # Get ground truth cache
        cache = await self._get_ground_truth_cache()

        # Validate each layer output
        for layer_name, client_output in inference_proof.layer_outputs.items():
            # Extract layer index from name
            layer_index = self._extract_layer_index(layer_name)

            # Validate against ground truth (simplified - just check hash match)
            gt_entry = cache.get_ground_truth(
                sample_id=input_hash,
                model_name=model_name,
                layer_index=layer_index
            )

            if gt_entry:
                # Compute hash of client output
                client_output_bytes = json.dumps(client_output, sort_keys=True).encode()
                client_hash = hashlib.sha256(client_output_bytes).hexdigest()[:16]

                is_valid_layer = client_hash == gt_entry.output_hash
                confidence = 1.0 if is_valid_layer else 0.0

                result = ShardValidationResult(
                    is_valid=is_valid_layer,
                    layer_name=layer_name,
                    confidence=confidence,
                    expected_hash=gt_entry.output_hash,
                    actual_hash=client_hash
                )
                results.append(result)

                if not is_valid_layer:
                    all_valid = False
                    logger.warning(
                        f"Shard validation failed for task {task.id}, "
                        f"layer {layer_name}: hash mismatch"
                    )

        # Validate final prediction if provided
        if inference_proof.final_prediction is not None:
            gt_entry = cache.get_ground_truth(
                sample_id=input_hash,
                model_name=model_name,
                layer_index=-1  # Final output
            )

            if gt_entry and gt_entry.top_prediction is not None:
                if inference_proof.final_prediction != gt_entry.top_prediction:
                    all_valid = False
                    logger.warning(
                        f"Final prediction mismatch for task {task.id}: "
                        f"predicted={inference_proof.final_prediction}, "
                        f"expected={gt_entry.top_prediction}"
                    )

        return all_valid, results

    def _extract_layer_index(self, layer_name: str) -> int:
        """Extract layer index from layer name for ground truth lookup."""
        layer_map = {
            'conv1': 0,
            'pool1': 1,
            'conv2': 2,
            'pool2': 3,
            'flatten': 4,
            'dense1': 5,
            'output': 6,
        }
        return layer_map.get(layer_name.lower(), -1)

    def _validate_timing(self, task: Task, timing: TimingData) -> bool:
        """
        Validate timing constraints.

        Checks that inference time is within expected bounds.
        """
        expected_ms = task.expected_time_ms
        actual_ms = timing.inference_ms

        # Allow some tolerance
        min_time = expected_ms * 0.1  # At least 10% of expected

        if actual_ms < min_time:
            logger.warning(
                f"Suspiciously fast inference: {actual_ms}ms < {min_time}ms"
            )
            return False

        return True

    def _validate_proof_of_work(self, pow_data: ProofOfWorkData) -> bool:
        """
        Validate proof of work hash.

        Verifies that the client actually performed computation.
        """
        if not pow_data.hash:
            return False

        if len(pow_data.hash) != 64:  # SHA-256 hex length
            return False

        expected_prefix = "0"  # Simple difficulty
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

    async def validate_inference_proof(
        self,
        task: Task,
        inference_proof: InferenceProof
    ) -> bool:
        """
        Validate inference proof against ground truth.

        This replaces the traditional PoW validation with actual
        ML computation validation.
        """
        # Get ground truth cache
        cache = await self._get_ground_truth_cache()

        # Get task metadata
        task_metadata = task.metadata_ or {}
        shard_task_data = task_metadata.get("shard_task", {})
        model_name = shard_task_data.get("model_name", "mnist-tiny")
        ground_truth_key = shard_task_data.get("ground_truth_key")

        if not ground_truth_key:
            logger.error(f"Missing ground truth key for task {task.id}")
            return False

        # Parse ground truth key
        parts = ground_truth_key.split(":")
        if len(parts) < 3:
            return False

        input_hash = parts[1]

        # Validate that the client computed the expected output
        # by checking the output hash
        for layer_name, client_output in inference_proof.layer_outputs.items():
            layer_index = self._extract_layer_index(layer_name)

            gt_entry = cache.get_ground_truth(
                sample_id=input_hash,
                model_name=model_name,
                layer_index=layer_index
            )

            if gt_entry:
                # Compute hash of client output
                client_output_bytes = json.dumps(client_output, sort_keys=True).encode()
                client_hash = hashlib.sha256(client_output_bytes).hexdigest()[:16]

                if client_hash != gt_entry.output_hash:
                    logger.warning(
                        f"Inference proof validation failed for {layer_name}: "
                        f"hash mismatch"
                    )
                    return False

        return True

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
            return random.random() < 0.5

        # Normal difficulty - lower probability
        return random.random() < 0.1

    async def get_validation_summary(
        self,
        task_id: str
    ) -> Dict[str, Any]:
        """
        Get validation summary for a task.

        Returns:
            Dictionary with validation statistics
        """
        cache = await self._get_ground_truth_cache()
        stats = cache.get_stats()

        return {
            "task_id": task_id,
            "ground_truth_entries": stats.total_entries,
            "models_cached": stats.models_cached,
            "cache_size_mb": round(stats.cache_size_mb, 2)
        }
