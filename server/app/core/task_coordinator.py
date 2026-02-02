"""
Task Coordinator for assigning ML tasks to sessions.

This module coordinates task assignment with support for shard-based
federated inference, replacing the previous full-model approach.
"""

import logging
import random
import uuid
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio import Redis

from app.config import get_settings
from app.models import Sample, Task
from app.ml.shard_manager import get_shard_manager, ShardAssignment
from app.ml.ground_truth_cache import get_ground_truth_cache

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class ShardTask:
    """Task configuration for shard-based inference."""
    task_id: uuid.UUID
    model_name: str
    model_version: str
    shards: list
    input_data: list
    input_shape: list
    expected_layers: int
    difficulty: str
    expected_time_ms: int
    ground_truth_key: str


class TaskCoordinator:
    """
    Coordinates ML task assignment based on risk and difficulty.

    Responsibilities:
    - Assign appropriate tasks based on difficulty tier
    - Balance workload across sample pool
    - Inject known samples for validation
    - Distribute model shards for federated inference
    """

    # Difficulty tier configurations with shard-based inference
    DIFFICULTY_TIERS = {
        "normal": {
            "risk_score_max": 0.3,
            "inference_time_ms": 20,
            "task_type": "shard_inference",
            "verification_probability": 0.2,
            "shard_difficulty": "easy",
            "layers": 1,
        },
        "suspicious": {
            "risk_score_max": 0.7,
            "inference_time_ms": 100,
            "task_type": "shard_inference",
            "verification_probability": 0.5,
            "shard_difficulty": "medium",
            "layers": 3,
        },
        "bot_like": {
            "risk_score_max": 1.0,
            "inference_time_ms": 200,
            "task_type": "shard_inference",
            "verification_probability": 1.0,
            "shard_difficulty": "hard",
            "layers": 6,
        },
    }

    # Default model for shard-based inference
    DEFAULT_MODEL = "mnist-tiny"

    def __init__(self, db: AsyncSession, redis: Redis):
        self.db = db
        self.redis = redis
        self._shard_manager = None
        self._ground_truth_cache = None

    async def _get_shard_manager(self):
        """Lazy initialization of shard manager."""
        if self._shard_manager is None:
            self._shard_manager = await get_shard_manager()
        return self._shard_manager

    async def _get_ground_truth_cache(self):
        """Lazy initialization of ground truth cache."""
        if self._ground_truth_cache is None:
            self._ground_truth_cache = await get_ground_truth_cache()
        return self._ground_truth_cache

    def get_difficulty_tier(self, risk_score: float) -> str:
        """
        Determine difficulty tier based on risk score.

        Args:
            risk_score: Computed risk score (0.0 - 1.0)

        Returns:
            Difficulty tier: 'normal', 'suspicious', or 'bot_like'
        """
        if risk_score <= self.DIFFICULTY_TIERS["normal"]["risk_score_max"]:
            return "normal"
        elif risk_score <= self.DIFFICULTY_TIERS["suspicious"]["risk_score_max"]:
            return "suspicious"
        else:
            return "bot_like"

    async def assign_shard_task(
        self,
        session_id: uuid.UUID,
        difficulty: str,
    ) -> ShardTask:
        """
        Assign a shard-based inference task to a session.

        Args:
            session_id: Session UUID
            difficulty: Difficulty tier

        Returns:
            ShardTask configuration
        """
        tier_config = self.DIFFICULTY_TIERS.get(difficulty, self.DIFFICULTY_TIERS["normal"])
        shard_difficulty = tier_config.get("shard_difficulty", "easy")

        # Get shard manager
        shard_manager = await self._get_shard_manager()

        # Generate task ID
        task_id = uuid.uuid4()

        # Get available models
        available_models = shard_manager.get_available_models()
        model_name = self.DEFAULT_MODEL if self.DEFAULT_MODEL in available_models else available_models[0] if available_models else self.DEFAULT_MODEL

        # Assign shards based on difficulty
        shard_assignment = await shard_manager.assign_shards(
            task_id=str(task_id),
            model_name=model_name,
            difficulty=shard_difficulty
        )

        # Create ground truth key for validation
        input_hash = hash(tuple(shard_assignment.input_data)) % (2**32)
        ground_truth_key = f"{model_name}:{input_hash}:{shard_assignment.expected_layers - 1}"

        # Create shard task
        shard_task = ShardTask(
            task_id=task_id,
            model_name=model_name,
            model_version=shard_assignment.model_version,
            shards=[s.to_dict() for s in shard_assignment.shards],
            input_data=shard_assignment.input_data,
            input_shape=shard_assignment.input_shape,
            expected_layers=shard_assignment.expected_layers,
            difficulty=shard_difficulty,
            expected_time_ms=tier_config["inference_time_ms"],
            ground_truth_key=ground_truth_key
        )

        # Store task in database
        task = Task(
            session_id=session_id,
            task_type=tier_config["task_type"],
            expected_time_ms=tier_config["inference_time_ms"],
            is_known_sample=False,
            status="assigned",
            metadata_={
                "shard_task": {
                    "task_id": str(shard_task.task_id),
                    "model_name": shard_task.model_name,
                    "model_version": shard_task.model_version,
                    "expected_layers": shard_task.expected_layers,
                    "difficulty": shard_task.difficulty,
                    "ground_truth_key": shard_task.ground_truth_key
                }
            }
        )

        self.db.add(task)
        await self.db.flush()

        logger.debug(
            f"Assigned shard task {task_id} with {shard_assignment.expected_layers} layers "
            f"(difficulty: {shard_difficulty})"
        )

        return shard_task

    async def assign_task(
        self,
        session_id: uuid.UUID,
        difficulty: str,
    ) -> Tuple[Task, Sample]:
        """
        Assign an ML task to a session.

        Args:
            session_id: Session UUID
            difficulty: Difficulty tier

        Returns:
            Tuple of (Task, Sample)
        """
        tier_config = self.DIFFICULTY_TIERS.get(difficulty, self.DIFFICULTY_TIERS["normal"])

        # Determine if we should use a known sample (honeypot)
        use_known_sample = random.random() < settings.known_sample_rate

        # Select sample
        sample = await self._select_sample(use_known_sample)

        if not sample:
            # Create a dummy sample if none exist
            sample = await self._create_dummy_sample()

        # Create task with shard-based configuration
        task = Task(
            session_id=session_id,
            sample_id=sample.id,
            task_type=tier_config["task_type"],
            expected_time_ms=tier_config["inference_time_ms"],
            is_known_sample=use_known_sample,
            known_label=sample.metadata_.get("known_label") if use_known_sample else None,
            status="assigned",
            metadata_={
                "shard_difficulty": tier_config.get("shard_difficulty", "easy"),
                "expected_layers": tier_config.get("layers", 1),
            }
        )

        self.db.add(task)

        # Increment times served
        sample.times_served += 1

        await self.db.flush()

        logger.debug(f"Assigned task {task.id} with sample {sample.id}")

        return task, sample

    async def _select_sample(self, use_known: bool = False) -> Optional[Sample]:
        """
        Select a sample for the task.

        Prioritizes samples with fewer serves for balanced coverage.
        """
        query = select(Sample)

        if use_known:
            # Select known samples (samples with known_label in metadata)
            query = query.where(
                Sample.metadata_["known_label"].isnot(None)
            )
        else:
            # Select regular samples, prioritize less-served ones
            query = query.order_by(Sample.times_served.asc())

        query = query.limit(10)

        result = await self.db.execute(query)
        samples = result.scalars().all()

        if not samples:
            return None

        # Random selection from least-served samples
        return random.choice(samples)

    async def _create_dummy_sample(self) -> Sample:
        """
        Create a dummy sample for testing when no samples exist.
        """
        import hashlib
        import os

        # Generate random data
        random_data = os.urandom(1024)
        data_hash = hashlib.sha256(random_data).hexdigest()

        sample = Sample(
            data_type="image",
            model_type="mnist",
            data_hash=data_hash,
            data_blob=random_data,
            metadata_={
                "width": 28,
                "height": 28,
                "channels": 1,
                "is_dummy": True,
            },
        )

        self.db.add(sample)
        await self.db.flush()

        logger.info(f"Created dummy sample: {sample.id}")

        return sample

    async def get_task_stats(self) -> dict:
        """Get task assignment statistics."""
        # Count by difficulty
        result = await self.db.execute(
            select(
                Task.task_type,
                func.count(Task.id).label("count"),
            ).group_by(Task.task_type)
        )

        stats = {row.task_type: row.count for row in result}

        return {
            "task_counts": stats,
            "total_tasks": sum(stats.values()),
        }

    async def validate_shard_output(
        self,
        task_id: uuid.UUID,
        layer_index: int,
        client_output: list
    ) -> Tuple[bool, str]:
        """
        Validate client shard output against ground truth.

        Args:
            task_id: Task identifier
            layer_index: Which layer output to validate
            client_output: Output from client

        Returns:
            Tuple of (is_valid, message)
        """
        # Get ground truth cache
        cache = await self._get_ground_truth_cache()

        # Get task from database to retrieve ground truth key
        result = await self.db.execute(
            select(Task).where(Task.id == task_id)
        )
        task = result.scalar_one_or_none()

        if not task or not task.metadata_:
            return False, "Task not found or missing metadata"

        shard_task_data = task.metadata_.get("shard_task", {})
        ground_truth_key = shard_task_data.get("ground_truth_key")
        model_name = shard_task_data.get("model_name", self.DEFAULT_MODEL)

        if not ground_truth_key:
            return False, "Missing ground truth key"

        # Parse ground truth key
        parts = ground_truth_key.split(":")
        if len(parts) < 3:
            return False, "Invalid ground truth key format"

        input_hash = parts[1]

        # Validate against ground truth
        import numpy as np
        client_output_array = np.array(client_output)

        is_valid, error_msg = await cache.validate_output(
            sample_id=input_hash,
            model_name=model_name,
            layer_index=layer_index,
            client_output=client_output_array
        )

        return is_valid, error_msg or "Validation successful"
