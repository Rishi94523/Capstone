"""Task Coordinator for assigning shard-based ML CAPTCHA tasks."""

import logging
import random
import uuid
from typing import Tuple, Optional
from dataclasses import dataclass

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio import Redis

from app.config import get_settings
from app.models import Sample, Task
from app.ml.mnist_tiny import encode_input_data, sample_to_input_vector
from app.ml.shard_manager import get_shard_manager
from app.ml.ground_truth_cache import get_ground_truth_cache

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class ShardTask:
    """Task configuration for shard-based inference."""
    task_id: uuid.UUID
    sample_id: str
    model_name: str
    model_version: str
    shards: list
    input_data: str
    input_shape: list
    expected_layers: int
    difficulty: str
    expected_time_ms: int
    ground_truth_key: str
    labels: list
    model_checksum: str


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
            "inference_time_ms": 60,
            "task_type": "shard_inference",
            "verification_probability": 0.2,
            "shard_difficulty": "easy",
            "layers": 1,
        },
        "suspicious": {
            "risk_score_max": 0.7,
            "inference_time_ms": 120,
            "task_type": "shard_inference",
            "verification_probability": 0.5,
            "shard_difficulty": "medium",
            "layers": 2,
        },
        "bot_like": {
            "risk_score_max": 1.0,
            "inference_time_ms": 180,
            "task_type": "shard_inference",
            "verification_probability": 1.0,
            "shard_difficulty": "hard",
            "layers": 3,
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
        sample: Sample,
        difficulty: str,
    ) -> Tuple[ShardTask, Task]:
        """
        Assign a shard-based inference task to a session.

        Args:
            session_id: Session UUID
            difficulty: Difficulty tier

        Returns:
            ShardTask configuration
        """
        tier_config = self.DIFFICULTY_TIERS.get(
            difficulty, self.DIFFICULTY_TIERS["normal"]
        )
        shard_difficulty = tier_config.get("shard_difficulty", "easy")
        shard_manager = await self._get_shard_manager()

        task_id = uuid.uuid4()
        available_models = shard_manager.get_available_models()
        model_name = (
            self.DEFAULT_MODEL
            if self.DEFAULT_MODEL in available_models
            else available_models[0]
            if available_models
            else self.DEFAULT_MODEL
        )

        input_data = sample_to_input_vector(sample.data_blob, sample.data_url)

        shard_assignment = await shard_manager.assign_shards(
            task_id=str(task_id),
            model_name=model_name,
            difficulty=shard_difficulty,
            input_sample=input_data,
            sample_id=str(sample.id),
        )

        ground_truth_key = (
            f"{model_name}:{sample.id}:{shard_assignment.expected_layers - 1}"
        )

        outputs = shard_manager.execute_assignment(shard_assignment)
        cache = await self._get_ground_truth_cache()
        for layer_index, output in enumerate(outputs):
            await cache.add_ground_truth(
                sample_id=str(sample.id),
                model_name=model_name,
                model_version=shard_assignment.model_version,
                layer_index=layer_index,
                layer_name=shard_assignment.shards[layer_index].name,
                input_data=shard_assignment.input_data,
                output_data=output,
                store_full_output=(layer_index == len(outputs) - 1),
            )

        shard_task = ShardTask(
            task_id=task_id,
            sample_id=str(sample.id),
            model_name=model_name,
            model_version=shard_assignment.model_version,
            shards=[s.to_dict() for s in shard_assignment.shards],
            input_data=encode_input_data(shard_assignment.input_data),
            input_shape=shard_assignment.input_shape,
            expected_layers=shard_assignment.expected_layers,
            difficulty=shard_difficulty,
            expected_time_ms=tier_config["inference_time_ms"],
            ground_truth_key=ground_truth_key,
            labels=shard_assignment.labels,
            model_checksum=shard_assignment.model_checksum,
        )

        task = Task(
            session_id=session_id,
            sample_id=sample.id,
            task_type=tier_config["task_type"],
            expected_time_ms=tier_config["inference_time_ms"],
            is_known_sample=False,
            status="assigned",
            metadata_={
                "shard_task": {
                    "task_id": str(shard_task.task_id),
                    "sample_id": str(sample.id),
                    "model_name": shard_task.model_name,
                    "model_version": shard_task.model_version,
                    "expected_layers": shard_task.expected_layers,
                    "difficulty": shard_task.difficulty,
                    "ground_truth_key": shard_task.ground_truth_key,
                    "labels": shard_task.labels,
                    "model_checksum": shard_task.model_checksum,
                }
            }
        )

        self.db.add(task)
        await self.db.flush()

        logger.debug(
            f"Assigned shard task {task_id} with {shard_assignment.expected_layers} layers "
            f"(difficulty: {shard_difficulty})"
        )

        return shard_task, task

    async def assign_task(
        self,
        session_id: uuid.UUID,
        difficulty: str,
    ) -> Tuple[Task, Sample, ShardTask]:
        """
        Assign an ML task to a session.

        Args:
            session_id: Session UUID
            difficulty: Difficulty tier

        Returns:
            Tuple of (Task, Sample)
        """
        use_known_sample = random.random() < settings.known_sample_rate

        sample = await self._select_sample(use_known_sample)

        if not sample:
            sample = await self._create_dummy_sample()

        shard_task, task = await self.assign_shard_task(
            session_id=session_id,
            sample=sample,
            difficulty=difficulty,
        )
        task.is_known_sample = use_known_sample
        task.known_label = sample.metadata_.get("known_label") if use_known_sample else None

        sample.times_served += 1
        await self.db.flush()

        logger.debug(f"Assigned task {task.id} with sample {sample.id}")
        return task, sample, shard_task

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
