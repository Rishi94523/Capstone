"""
Task Coordinator for assigning ML tasks to sessions.
"""

import logging
import random
import uuid
from typing import Tuple, Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio import Redis

from app.config import get_settings
from app.models import Sample, Task

logger = logging.getLogger(__name__)
settings = get_settings()


class TaskCoordinator:
    """
    Coordinates ML task assignment based on risk and difficulty.

    Responsibilities:
    - Assign appropriate tasks based on difficulty tier
    - Balance workload across sample pool
    - Inject known samples for validation
    """

    # Difficulty tier configurations
    DIFFICULTY_TIERS = {
        "normal": {
            "risk_score_max": 0.3,
            "inference_time_ms": 500,
            "task_type": "inference",
            "verification_probability": 0.2,
        },
        "suspicious": {
            "risk_score_max": 0.7,
            "inference_time_ms": 3000,
            "task_type": "inference",
            "verification_probability": 0.5,
        },
        "bot_like": {
            "risk_score_max": 1.0,
            "inference_time_ms": 10000,
            "task_type": "training",
            "verification_probability": 1.0,
        },
    }

    def __init__(self, db: AsyncSession, redis: Redis):
        self.db = db
        self.redis = redis

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

        # Create task
        task = Task(
            session_id=session_id,
            sample_id=sample.id,
            task_type=tier_config["task_type"],
            expected_time_ms=tier_config["inference_time_ms"],
            is_known_sample=use_known_sample,
            known_label=sample.metadata_.get("known_label") if use_known_sample else None,
            status="assigned",
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
            model_type="cifar10",
            data_hash=data_hash,
            data_blob=random_data,
            metadata_={
                "width": 32,
                "height": 32,
                "channels": 3,
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
