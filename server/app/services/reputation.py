"""
Reputation Service for managing user reputation scores.
"""

import logging
from typing import Optional
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models import ReputationScore

logger = logging.getLogger(__name__)
settings = get_settings()


class ReputationService:
    """
    Manages anonymous user reputation scores.

    Responsibilities:
    - Track verification accuracy
    - Update reputation scores
    - Provide reputation for consensus weighting
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_reputation(
        self, fingerprint_hash: str
    ) -> Optional[ReputationScore]:
        """
        Get reputation for a user.

        Args:
            fingerprint_hash: Anonymous fingerprint hash

        Returns:
            ReputationScore or None if not found
        """
        result = await self.db.execute(
            select(ReputationScore).where(
                ReputationScore.fingerprint_hash == fingerprint_hash
            )
        )
        return result.scalar_one_or_none()

    async def get_or_create_reputation(
        self, fingerprint_hash: str
    ) -> ReputationScore:
        """
        Get or create reputation for a user.

        Args:
            fingerprint_hash: Anonymous fingerprint hash

        Returns:
            ReputationScore
        """
        existing = await self.get_reputation(fingerprint_hash)

        if existing:
            return existing

        # Create new reputation
        reputation = ReputationScore(
            fingerprint_hash=fingerprint_hash,
            score=settings.initial_reputation,
            correct_verifications=0,
            incorrect_verifications=0,
            total_sessions=0,
        )

        self.db.add(reputation)
        await self.db.flush()

        logger.info(f"Created reputation for {fingerprint_hash[:8]}...")

        return reputation

    async def update_reputation(
        self,
        fingerprint_hash: str,
        was_correct: bool,
    ) -> ReputationScore:
        """
        Update reputation based on verification result.

        Args:
            fingerprint_hash: Anonymous fingerprint hash
            was_correct: Whether the verification was correct

        Returns:
            Updated ReputationScore
        """
        reputation = await self.get_or_create_reputation(fingerprint_hash)

        if was_correct:
            delta = settings.correct_verification_bonus
            reputation.correct_verifications += 1
        else:
            delta = -settings.incorrect_verification_penalty
            reputation.incorrect_verifications += 1

        # Update score with bounds
        new_score = reputation.score + delta
        reputation.score = max(
            settings.min_reputation,
            min(settings.max_reputation, new_score),
        )

        reputation.last_activity = datetime.utcnow()

        logger.debug(
            f"Updated reputation for {fingerprint_hash[:8]}...: "
            f"{reputation.score - delta:.2f} -> {reputation.score:.2f}"
        )

        return reputation

    async def increment_session_count(
        self, fingerprint_hash: str
    ) -> ReputationScore:
        """
        Increment session count for a user.

        Args:
            fingerprint_hash: Anonymous fingerprint hash

        Returns:
            Updated ReputationScore
        """
        reputation = await self.get_or_create_reputation(fingerprint_hash)
        reputation.total_sessions += 1
        reputation.last_activity = datetime.utcnow()

        return reputation

    async def get_top_users(self, limit: int = 100) -> list:
        """Get users with highest reputation."""
        result = await self.db.execute(
            select(ReputationScore)
            .order_by(ReputationScore.score.desc())
            .limit(limit)
        )

        return list(result.scalars().all())

    async def decay_inactive_reputations(
        self,
        inactive_days: int = 30,
        decay_factor: float = 0.95,
    ) -> int:
        """
        Apply decay to inactive user reputations.

        Args:
            inactive_days: Days of inactivity before decay
            decay_factor: Multiplier for decay (0.95 = 5% decay)

        Returns:
            Number of reputations updated
        """
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(days=inactive_days)

        result = await self.db.execute(
            select(ReputationScore).where(
                ReputationScore.last_activity < cutoff,
                ReputationScore.score > settings.initial_reputation,
            )
        )

        reputations = result.scalars().all()
        count = 0

        for rep in reputations:
            rep.score = max(
                settings.initial_reputation,
                rep.score * decay_factor,
            )
            count += 1

        if count > 0:
            logger.info(f"Decayed {count} inactive reputations")

        return count
