"""
Reputation model for anonymous user reputation tracking.
"""

import uuid
from datetime import datetime

from sqlalchemy import String, Float, Integer, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class ReputationScore(Base):
    """Anonymous user reputation score."""

    __tablename__ = "reputation_scores"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    fingerprint_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
        index=True,
        comment="Anonymous fingerprint hash",
    )
    score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=1.0,
        comment="Reputation score (0.0 - 5.0)",
    )
    correct_verifications: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of correct verifications",
    )
    incorrect_verifications: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of incorrect verifications",
    )
    total_sessions: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Total CAPTCHA sessions",
    )
    last_activity: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        comment="Last activity timestamp",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    def __repr__(self) -> str:
        return f"<ReputationScore {self.fingerprint_hash[:8]}... score={self.score}>"

    @property
    def accuracy(self) -> float:
        """Calculate verification accuracy."""
        total = self.correct_verifications + self.incorrect_verifications
        if total == 0:
            return 0.0
        return self.correct_verifications / total
