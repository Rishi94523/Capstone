"""
Verification model for human verification responses.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, Float, Integer, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class Verification(Base):
    """Human verification response."""

    __tablename__ = "verifications"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("predictions.id", ondelete="CASCADE"),
        nullable=False,
    )
    sample_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("samples.id"),
        nullable=False,
        index=True,
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    response_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="Response type: confirm, reject, correct",
    )
    original_label: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Original predicted label",
    )
    verified_label: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="Corrected label if response_type is 'correct'",
    )
    response_time_ms: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Time taken to respond in milliseconds",
    )
    reputation_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="User's reputation at time of verification",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    # Relationships
    prediction = relationship("Prediction", back_populates="verification")
    sample = relationship("Sample")
    session = relationship("Session", back_populates="verifications")

    def __repr__(self) -> str:
        return f"<Verification {self.id} type={self.response_type}>"

    @property
    def final_label(self) -> str:
        """Get the final label (corrected or original)."""
        return self.verified_label or self.original_label
