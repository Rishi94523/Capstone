"""
Prediction model for client ML predictions.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, Float, Integer, Boolean, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class Prediction(Base):
    """Client prediction result."""

    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    task_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tasks.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sample_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("samples.id"),
        nullable=False,
        index=True,
    )
    predicted_label: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Predicted label",
    )
    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Prediction confidence (0-1)",
    )
    inference_time_ms: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Inference time in milliseconds",
    )
    pow_hash: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="Proof-of-work hash",
    )
    is_valid: Mapped[Optional[bool]] = mapped_column(
        Boolean,
        nullable=True,
        comment="Whether prediction passed validation",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    # Relationships
    task = relationship("Task", back_populates="prediction")
    session = relationship("Session", back_populates="predictions")
    sample = relationship("Sample")
    verification = relationship("Verification", back_populates="prediction", uselist=False)

    def __repr__(self) -> str:
        return f"<Prediction {self.id} label={self.predicted_label}>"
