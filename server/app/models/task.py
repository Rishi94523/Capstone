"""
Task model for ML task assignments.
"""

import uuid
from datetime import datetime

from sqlalchemy import String, Integer, Boolean, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class Task(Base):
    """Assigned ML task for a session."""

    __tablename__ = "tasks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
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
    task_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Task type: inference, gradient, training",
    )
    expected_time_ms: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Expected completion time in milliseconds",
    )
    is_known_sample: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether this is a honeypot sample",
    )
    known_label: Mapped[str] = mapped_column(
        String(100),
        nullable=True,
        comment="Expected label for known samples",
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="assigned",
        comment="Task status: assigned, processing, completed, failed",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    # Relationships
    session = relationship("Session", back_populates="tasks")
    sample = relationship("Sample")
    prediction = relationship("Prediction", back_populates="task", uselist=False)

    def __repr__(self) -> str:
        return f"<Task {self.id} type={self.task_type}>"
