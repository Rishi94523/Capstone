"""
Session model for CAPTCHA sessions.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, Float, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class Session(Base):
    """CAPTCHA session tracking."""

    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    domain: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Domain requesting CAPTCHA",
    )
    session_token: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        index=True,
        comment="Unique session token",
    )
    risk_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Computed risk score",
    )
    difficulty_tier: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="normal",
        comment="Difficulty tier: normal, suspicious, bot_like",
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="pending",
        index=True,
        comment="Session status: pending, processing, completed, failed",
    )
    client_fingerprint: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        comment="Anonymous client fingerprint hash",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        nullable=True,
        comment="When session was completed",
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        comment="Session expiry time",
    )

    # Relationships
    tasks = relationship("Task", back_populates="session")
    predictions = relationship("Prediction", back_populates="session")
    verifications = relationship("Verification", back_populates="session")

    def __repr__(self) -> str:
        return f"<Session {self.id} status={self.status}>"

    @property
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.utcnow() >= self.expires_at

    @property
    def is_completed(self) -> bool:
        """Check if session is completed."""
        return self.status == "completed"
