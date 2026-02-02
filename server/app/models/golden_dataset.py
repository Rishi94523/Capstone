"""
Golden Dataset model for verified, consensus-reached labels.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, Float, Integer, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class GoldenDataset(Base):
    """Verified sample with consensus-reached label."""

    __tablename__ = "golden_dataset"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    sample_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("samples.id"),
        nullable=False,
        unique=True,
        index=True,
    )
    data_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Type of data",
    )
    verified_label: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Consensus-verified label",
    )
    confidence_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Overall confidence score",
    )
    verification_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Number of verifications",
    )
    agreement_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Percentage agreement",
    )
    weighted_agreement: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Reputation-weighted agreement",
    )
    domain_attribution: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Primary contributing domain",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # Relationships
    sample = relationship("Sample")

    def __repr__(self) -> str:
        return f"<GoldenDataset {self.id} label={self.verified_label}>"
