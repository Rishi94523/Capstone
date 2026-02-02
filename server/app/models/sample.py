"""
Sample model for ML training data.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, Integer, Text, DateTime, LargeBinary, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Sample(Base):
    """Unlabeled ML sample for inference tasks."""

    __tablename__ = "samples"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    data_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Type of data: 'image', 'text'",
    )
    model_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Model type: 'cifar10', 'imdb'",
    )
    data_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
        comment="SHA-256 hash of data",
    )
    data_url: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="CDN URL for sample",
    )
    data_blob: Mapped[Optional[bytes]] = mapped_column(
        LargeBinary,
        nullable=True,
        comment="Raw data blob",
    )
    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSON,  # Use JSON instead of JSONB for SQLite compatibility
        nullable=False,
        default=dict,
        comment="Additional metadata",
    )
    times_served: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of times served",
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

    def __repr__(self) -> str:
        return f"<Sample {self.id} type={self.data_type}>"
