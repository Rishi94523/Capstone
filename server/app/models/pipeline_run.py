"""
PipelineRun model: one distributed inference of a sample through a model.

A run advances layer by layer as different CAPTCHA sessions each compute a
verified segment. The stored activation is the handoff point between
contributors; when the final layer completes, the run yields the sample's
predicted label — the full picture pieced together from partial computations.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, Integer, Float, DateTime, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class PipelineRun(Base):
    """Distributed inference run over a sample, advanced segment by segment."""

    __tablename__ = "pipeline_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    sample_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("samples.id"),
        nullable=False,
        index=True,
    )
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    next_layer: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Index of the next layer to compute",
    )
    activation: Mapped[Optional[list]] = mapped_column(
        JSON,
        nullable=True,
        comment="Verified post-activation output of the last completed layer "
        "(input for the next contributor); null means start from sample input",
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="in_progress",
        index=True,
        comment="'in_progress', 'completed', 'failed'",
    )
    predicted_label: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    contributors: Mapped[list] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        comment="Session ids and segments that contributed to this run",
    )
    claimed_by_task: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="Task currently working on this run's next segment",
    )
    claimed_until: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        nullable=True,
        comment="Claim expiry; after this the segment can be reassigned",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    def __repr__(self) -> str:
        return (
            f"<PipelineRun {self.id} sample={self.sample_id} "
            f"next_layer={self.next_layer} status={self.status}>"
        )
