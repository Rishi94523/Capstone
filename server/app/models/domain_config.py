"""
Domain configuration model.
"""

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import String, Float, Boolean, DateTime, Text, ARRAY
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class DomainConfig(Base):
    """Per-domain CAPTCHA configuration."""

    __tablename__ = "domain_config"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    domain: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        index=True,
        comment="Domain name",
    )
    api_key_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="Hashed API key",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment="Whether domain is active",
    )
    verification_rate: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.2,
        comment="Rate of sessions requiring verification",
    )
    difficulty_multiplier: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=1.0,
        comment="Difficulty multiplier",
    )
    allowed_origins: Mapped[Optional[List[str]]] = mapped_column(
        ARRAY(Text),
        nullable=True,
        comment="Allowed CORS origins",
    )
    webhook_url: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Webhook URL for notifications",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    def __repr__(self) -> str:
        return f"<DomainConfig {self.domain}>"
