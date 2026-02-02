"""
Tests for Task Coordinator.
"""

import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock

from app.core.task_coordinator import TaskCoordinator


class TestTaskCoordinator:
    """Tests for TaskCoordinator class."""

    def test_get_difficulty_tier_normal(self):
        """Test normal difficulty tier for low risk."""
        db = MagicMock()
        redis = MagicMock()
        coordinator = TaskCoordinator(db, redis)

        tier = coordinator.get_difficulty_tier(0.1)
        assert tier == "normal"

        tier = coordinator.get_difficulty_tier(0.3)
        assert tier == "normal"

    def test_get_difficulty_tier_suspicious(self):
        """Test suspicious difficulty tier for medium risk."""
        db = MagicMock()
        redis = MagicMock()
        coordinator = TaskCoordinator(db, redis)

        tier = coordinator.get_difficulty_tier(0.5)
        assert tier == "suspicious"

        tier = coordinator.get_difficulty_tier(0.7)
        assert tier == "suspicious"

    def test_get_difficulty_tier_bot_like(self):
        """Test bot_like difficulty tier for high risk."""
        db = MagicMock()
        redis = MagicMock()
        coordinator = TaskCoordinator(db, redis)

        tier = coordinator.get_difficulty_tier(0.8)
        assert tier == "bot_like"

        tier = coordinator.get_difficulty_tier(1.0)
        assert tier == "bot_like"

    def test_difficulty_tier_configurations(self):
        """Test difficulty tier configurations are valid."""
        db = MagicMock()
        redis = MagicMock()
        coordinator = TaskCoordinator(db, redis)

        for tier_name, config in coordinator.DIFFICULTY_TIERS.items():
            assert "risk_score_max" in config
            assert "inference_time_ms" in config
            assert "task_type" in config
            assert "verification_probability" in config
            assert config["inference_time_ms"] > 0
            assert 0 <= config["verification_probability"] <= 1
