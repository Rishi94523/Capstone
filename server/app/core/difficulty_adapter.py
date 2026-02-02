"""
Difficulty Adapter for dynamic difficulty adjustment.
"""

import logging
from typing import Dict, Any

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class DifficultyAdapter:
    """
    Adapts difficulty parameters based on various factors.

    Can adjust:
    - Inference time requirements
    - Task complexity
    - Verification probability
    - Model selection
    """

    def __init__(self):
        self.base_configs = {
            "normal": {
                "inference_time_ms": settings.normal_difficulty_time_ms,
                "task_type": "inference",
                "verification_probability": settings.verification_rate,
                "model": "cifar10-mobilenet",
                "batch_size": 1,
            },
            "suspicious": {
                "inference_time_ms": settings.suspicious_difficulty_time_ms,
                "task_type": "inference",
                "verification_probability": 0.5,
                "model": "cifar10-mobilenet",
                "batch_size": 1,
            },
            "bot_like": {
                "inference_time_ms": settings.bot_difficulty_time_ms,
                "task_type": "training",
                "verification_probability": 1.0,
                "model": "cifar10-mobilenet",
                "batch_size": 10,
            },
        }

    def get_config(
        self,
        difficulty: str,
        domain_multiplier: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Get difficulty configuration.

        Args:
            difficulty: Base difficulty tier
            domain_multiplier: Domain-specific multiplier

        Returns:
            Configuration dictionary
        """
        base = self.base_configs.get(difficulty, self.base_configs["normal"]).copy()

        # Apply domain multiplier
        base["inference_time_ms"] = int(base["inference_time_ms"] * domain_multiplier)

        return base

    def adjust_for_time_of_day(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust difficulty based on time of day.

        Higher difficulty during peak bot activity hours.
        """
        from datetime import datetime

        hour = datetime.utcnow().hour

        # Peak bot hours (typically late night / early morning)
        if 0 <= hour < 6:
            config["inference_time_ms"] = int(config["inference_time_ms"] * 1.2)
            config["verification_probability"] = min(
                1.0, config["verification_probability"] * 1.2
            )

        return config

    def adjust_for_attack_detection(
        self, config: Dict[str, Any], attack_level: float
    ) -> Dict[str, Any]:
        """
        Adjust difficulty during detected attacks.

        Args:
            config: Base configuration
            attack_level: Attack severity (0.0 - 1.0)

        Returns:
            Adjusted configuration
        """
        if attack_level > 0.5:
            # Under attack - increase difficulty significantly
            config["inference_time_ms"] = int(
                config["inference_time_ms"] * (1 + attack_level)
            )
            config["verification_probability"] = min(
                1.0, config["verification_probability"] + attack_level * 0.3
            )
            config["batch_size"] = max(1, int(config["batch_size"] * (1 + attack_level)))

        return config

    def get_model_for_difficulty(self, difficulty: str) -> str:
        """Get appropriate model for difficulty level."""
        models = {
            "normal": "cifar10-mobilenet",
            "suspicious": "cifar10-mobilenet",
            "bot_like": "imdb-distilbert",  # Larger model for bots
        }
        return models.get(difficulty, "cifar10-mobilenet")
