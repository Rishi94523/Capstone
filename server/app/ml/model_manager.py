"""
Model Manager for ML model lifecycle.
"""

import logging
import hashlib
from typing import Dict, List, Optional
from dataclasses import dataclass

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class ModelInfo:
    """Information about an ML model."""

    name: str
    version: str
    url: str
    checksum: str
    input_shape: List[int]
    output_labels: List[str]
    size_bytes: int
    task_type: str  # classification, sentiment, etc.


class ModelManager:
    """
    Manages ML model metadata and versioning.

    Responsibilities:
    - Track available models
    - Manage model versions
    - Provide model URLs for clients
    - Validate model checksums
    """

    # Registered models
    MODELS: Dict[str, ModelInfo] = {
        "cifar10-mobilenet": ModelInfo(
            name="cifar10-mobilenet",
            version="1.0.0",
            url=f"{settings.model_cdn_url}/cifar10-mobilenet/model.json",
            checksum="abc123",  # TODO: Real checksum
            input_shape=[1, 32, 32, 3],
            output_labels=[
                "airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck",
            ],
            size_bytes=3 * 1024 * 1024,  # ~3MB
            task_type="image_classification",
        ),
        "imdb-distilbert": ModelInfo(
            name="imdb-distilbert",
            version="1.0.0",
            url=f"{settings.model_cdn_url}/imdb-distilbert/model.json",
            checksum="def456",  # TODO: Real checksum
            input_shape=[1, 256],
            output_labels=["negative", "positive"],
            size_bytes=25 * 1024 * 1024,  # ~25MB
            task_type="text_classification",
        ),
    }

    def __init__(self):
        pass

    def get_model(self, name: str) -> Optional[ModelInfo]:
        """Get model information by name."""
        return self.MODELS.get(name)

    def get_default_model(self) -> ModelInfo:
        """Get the default model."""
        return self.MODELS[settings.default_model]

    def list_models(self) -> List[ModelInfo]:
        """List all available models."""
        return list(self.MODELS.values())

    def get_model_for_task(self, task_type: str) -> Optional[ModelInfo]:
        """Get appropriate model for a task type."""
        for model in self.MODELS.values():
            if model.task_type == task_type:
                return model
        return None

    def validate_checksum(self, name: str, checksum: str) -> bool:
        """Validate a model checksum."""
        model = self.get_model(name)
        if not model:
            return False
        return model.checksum == checksum

    def get_labels(self, name: str) -> List[str]:
        """Get output labels for a model."""
        model = self.get_model(name)
        if not model:
            return []
        return model.output_labels

    @staticmethod
    def compute_checksum(data: bytes) -> str:
        """Compute SHA-256 checksum of model data."""
        return hashlib.sha256(data).hexdigest()

    def register_model(self, model_info: ModelInfo) -> None:
        """Register a new model."""
        self.MODELS[model_info.name] = model_info
        logger.info(f"Registered model: {model_info.name} v{model_info.version}")

    def update_model_version(
        self,
        name: str,
        version: str,
        url: str,
        checksum: str,
    ) -> bool:
        """Update a model's version."""
        model = self.get_model(name)
        if not model:
            return False

        # Create new model info with updated version
        updated = ModelInfo(
            name=model.name,
            version=version,
            url=url,
            checksum=checksum,
            input_shape=model.input_shape,
            output_labels=model.output_labels,
            size_bytes=model.size_bytes,
            task_type=model.task_type,
        )

        self.MODELS[name] = updated
        logger.info(f"Updated model {name} to version {version}")

        return True
