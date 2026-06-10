"""
Model Manager: registry view over the plug-and-play model store.

Models are defined by manifest + weights directories under ``models/`` (see
model_store). This module keeps the legacy ModelInfo facade used by API
consumers, now backed by real loaded models with real checksums.
"""

import hashlib
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

from app.config import get_settings
from app.ml.model_store import get_model_store

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
    task_type: str


class ModelManager:
    """Registry of available models, backed by the model store."""

    def _models(self) -> Dict[str, ModelInfo]:
        store = get_model_store()
        models = {}
        for spec in store.list_models():
            weight_bytes = sum(
                layer.weights.nbytes + layer.biases.nbytes for layer in spec.layers
            )
            models[spec.name] = ModelInfo(
                name=spec.name,
                version=spec.version,
                url=f"{settings.model_cdn_url}/{spec.name}/manifest.json",
                checksum=spec.checksum,
                input_shape=spec.input_shape,
                output_labels=spec.labels,
                size_bytes=weight_bytes,
                task_type=spec.task_type,
            )
        return models

    def get_model(self, name: str) -> Optional[ModelInfo]:
        return self._models().get(name)

    def get_default_model(self) -> ModelInfo:
        spec = get_model_store().get_default()
        return self._models()[spec.name]

    def list_models(self) -> List[ModelInfo]:
        return list(self._models().values())

    def get_model_for_task(self, task_type: str) -> Optional[ModelInfo]:
        for model in self._models().values():
            if model.task_type == task_type:
                return model
        return None

    def validate_checksum(self, name: str, checksum: str) -> bool:
        model = self.get_model(name)
        if not model:
            return False
        return model.checksum == checksum

    def get_labels(self, name: str) -> List[str]:
        model = self.get_model(name)
        if not model:
            return []
        return model.output_labels

    @staticmethod
    def compute_checksum(data: bytes) -> str:
        """Compute SHA-256 checksum of model data."""
        return hashlib.sha256(data).hexdigest()
