"""
Model Shard Manager for distributing executable ML model shards to clients.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import numpy as np

from app.ml.mnist_tiny import (
    LABELS,
    MODEL_CHECKSUM,
    MODEL_NAME,
    MODEL_VERSION,
    encode_input_data,
    execute_shards,
    get_model_shards,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelShard:
    """Represents a single layer shard of a model."""
    index: int
    name: str
    layer_type: str
    weights: Dict[str, Any]  # Serialized layer weights
    input_shape: List[int]
    output_shape: List[int]
    activation: Optional[str] = None
    layers: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict:
        """Convert shard to dictionary for JSON serialization."""
        return {
            'index': self.index,
            'name': self.name,
            'layerType': self.layer_type,
            'weights': self.weights,
            'inputShape': self.input_shape,
            'outputShape': self.output_shape,
            'activation': self.activation,
            'layers': self.layers or [],
        }


@dataclass
class ShardAssignment:
    """Assignment of shards to a client based on difficulty."""
    task_id: str
    model_name: str
    model_version: str
    shards: List[ModelShard]
    sample_id: str
    input_data: List[float]  # Flattened input tensor
    input_shape: List[int]
    expected_layers: int  # Number of layers client should compute
    difficulty: str  # 'easy', 'medium', 'hard'
    labels: List[str]
    model_checksum: str
    
    def to_dict(self) -> Dict:
        """Convert assignment to dictionary."""
        return {
            'task_id': self.task_id,
            'sample_id': self.sample_id,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'shards': [s.to_dict() for s in self.shards],
            'input_data': encode_input_data(self.input_data),
            'input_shape': self.input_shape,
            'expected_layers': self.expected_layers,
            'difficulty': self.difficulty
        }


class ShardManager:
    """
    Manages model shards and distributes them based on difficulty.
    
    Responsibilities:
    - Load and cache model shards
    - Assign appropriate shards based on difficulty tier
    - Track shard checksums for integrity
    - Manage progressive disclosure (more layers for higher difficulty)
    """
    
    DIFFICULTY_LAYERS = {
        'easy': 1,
        'medium': 2,
        'hard': 3,
    }
    
    def __init__(self, models_dir: str = 'models'):
        self.models_dir = Path(models_dir)
        self._shard_cache: Dict[str, List[ModelShard]] = {}
        self._model_metadata: Dict[str, Dict] = {}
        self._loaded = False
        
    async def initialize(self):
        """Load available models and their shards."""
        if self._loaded:
            return
            
        logger.info("Initializing ShardManager...")
        self._load_builtin_model()
        self._loaded = True
        logger.info(f"ShardManager initialized with {len(self._shard_cache)} models")

    def _load_builtin_model(self) -> None:
        """Load the deterministic built-in MNIST shard model."""
        shard_dicts = get_model_shards()
        self._shard_cache[MODEL_NAME] = [
            ModelShard(
                index=shard["index"],
                name=shard["name"],
                layer_type=shard["layerType"],
                weights={},
                input_shape=shard["inputShape"],
                output_shape=shard["outputShape"],
                activation=shard["layers"][0]["activation"],
                layers=shard["layers"],
            )
            for shard in shard_dicts
        ]
        self._model_metadata[MODEL_NAME] = {
            "name": MODEL_NAME,
            "version": MODEL_VERSION,
            "labels": LABELS,
            "checksum": MODEL_CHECKSUM,
            "input_shape": [1, 784],
        }
        
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return list(self._shard_cache.keys())
        
    def get_model_metadata(self, model_name: str) -> Optional[Dict]:
        """Get metadata for a model."""
        return self._model_metadata.get(model_name)
        
    async def assign_shards(
        self,
        task_id: str,
        model_name: str,
        difficulty: str,
        input_sample: Optional[List[float]] = None,
        sample_id: str = "",
    ) -> ShardAssignment:
        """
        Assign shards to a client based on difficulty.
        
        Args:
            task_id: Unique task identifier
            model_name: Name of the model to use
            difficulty: 'easy', 'medium', or 'hard'
            input_sample: Optional pre-selected input sample
            
        Returns:
            ShardAssignment with appropriate layers for difficulty
        """
        if model_name not in self._shard_cache:
            raise ValueError(f"Model '{model_name}' not found")
            
        all_shards = self._shard_cache[model_name]
        num_layers = self.DIFFICULTY_LAYERS.get(difficulty, 1)
        
        # Limit to available layers
        num_layers = min(num_layers, len(all_shards))
        
        # Select first N layers (progressive disclosure)
        assigned_shards = all_shards[:num_layers]
        
        # Generate or use provided input
        if input_sample is None:
            input_sample = self._generate_random_input(assigned_shards[0].input_shape)
            
        # Get model version from metadata
        metadata = self._model_metadata.get(model_name, {})
        model_version = metadata.get('version', '1.0.0')
        
        assignment = ShardAssignment(
            task_id=task_id,
            sample_id=sample_id,
            model_name=model_name,
            model_version=model_version,
            shards=assigned_shards,
            input_data=input_sample,
            input_shape=assigned_shards[0].input_shape,
            expected_layers=num_layers,
            difficulty=difficulty,
            labels=metadata.get("labels", LABELS),
            model_checksum=metadata.get("checksum", MODEL_CHECKSUM),
        )
        
        logger.debug(
            f"Assigned {num_layers} layers for task {task_id} "
            f"(difficulty: {difficulty})"
        )
        
        return assignment
        
    def _generate_random_input(self, shape: List[int]) -> List[float]:
        """Generate random input matching the shape."""
        import random
        
        # Remove None/batch dimension
        actual_shape = [s for s in shape if s is not None]
        
        size = 1
        for dim in actual_shape:
            size *= dim
            
        # Generate random values in [0, 1] range (normalized)
        return [random.random() for _ in range(size)]
        
    def get_shard_by_index(self, model_name: str, index: int) -> Optional[ModelShard]:
        """Get a specific shard by model and index."""
        if model_name not in self._shard_cache:
            return None
            
        shards = self._shard_cache[model_name]
        if index < 0 or index >= len(shards):
            return None
            
        return shards[index]

    def execute_assignment(self, assignment: ShardAssignment) -> List[np.ndarray]:
        """Execute assigned shards and return intermediate outputs."""
        shard_payloads = [shard.to_dict() for shard in assignment.shards]
        return execute_shards(assignment.input_data, shard_payloads)


# Global shard manager instance
_shard_manager: Optional[ShardManager] = None


async def get_shard_manager() -> ShardManager:
    """Get or create the global shard manager instance."""
    global _shard_manager
    if _shard_manager is None:
        _shard_manager = ShardManager()
        await _shard_manager.initialize()
    return _shard_manager


def reset_shard_manager():
    """Reset the global shard manager (useful for testing)."""
    global _shard_manager
    _shard_manager = None
