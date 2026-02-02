"""
Model Shard Manager for distributing ML model shards to clients.

This module manages the splitting and distribution of ML models into
layer-wise shards for federated inference. Different difficulty levels
receive different numbers of layers to compute.
"""

import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

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
    
    def to_dict(self) -> Dict:
        """Convert shard to dictionary for JSON serialization."""
        return {
            'index': self.index,
            'name': self.name,
            'layer_type': self.layer_type,
            'weights': self.weights,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'activation': self.activation
        }
    
    @property
    def id(self) -> str:
        """Generate unique shard ID based on content hash."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class ShardAssignment:
    """Assignment of shards to a client based on difficulty."""
    task_id: str
    model_name: str
    model_version: str
    shards: List[ModelShard]
    input_data: List[float]  # Flattened input tensor
    input_shape: List[int]
    expected_layers: int  # Number of layers client should compute
    difficulty: str  # 'easy', 'medium', 'hard'
    
    def to_dict(self) -> Dict:
        """Convert assignment to dictionary."""
        return {
            'task_id': self.task_id,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'shards': [s.to_dict() for s in self.shards],
            'input_data': self.input_data,
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
    
    # Difficulty to number of layers mapping
    DIFFICULTY_LAYERS = {
        'easy': 1,      # Single layer (~10ms)
        'medium': 3,    # First 3 layers (~50ms)
        'hard': 6       # Full model (~100-200ms)
    }
    
    # Expected computation time per layer (ms)
    LAYER_TIME_MS = {
        'Conv2D': 5,
        'MaxPooling2D': 2,
        'Flatten': 1,
        'Dense': 3,
        'Activation': 1
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
        
        # Look for model directories
        if not self.models_dir.exists():
            logger.warning(f"Models directory {self.models_dir} does not exist")
            self._loaded = True
            return
            
        for model_dir in self.models_dir.iterdir():
            if not model_dir.is_dir():
                continue
                
            try:
                await self._load_model(model_dir)
            except Exception as e:
                logger.warning(f"Failed to load model from {model_dir}: {e}")
                continue
                
        self._loaded = True
        logger.info(f"ShardManager initialized with {len(self._shard_cache)} models")
        
    async def _load_model(self, model_dir: Path):
        """Load a model and its shards from directory."""
        model_name = model_dir.name
        
        # Load metadata
        metadata_path = model_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path) as f:
                self._model_metadata[model_name] = json.load(f)
        
        # Load shard manifest
        manifest_path = model_dir / 'shard_manifest.json'
        if not manifest_path.exists():
            logger.warning(f"No shard manifest found for {model_name}")
            return
            
        with open(manifest_path) as f:
            manifest = json.load(f)
            
        # Load individual shards
        shards = []
        for shard_meta in manifest.get('shards', []):
            shard = await self._load_shard(model_dir, shard_meta)
            if shard:
                shards.append(shard)
                
        if shards:
            self._shard_cache[model_name] = shards
            logger.info(f"Loaded {len(shards)} shards for model '{model_name}'")
            
    async def _load_shard(self, model_dir: Path, shard_meta: Dict) -> Optional[ModelShard]:
        """Load a single shard from disk."""
        try:
            # For MVP, we create shard from metadata
            # In production, this would load actual weights
            
            # Parse shapes
            input_shape = self._parse_shape(shard_meta.get('input_shape', '[28,28,1]'))
            output_shape = self._parse_shape(shard_meta['output_shape'])
            
            # Create stub weights for now
            # In production, load from H5 or JSON files
            weights = {}
            
            return ModelShard(
                index=shard_meta['index'],
                name=shard_meta['name'],
                layer_type=shard_meta['type'],
                weights=weights,
                input_shape=input_shape,
                output_shape=output_shape,
                activation=self._infer_activation(shard_meta['name'])
            )
        except Exception as e:
            logger.error(f"Failed to load shard {shard_meta.get('name')}: {e}")
            return None
            
    def _parse_shape(self, shape_str: str) -> List[int]:
        """Parse shape string to list of ints."""
        # Handle '(None, 28, 28, 1)' or '[None, 28, 28, 1]'
        shape_str = str(shape_str).strip('()[]')
        parts = shape_str.split(',')
        shape = []
        for p in parts:
            p = p.strip()
            if p.lower() == 'none':
                shape.append(None)
            else:
                try:
                    shape.append(int(p))
                except ValueError:
                    pass
        return shape
        
    def _infer_activation(self, layer_name: str) -> Optional[str]:
        """Infer activation function from layer name."""
        if 'relu' in layer_name.lower():
            return 'relu'
        elif 'softmax' in layer_name.lower():
            return 'softmax'
        elif 'sigmoid' in layer_name.lower():
            return 'sigmoid'
        return None
        
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
        input_sample: Optional[List[float]] = None
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
            model_name=model_name,
            model_version=model_version,
            shards=assigned_shards,
            input_data=input_sample,
            input_shape=assigned_shards[0].input_shape,
            expected_layers=num_layers,
            difficulty=difficulty
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
        
        # Generate flat array
        size = 1
        for dim in actual_shape:
            size *= dim
            
        # Generate random values in [0, 1] range (normalized)
        return [random.random() for _ in range(size)]
        
    def estimate_computation_time(self, shards: List[ModelShard]) -> int:
        """
        Estimate computation time in milliseconds for given shards.
        
        Args:
            shards: List of shards to compute
            
        Returns:
            Estimated time in milliseconds
        """
        total_ms = 0
        for shard in shards:
            layer_time = self.LAYER_TIME_MS.get(shard.layer_type, 5)
            total_ms += layer_time
        return total_ms
        
    def validate_shard_hash(self, shard: ModelShard, expected_hash: str) -> bool:
        """Validate shard integrity against expected hash."""
        return shard.id == expected_hash
        
    def get_shard_by_index(self, model_name: str, index: int) -> Optional[ModelShard]:
        """Get a specific shard by model and index."""
        if model_name not in self._shard_cache:
            return None
            
        shards = self._shard_cache[model_name]
        if index < 0 or index >= len(shards):
            return None
            
        return shards[index]


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
