"""
Ground Truth Cache for pre-computing and caching expected ML outputs.

This module maintains a cache of pre-computed model outputs for validation.
It caches ground truth for different layer depths to support shard validation.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GroundTruthEntry:
    """Pre-computed output for a specific input at a specific layer."""
    sample_id: str
    model_name: str
    model_version: str
    layer_index: int  # -1 means final output
    layer_name: str
    input_hash: str  # Hash of input data for verification
    output_hash: str  # Hash of expected output
    output_data: Optional[List[float]] = None  # Actual output (optional, for detailed validation)
    top_prediction: Optional[int] = None  # For classification tasks
    confidence: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'sample_id': self.sample_id,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'layer_index': self.layer_index,
            'layer_name': self.layer_name,
            'input_hash': self.input_hash,
            'output_hash': self.output_hash,
            'output_data': self.output_data,
            'top_prediction': self.top_prediction,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GroundTruthEntry':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class CacheStats:
    """Statistics for the ground truth cache."""
    total_entries: int
    models_cached: List[str]
    avg_entries_per_model: float
    cache_size_mb: float


class GroundTruthCache:
    """
    Manages pre-computed ground truth outputs for model validation.
    
    The cache stores expected outputs at various layer depths:
    - Layer 0: Output after first layer (for easy difficulty)
    - Layer N: Output after N layers (for medium difficulty)
    - Final: Final model output (for hard difficulty)
    
    This allows the server to validate partial computations without
    running the full model on every request.
    """
    
    def __init__(self, cache_dir: str = 'cache/ground_truth'):
        self.cache_dir = Path(cache_dir)
        self._cache: Dict[str, GroundTruthEntry] = {}
        self._initialized = False
        
    async def initialize(self):
        """Initialize the cache from disk."""
        if self._initialized:
            return
            
        logger.info("Initializing GroundTruthCache...")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing cache files
        await self._load_cache_files()
        
        self._initialized = True
        logger.info(f"GroundTruthCache initialized with {len(self._cache)} entries")
        
    async def _load_cache_files(self):
        """Load all cache files from disk."""
        if not self.cache_dir.exists():
            return
            
        for cache_file in self.cache_dir.glob('*.json'):
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                    
                if isinstance(data, list):
                    for entry_data in data:
                        entry = GroundTruthEntry.from_dict(entry_data)
                        cache_key = self._make_cache_key(
                            entry.sample_id,
                            entry.model_name,
                            entry.layer_index
                        )
                        self._cache[cache_key] = entry
                elif isinstance(data, dict):
                    entry = GroundTruthEntry.from_dict(data)
                    cache_key = self._make_cache_key(
                        entry.sample_id,
                        entry.model_name,
                        entry.layer_index
                    )
                    self._cache[cache_key] = entry
                    
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}")
                
    def _make_cache_key(self, sample_id: str, model_name: str, layer_index: int) -> str:
        """Generate a unique cache key."""
        return f"{model_name}:{sample_id}:{layer_index}"
    
    def _hash_input(self, input_data: List[float]) -> str:
        """Generate hash for input data."""
        input_bytes = json.dumps(input_data, sort_keys=True).encode()
        return hashlib.sha256(input_bytes).hexdigest()[:16]
    
    def _hash_output(self, output_data: np.ndarray) -> str:
        """Generate hash for output tensor."""
        output_bytes = output_data.tobytes()
        return hashlib.sha256(output_bytes).hexdigest()[:16]
    
    async def add_ground_truth(
        self,
        sample_id: str,
        model_name: str,
        model_version: str,
        layer_index: int,
        layer_name: str,
        input_data: List[float],
        output_data: np.ndarray,
        store_full_output: bool = False
    ) -> GroundTruthEntry:
        """
        Add a ground truth entry to the cache.
        
        Args:
            sample_id: Unique sample identifier
            model_name: Name of the model
            model_version: Model version string
            layer_index: Layer depth (-1 for final output)
            layer_name: Name of the layer
            input_data: Input tensor as flat list
            output_data: Output tensor as numpy array
            store_full_output: Whether to store full output data
            
        Returns:
            The created GroundTruthEntry
        """
        input_hash = self._hash_input(input_data)
        output_hash = self._hash_output(output_data)
        
        # Get top prediction for classification
        top_prediction = None
        confidence = None
        if output_data.size > 0:
            flat_output = output_data.flatten()
            top_prediction = int(np.argmax(flat_output))
            exp_sum = np.sum(np.exp(flat_output - np.max(flat_output)))
            if exp_sum > 0:
                confidence = float(np.exp(flat_output[top_prediction] - np.max(flat_output)) / exp_sum)
        
        output_list = None
        if store_full_output:
            output_list = output_data.flatten().tolist()
        
        entry = GroundTruthEntry(
            sample_id=sample_id,
            model_name=model_name,
            model_version=model_version,
            layer_index=layer_index,
            layer_name=layer_name,
            input_hash=input_hash,
            output_hash=output_hash,
            output_data=output_list,
            top_prediction=top_prediction,
            confidence=confidence
        )
        
        cache_key = self._make_cache_key(sample_id, model_name, layer_index)
        self._cache[cache_key] = entry
        
        return entry
    
    def get_ground_truth(
        self,
        sample_id: str,
        model_name: str,
        layer_index: int = -1
    ) -> Optional[GroundTruthEntry]:
        """
        Retrieve a ground truth entry from the cache.
        
        Args:
            sample_id: Unique sample identifier
            model_name: Name of the model
            layer_index: Layer depth (-1 for final output)
            
        Returns:
            GroundTruthEntry if found, None otherwise
        """
        cache_key = self._make_cache_key(sample_id, model_name, layer_index)
        return self._cache.get(cache_key)
    
    async def validate_output(
        self,
        sample_id: str,
        model_name: str,
        layer_index: int,
        client_output: np.ndarray,
        tolerance: float = 1e-5
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate client output against cached ground truth.
        
        Args:
            sample_id: Unique sample identifier
            model_name: Name of the model
            layer_index: Layer depth being validated
            client_output: Output from client as numpy array
            tolerance: Numerical tolerance for comparison
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        ground_truth = self.get_ground_truth(sample_id, model_name, layer_index)
        
        if ground_truth is None:
            return False, f"No ground truth found for {model_name}:{sample_id}:{layer_index}"
        
        # Compute hash of client output
        client_hash = self._hash_output(client_output)
        
        # Compare hashes (fast path)
        if client_hash == ground_truth.output_hash:
            return True, None
        
        # If hashes don't match and we have full output, do detailed comparison
        if ground_truth.output_data is not None:
            gt_output = np.array(ground_truth.output_data)
            
            if gt_output.shape != client_output.shape:
                return False, f"Output shape mismatch: expected {gt_output.shape}, got {client_output.shape}"
            
            # Check numerical difference
            diff = np.abs(gt_output - client_output.flatten())
            max_diff = np.max(diff)
            
            if max_diff > tolerance:
                return False, f"Output mismatch: max difference {max_diff} > tolerance {tolerance}"
        
        return True, None
    
    async def save_cache(self, model_name: Optional[str] = None):
        """
        Save cache to disk.
        
        Args:
            model_name: If specified, only save entries for this model
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Group entries by model
        entries_by_model: Dict[str, List[GroundTruthEntry]] = {}
        for entry in self._cache.values():
            if model_name and entry.model_name != model_name:
                continue
            if entry.model_name not in entries_by_model:
                entries_by_model[entry.model_name] = []
            entries_by_model[entry.model_name].append(entry)
        
        # Save each model's entries
        for model, entries in entries_by_model.items():
            cache_file = self.cache_dir / f"{model}_ground_truth.json"
            data = [e.to_dict() for e in entries]
            
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved {len(entries)} ground truth entries for {model}")
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        models = set()
        for entry in self._cache.values():
            models.add(entry.model_name)
        
        total_size = 0
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob('*.json'):
                total_size += cache_file.stat().st_size
        
        return CacheStats(
            total_entries=len(self._cache),
            models_cached=list(models),
            avg_entries_per_model=len(self._cache) / max(len(models), 1),
            cache_size_mb=total_size / (1024 * 1024)
        )
    
    def clear_cache(self):
        """Clear all cached entries."""
        self._cache.clear()
        logger.info("Ground truth cache cleared")
    
    async def warm_cache(
        self,
        model_name: str,
        model_version: str,
        samples: List[Tuple[str, List[float]]],
        layer_indices: List[int],
        model_forward_fn
    ):
        """
        Pre-compute ground truth for multiple samples and layers.
        
        Args:
            model_name: Name of the model
            model_version: Model version
            samples: List of (sample_id, input_data) tuples
            layer_indices: List of layer indices to compute (-1 for final)
            model_forward_fn: Function that takes input and layer_index, returns output
        """
        logger.info(f"Warming cache for {model_name} with {len(samples)} samples...")
        
        for sample_id, input_data in samples:
            for layer_index in layer_indices:
                try:
                    # Compute output up to this layer
                    output = model_forward_fn(input_data, layer_index)
                    
                    # Add to cache
                    await self.add_ground_truth(
                        sample_id=sample_id,
                        model_name=model_name,
                        model_version=model_version,
                        layer_index=layer_index,
                        layer_name=f"layer_{layer_index}",
                        input_data=input_data,
                        output_data=output,
                        store_full_output=(layer_index == -1)  # Store full output only for final
                    )
                except Exception as e:
                    logger.warning(f"Failed to compute ground truth for {sample_id}:{layer_index}: {e}")
        
        # Save updated cache
        await self.save_cache(model_name)
        
        logger.info(f"Cache warming complete for {model_name}")


# Global cache instance
_cache_instance: Optional[GroundTruthCache] = None


async def get_ground_truth_cache() -> GroundTruthCache:
    """Get or create the global ground truth cache."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = GroundTruthCache()
        await _cache_instance.initialize()
    return _cache_instance


def reset_ground_truth_cache():
    """Reset the global cache (useful for testing)."""
    global _cache_instance
    _cache_instance = None
