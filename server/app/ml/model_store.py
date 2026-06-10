"""
Plug-and-play model store.

Loads trained models from ``models/<name>/manifest.json`` + ``weights.npz``.
Each model declares its layers, labels, preprocessing and REAL SHA-256
checksums (one per layer over the exact wire bytes, plus a model checksum
that is a hash of the layer checksums, so clients holding only a segment of
the model can still verify integrity).

Adding a new dataset/model to the system means dropping a new manifest +
weights directory into ``models/`` — no code changes required as long as the
layers are dense (the only layer type the browser shard engine executes).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# models/ directory at the repository root (server/ is the CWD in dev)
DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[3] / "models"


@dataclass
class DenseLayer:
    """A dense layer with trained weights and its wire-format checksum."""

    index: int
    name: str
    activation: str
    input_size: int
    output_size: int
    weights: np.ndarray  # shape (input_size, output_size), float32
    biases: np.ndarray  # shape (output_size,), float32
    checksum: str

    def wire_payload(self) -> dict:
        """
        JSON payload the browser executes. Weights are flattened in
        (output, input) row-major order to match the client's
        ``weights[o * inputSize + i]`` indexing.
        """
        return {
            "name": self.name,
            "type": "dense",
            "weights": np.ascontiguousarray(self.weights.T, dtype=np.float32)
            .flatten()
            .tolist(),
            "biases": self.biases.astype(np.float32).tolist(),
            "inputShape": [1, self.input_size],
            "outputShape": [1, self.output_size],
            "activation": self.activation,
        }

    def compute_checksum(self) -> str:
        """SHA-256 over the exact float32 LE bytes a client receives."""
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.weights.T, dtype="<f4").tobytes())
        h.update(np.ascontiguousarray(self.biases, dtype="<f4").tobytes())
        return h.hexdigest()


@dataclass
class ModelSpec:
    """A loaded, integrity-verified model."""

    name: str
    version: str
    task_type: str
    labels: List[str]
    input_shape: List[int]
    preprocessing: str
    checksum: str
    layers: List[DenseLayer] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def total_layers(self) -> int:
        return len(self.layers)

    @property
    def input_size(self) -> int:
        return self.layers[0].input_size

    def shard_payloads(self, start: int, end: int) -> List[dict]:
        """Wire payloads (with checksums) for the layer segment [start, end)."""
        payloads = []
        for layer in self.layers[start:end]:
            payloads.append(
                {
                    "index": layer.index,
                    "name": layer.name,
                    "layerType": "dense",
                    "inputShape": [1, layer.input_size],
                    "outputShape": [1, layer.output_size],
                    "activation": layer.activation,
                    "checksum": layer.checksum,
                    "layers": [layer.wire_payload()],
                }
            )
        return payloads

    def apply_activation(self, z: np.ndarray, activation: str) -> np.ndarray:
        if activation == "relu":
            return np.maximum(z, 0.0)
        if activation == "softmax":
            shifted = z - np.max(z)
            e = np.exp(shifted)
            return e / e.sum()
        if activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-z))
        if activation == "tanh":
            return np.tanh(z)
        return z

    def forward_segment(
        self, x: np.ndarray, start: int, end: int
    ) -> tuple[List[np.ndarray], np.ndarray]:
        """
        Reference forward pass over layers [start, end).

        Returns (pre_activations per layer, final post-activation). Used only
        for spot audits and tests — routine validation uses the projection
        checks in proof_verifier, which never run this.
        """
        pre_activations: List[np.ndarray] = []
        h = np.asarray(x, dtype=np.float64)
        for layer in self.layers[start:end]:
            z = h @ layer.weights.astype(np.float64) + layer.biases.astype(np.float64)
            pre_activations.append(z)
            h = self.apply_activation(z, layer.activation)
        return pre_activations, h

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Full forward pass returning class probabilities."""
        _, h = self.forward_segment(x, 0, self.total_layers)
        return h

    def preprocess_sample(
        self, sample_blob: Optional[bytes], sample_url: Optional[str] = None
    ) -> List[float]:
        """Convert raw sample bytes into the model's input vector."""
        if sample_blob:
            try:
                side = int(np.sqrt(self.input_size))
                image = (
                    Image.open(io.BytesIO(sample_blob))
                    .convert("L")
                    .resize((side, side))
                )
                pixels = np.asarray(image, dtype=np.float32) / 255.0
                return pixels.flatten().tolist()
            except Exception:
                pass  # non-image blob: fall through to raw bytes

        source = sample_blob or (sample_url.encode("utf-8") if sample_url else b"")
        values = [byte / 255.0 for byte in source[: self.input_size]]
        values.extend([0.0] * (self.input_size - len(values)))
        return values


def encode_input_data(input_data: Sequence[float]) -> str:
    """Encode float32 input data as base64 for the browser shard engine."""
    packed = struct.pack(f"<{len(input_data)}f", *input_data)
    return base64.b64encode(packed).decode("ascii")


def decode_input_data(encoded: str) -> List[float]:
    """Decode base64 float32 data back to a list."""
    raw = base64.b64decode(encoded)
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


class ModelStore:
    """Loads and serves all models found under the models directory."""

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
        self._models: Dict[str, ModelSpec] = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        for manifest_path in sorted(self.models_dir.glob("*/manifest.json")):
            try:
                spec = self._load_model(manifest_path)
                self._models[spec.name] = spec
                logger.info(
                    "Loaded model %s v%s (%d layers, checksum %s…)",
                    spec.name,
                    spec.version,
                    spec.total_layers,
                    spec.checksum[:12],
                )
            except Exception as exc:
                logger.error("Failed to load model at %s: %s", manifest_path, exc)
        self._loaded = True
        if not self._models:
            raise RuntimeError(
                f"No models found in {self.models_dir}. "
                "Run scripts/train_mnist_numpy.py first."
            )

    def _load_model(self, manifest_path: Path) -> ModelSpec:
        with open(manifest_path) as f:
            manifest = json.load(f)

        model_dir = manifest_path.parent
        weights_path = model_dir / manifest["weights_file"]

        # Verify the weights file is exactly the one the manifest was built from
        actual_file_hash = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        expected_file_hash = manifest.get("weights_file_sha256")
        if expected_file_hash and actual_file_hash != expected_file_hash:
            raise ValueError(f"weights file hash mismatch for {manifest['name']}")

        weights = np.load(weights_path)
        layers: List[DenseLayer] = []
        for layer_manifest in manifest["layers"]:
            i = layer_manifest["index"]
            layer = DenseLayer(
                index=i,
                name=layer_manifest["name"],
                activation=layer_manifest["activation"],
                input_size=layer_manifest["input_size"],
                output_size=layer_manifest["output_size"],
                weights=weights[f"W{i}"].astype(np.float32),
                biases=weights[f"b{i}"].astype(np.float32),
                checksum=layer_manifest["checksum"],
            )
            # Verify each layer's declared checksum against the actual weights
            actual = layer.compute_checksum()
            if actual != layer.checksum:
                raise ValueError(
                    f"layer checksum mismatch for {manifest['name']} layer {i}"
                )
            layers.append(layer)

        # Model checksum = hash of layer checksums (verify it too)
        expected_model_checksum = hashlib.sha256(
            "".join(l.checksum for l in layers).encode("ascii")
        ).hexdigest()
        if expected_model_checksum != manifest["checksum"]:
            raise ValueError(f"model checksum mismatch for {manifest['name']}")

        return ModelSpec(
            name=manifest["name"],
            version=manifest["version"],
            task_type=manifest["task_type"],
            labels=manifest["labels"],
            input_shape=manifest["input"]["shape"],
            preprocessing=manifest["input"].get("preprocessing", ""),
            checksum=manifest["checksum"],
            layers=layers,
            metrics=manifest.get("metrics", {}),
        )

    def get(self, name: str) -> Optional[ModelSpec]:
        self.load()
        return self._models.get(name)

    def get_default(self) -> ModelSpec:
        self.load()
        if "mnist-tiny" in self._models:
            return self._models["mnist-tiny"]
        return next(iter(self._models.values()))

    def list_models(self) -> List[ModelSpec]:
        self.load()
        return list(self._models.values())


_store: Optional[ModelStore] = None


def get_model_store() -> ModelStore:
    """Get or create the global model store."""
    global _store
    if _store is None:
        _store = ModelStore()
        _store.load()
    return _store


def reset_model_store() -> None:
    """Reset the global store (for tests)."""
    global _store
    _store = None
