"""
Deterministic tiny MNIST model specification used for shard-based CAPTCHA tasks.

The weights are generated from fixed seeds so both task assignment and validation
can reproduce the same layer payloads without depending on TensorFlow at runtime.
"""

from __future__ import annotations

import base64
import io
import random
import struct
from typing import List, Sequence

import numpy as np
from PIL import Image

MODEL_NAME = "mnist-tiny"
MODEL_VERSION = "1.0.0"
MODEL_CHECKSUM = "mnist-tiny-v1-deterministic"
LABELS = [str(i) for i in range(10)]


def _make_dense_layer(
    *,
    index: int,
    name: str,
    input_size: int,
    output_size: int,
    activation: str,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    weights = [
        round(rng.uniform(-0.15, 0.15), 6)
        for _ in range(input_size * output_size)
    ]
    biases = [round(rng.uniform(-0.05, 0.05), 6) for _ in range(output_size)]
    layer = {
        "name": name,
        "type": "dense",
        "weights": weights,
        "biases": biases,
        "inputShape": [1, input_size],
        "outputShape": [1, output_size],
        "activation": activation,
    }
    return {
        "index": index,
        "name": name,
        "layerType": "dense",
        "inputShape": [1, input_size],
        "outputShape": [1, output_size],
        "layers": [layer],
    }


def get_model_shards() -> List[dict]:
    """Return the deterministic model shards used for CAPTCHA tasks."""
    return [
        _make_dense_layer(
            index=0,
            name="dense_hidden_1",
            input_size=784,
            output_size=128,
            activation="relu",
            seed=17,
        ),
        _make_dense_layer(
            index=1,
            name="dense_hidden_2",
            input_size=128,
            output_size=64,
            activation="relu",
            seed=23,
        ),
        _make_dense_layer(
            index=2,
            name="dense_output",
            input_size=64,
            output_size=10,
            activation="softmax",
            seed=31,
        ),
    ]


def sample_to_input_vector(sample_blob: bytes | None, sample_url: str | None = None) -> List[float]:
    """
    Convert sample bytes to a normalized 784-length input vector.

    If the sample already stores an input vector in raw bytes, we use that directly.
    Otherwise, we deterministically derive one from the available bytes/URL.
    """
    if sample_blob:
        try:
            image = Image.open(io.BytesIO(sample_blob)).convert("L").resize((28, 28))
            pixel_data = np.asarray(image, dtype=np.float32) / 255.0
            return pixel_data.flatten().tolist()
        except Exception:
            # Fall back to the raw-byte path below for non-image blobs.
            pass

    source = sample_blob or (sample_url.encode("utf-8") if sample_url else b"")
    if not source:
        source = bytes([0] * 784)

    values = [byte / 255.0 for byte in source[:784]]
    if len(values) < 784:
        values.extend([0.0] * (784 - len(values)))
    return values


def encode_input_data(input_data: Sequence[float]) -> str:
    """Encode float32 input data as base64 for the browser shard engine."""
    packed = struct.pack(f"<{len(input_data)}f", *input_data)
    return base64.b64encode(packed).decode("ascii")


def _apply_activation(data: np.ndarray, activation: str) -> np.ndarray:
    if activation == "relu":
        return np.maximum(data, 0)
    if activation == "softmax":
        shifted = data - np.max(data)
        exp_data = np.exp(shifted)
        return exp_data / np.sum(exp_data)
    if activation == "sigmoid":
        return 1 / (1 + np.exp(-data))
    if activation == "tanh":
        return np.tanh(data)
    return data


def execute_shards(input_data: Sequence[float], shards: Sequence[dict]) -> List[np.ndarray]:
    """Execute the deterministic shard model and return each layer output."""
    current = np.array(input_data, dtype=np.float32)
    outputs: List[np.ndarray] = []

    for shard in shards:
        for layer in shard.get("layers", []):
            weights = np.array(layer["weights"], dtype=np.float32).reshape(
                layer["outputShape"][-1],
                layer["inputShape"][-1],
            )
            biases = np.array(layer["biases"], dtype=np.float32)
            current = current @ weights.T + biases
            current = _apply_activation(current, layer.get("activation", "linear")).astype(
                np.float32
            )
            outputs.append(current.copy())

    return outputs
