"""
Train the real MNIST model for PoUW CAPTCHA in pure NumPy.

The production architecture must match what browsers execute layer-by-layer:
    dense 784 -> 128 (relu)
    dense 128 -> 64  (relu)
    dense  64 -> 10  (softmax)

Pure NumPy is used (no TensorFlow dependency) so the exact float32 forward
math the server uses for validation is the same math used in training.

Outputs (models/mnist-tiny/):
    weights.npz     - trained float32 weights (W1,b1,W2,b2,W3,b3)
    manifest.json   - plug-and-play model manifest with REAL SHA-256 checksums:
                      * one checksum per layer (over canonical float32 bytes)
                      * model checksum = sha256(layer_checksum_0 || ... || _n)
                        so a client holding only a segment can still verify it.

Usage:
    python scripts/train_mnist_numpy.py [--epochs 20] [--output models/mnist-tiny]
"""

import argparse
import gzip
import hashlib
import json
import logging
import struct
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_mnist")

MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

LAYER_SPECS = [
    {"name": "dense_hidden_1", "input_size": 784, "output_size": 128, "activation": "relu"},
    {"name": "dense_hidden_2", "input_size": 128, "output_size": 64, "activation": "relu"},
    {"name": "dense_output", "input_size": 64, "output_size": 10, "activation": "softmax"},
]


def download_mnist(data_dir: Path) -> dict:
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, filename in MNIST_FILES.items():
        path = data_dir / filename
        if not path.exists():
            url = f"{MNIST_MIRROR}/{filename}"
            logger.info(f"Downloading {url}")
            urllib.request.urlretrieve(url, path)
        paths[key] = path
    return paths


def load_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Bad magic in {path}: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(count, rows * cols).astype(np.float32) / 255.0


def load_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Bad magic in {path}: {magic}"
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)


def init_params(rng: np.random.Generator) -> list:
    """He-initialised weights for each dense layer (float32)."""
    params = []
    for spec in LAYER_SPECS:
        fan_in = spec["input_size"]
        w = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, spec["output_size"]))
        b = np.zeros(spec["output_size"])
        params.append([w.astype(np.float32), b.astype(np.float32)])
    return params


def forward(x: np.ndarray, params: list) -> tuple:
    """Forward pass; returns (softmax probs, per-layer activations for backprop)."""
    activations = [x]
    h = x
    for i, (w, b) in enumerate(params):
        z = h @ w + b
        if LAYER_SPECS[i]["activation"] == "relu":
            h = np.maximum(z, 0.0)
        else:  # softmax output layer
            z = z - z.max(axis=1, keepdims=True)
            e = np.exp(z)
            h = e / e.sum(axis=1, keepdims=True)
        activations.append(h)
    return h, activations


def train(params, x_train, y_train, x_val, y_val, epochs, batch_size, lr):
    """Mini-batch Adam training with cross-entropy loss."""
    rng = np.random.default_rng(7)
    n = len(x_train)
    # Adam state
    m = [[np.zeros_like(w), np.zeros_like(b)] for w, b in params]
    v = [[np.zeros_like(w), np.zeros_like(b)] for w, b in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0

    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            xb, yb = x_train[idx], y_train[idx]

            probs, acts = forward(xb, params)
            batch = len(xb)
            loss = -np.log(np.clip(probs[np.arange(batch), yb], 1e-9, None)).mean()
            epoch_loss += loss
            batches += 1

            # Backprop
            grad_z = probs.copy()
            grad_z[np.arange(batch), yb] -= 1.0
            grad_z /= batch

            grads = [None] * len(params)
            for i in range(len(params) - 1, -1, -1):
                h_in = acts[i]
                grads[i] = [h_in.T @ grad_z, grad_z.sum(axis=0)]
                if i > 0:
                    grad_h = grad_z @ params[i][0].T
                    grad_z = grad_h * (acts[i] > 0)  # relu derivative

            # Adam update
            step += 1
            for i in range(len(params)):
                for j in range(2):
                    g = grads[i][j]
                    m[i][j] = beta1 * m[i][j] + (1 - beta1) * g
                    v[i][j] = beta2 * v[i][j] + (1 - beta2) * g * g
                    m_hat = m[i][j] / (1 - beta1 ** step)
                    v_hat = v[i][j] / (1 - beta2 ** step)
                    params[i][j] = (
                        params[i][j] - lr * m_hat / (np.sqrt(v_hat) + eps)
                    ).astype(np.float32)

        val_acc = evaluate(params, x_val, y_val)
        logger.info(
            f"epoch {epoch:2d}/{epochs}  loss={epoch_loss / batches:.4f}  val_acc={val_acc:.4f}"
        )
    return params


def evaluate(params, x, y) -> float:
    probs, _ = forward(x, params)
    return float((probs.argmax(axis=1) == y).mean())


def layer_checksum(w: np.ndarray, b: np.ndarray) -> str:
    """
    Canonical layer checksum: sha256 over float32 little-endian bytes in WIRE
    order — weights flattened (output, input) row-major, then biases. This is
    exactly the byte sequence a browser client can rebuild from the JSON shard
    payload via Float32Array, so clients can verify shards without NumPy.
    """
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(w.T, dtype="<f4").tobytes())
    h.update(np.ascontiguousarray(b, dtype="<f4").tobytes())
    return h.hexdigest()


def export(params, output_dir: Path, test_accuracy: float, version: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = {}
    for i, (w, b) in enumerate(params):
        weights[f"W{i}"] = w
        weights[f"b{i}"] = b
    weights_path = output_dir / "weights.npz"
    np.savez(weights_path, **weights)

    layer_checksums = [layer_checksum(w, b) for w, b in params]
    model_checksum = hashlib.sha256("".join(layer_checksums).encode("ascii")).hexdigest()

    manifest = {
        "name": "mnist-tiny",
        "version": version,
        "task_type": "image_classification",
        "labels": [str(i) for i in range(10)],
        "input": {
            "shape": [1, 784],
            "preprocessing": "grayscale 28x28, normalize /255, flatten row-major",
        },
        "weights_file": "weights.npz",
        "checksum": model_checksum,
        "layers": [
            {
                "index": i,
                "name": spec["name"],
                "type": "dense",
                "activation": spec["activation"],
                "input_size": spec["input_size"],
                "output_size": spec["output_size"],
                "checksum": layer_checksums[i],
            }
            for i, spec in enumerate(LAYER_SPECS)
        ],
        "metrics": {"test_accuracy": round(test_accuracy, 4)},
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "weights_file_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
    }

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Train MNIST MLP in pure NumPy")
    parser.add_argument("--output", default="models/mnist-tiny")
    parser.add_argument("--data-dir", default="data/mnist")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--version", default="2.0.0")
    args = parser.parse_args()

    paths = download_mnist(Path(args.data_dir))
    x_train = load_idx_images(paths["train_images"])
    y_train = load_idx_labels(paths["train_labels"])
    x_test = load_idx_images(paths["test_images"])
    y_test = load_idx_labels(paths["test_labels"])

    x_val, y_val = x_train[-5000:], y_train[-5000:]
    x_train, y_train = x_train[:-5000], y_train[:-5000]
    logger.info(f"train={len(x_train)} val={len(x_val)} test={len(x_test)}")

    params = init_params(np.random.default_rng(42))
    start = time.time()
    params = train(params, x_train, y_train, x_val, y_val, args.epochs, args.batch_size, args.lr)
    logger.info(f"Training took {time.time() - start:.1f}s")

    test_acc = evaluate(params, x_test, y_test)
    logger.info(f"Test accuracy: {test_acc:.4f}")

    manifest = export(params, Path(args.output), test_acc, args.version)
    logger.info(f"Model checksum: {manifest['checksum']}")
    logger.info(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
