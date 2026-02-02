"""
Training script for tiny MNIST CNN model for PoUW CAPTCHA.

This script trains a lightweight CNN for MNIST digit classification,
optimized for browser-based inference with small model size (~50KB quantized).

Architecture:
- Conv2D: 8 filters, 3x3 kernel
- MaxPool2D: 2x2
- Conv2D: 16 filters, 3x3 kernel
- MaxPool2D: 2x2
- Flatten
- Dense: 32 units
- Output: 10 units (digits 0-9)

Usage:
    python scripts/train_mnist_model.py --output models/mnist-tiny/
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TinyMNISTModel:
    """Tiny CNN for MNIST digit classification."""

    def __init__(self):
        self.model: keras.Model = None
        self.input_shape = (28, 28, 1)
        self.num_classes = 10
        self.labels = [str(i) for i in range(10)]

    def build(self) -> keras.Model:
        """Build the tiny CNN architecture."""
        model = keras.Sequential([
            # First conv block
            layers.Conv2D(
                8, (3, 3),
                activation='relu',
                input_shape=self.input_shape,
                name='conv1'
            ),
            layers.MaxPooling2D((2, 2), name='pool1'),

            # Second conv block
            layers.Conv2D(
                16, (3, 3),
                activation='relu',
                name='conv2'
            ),
            layers.MaxPooling2D((2, 2), name='pool2'),

            # Flatten and dense layers
            layers.Flatten(name='flatten'),
            layers.Dense(32, activation='relu', name='dense1'),
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ])

        self.model = model
        return model

    def compile(self):
        """Compile the model with optimizer and loss."""
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 10,
        batch_size: int = 128
    ) -> keras.callbacks.History:
        """Train the model."""
        logger.info(f"Training for {epochs} epochs with batch size {batch_size}")

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )
        ]

        history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        return history

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the model on test data."""
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        return {'loss': loss, 'accuracy': accuracy}

    def get_layer_info(self) -> List[Dict]:
        """Get information about each layer for sharding."""
        layer_info = []
        for i, layer in enumerate(self.model.layers):
            info = {
                'index': i,
                'name': layer.name,
                'type': layer.__class__.__name__,
                'output_shape': str(layer.output_shape),
            }

            # Get weights info if applicable
            weights = layer.get_weights()
            if weights:
                info['weights_shape'] = [w.shape for w in weights]
                info['weights_size'] = sum(w.nbytes for w in weights)

            layer_info.append(info)

        return layer_info


def load_and_preprocess_data() -> Tuple[np.ndarray, ...]:
    """Load and preprocess MNIST dataset."""
    logger.info("Loading MNIST dataset...")

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Add channel dimension
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Split training data for validation
    val_size = 5000
    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    x_train = x_train[:-val_size]
    y_train = y_train[:-val_size]

    logger.info(f"Training samples: {len(x_train)}")
    logger.info(f"Validation samples: {len(x_val)}")
    logger.info(f"Test samples: {len(x_test)}")

    return x_train, y_train, x_val, y_val, x_test, y_test


def quantize_model(model: keras.Model) -> tf.lite.Interpreter:
    """Convert model to quantized TFLite for smaller size."""
    logger.info("Quantizing model to INT8...")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.int8]

    # Representative dataset for calibration
    def representative_dataset():
        for _ in range(100):
            data = np.random.rand(1, 28, 28, 1).astype(np.float32)
            yield [data]

    converter.representative_dataset = representative_dataset

    tflite_model = converter.convert()
    return tflite_model


def save_model_shards(
    model: keras.Model,
    output_dir: Path,
    include_ground_truth: bool = True
) -> Dict:
    """Save model as layer-wise shards for federated inference."""
    logger.info("Creating model shards...")

    shards_dir = output_dir / 'shards'
    shards_dir.mkdir(parents=True, exist_ok=True)

    shard_info = {
        'model_name': 'mnist-tiny',
        'version': '1.0.0',
        'total_layers': len(model.layers),
        'shards': []
    }

    # Save full model first
    model.save(output_dir / 'model_full.keras')
    logger.info(f"Saved full model to {output_dir / 'model_full.keras'}")

    # Also save as H5 for compatibility
    model.save(output_dir / 'model.h5')
    logger.info(f"Saved H5 model to {output_dir / 'model.h5'}")

    # Create individual layer shards
    for i, layer in enumerate(model.layers):
        shard_name = f'shard_{i:02d}_{layer.name}'

        # Create a submodel up to this layer
        if i == 0:
            # First layer - just the layer itself
            shard_model = keras.Sequential([layer])
        else:
            # Subsequent layers - build up to this layer
            inputs = keras.Input(shape=model.input_shape[1:])
            x = inputs
            for j in range(i + 1):
                x = model.layers[j](x)
            shard_model = keras.Model(inputs=inputs, outputs=x)

        # Save shard
        shard_path = shards_dir / f'{shard_name}.keras'
        shard_model.save(shard_path)

        # Get shard metadata
        shard_meta = {
            'index': i,
            'name': layer.name,
            'type': layer.__class__.__name__,
            'file': f'shards/{shard_name}.keras',
            'input_shape': str(model.input_shape if i == 0 else model.layers[i-1].output_shape),
            'output_shape': str(layer.output_shape),
        }

        # Calculate shard size
        shard_size = shard_path.stat().st_size
        shard_meta['size_bytes'] = shard_size

        shard_info['shards'].append(shard_meta)
        logger.info(f"Created shard {i}: {layer.name} ({shard_size} bytes)")

    # Save shard manifest
    with open(output_dir / 'shard_manifest.json', 'w') as f:
        json.dump(shard_info, f, indent=2)

    return shard_info


def compute_ground_truth(
    model: keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
    num_samples: int = 1000
) -> Dict:
    """Pre-compute ground truth outputs for validation."""
    logger.info(f"Computing ground truth for {num_samples} samples...")

    # Select random samples
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    x_samples = x_test[indices]
    y_samples = y_test[indices]

    # Compute intermediate outputs for each layer
    ground_truth = {
        'model_name': 'mnist-tiny',
        'version': '1.0.0',
        'num_samples': num_samples,
        'samples': []
    }

    for idx, (x, y) in enumerate(zip(x_samples, y_samples)):
        sample_data = {
            'index': int(idx),
            'true_label': int(y),
            'input_shape': list(x.shape),
            'layer_outputs': []
        }

        # Compute output at each layer
        x_batch = np.expand_dims(x, 0)
        current_output = x_batch

        for layer in model.layers:
            current_output = layer(current_output)

            # Hash the output for efficient comparison
            output_flat = current_output.numpy().flatten()
            output_hash = hash(output_flat.tobytes()) % (2**32)

            layer_output = {
                'layer_name': layer.name,
                'output_shape': list(current_output.shape[1:]),
                'output_hash': int(output_hash),
                'top_prediction': int(np.argmax(output_flat)) if output_flat.size == 10 else None
            }

            sample_data['layer_outputs'].append(layer_output)

        ground_truth['samples'].append(sample_data)

        if (idx + 1) % 100 == 0:
            logger.info(f"Processed {idx + 1}/{num_samples} samples")

    # Save ground truth
    gt_path = output_dir / 'ground_truth.json'
    with open(gt_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    logger.info(f"Saved ground truth to {gt_path}")
    return ground_truth


def export_metadata(
    model: keras.Model,
    output_dir: Path,
    test_accuracy: float
) -> Dict:
    """Export model metadata for client consumption."""
    metadata = {
        'name': 'mnist-tiny',
        'version': '1.0.0',
        'description': 'Tiny CNN for MNIST digit classification',
        'architecture': {
            'input_shape': [28, 28, 1],
            'output_shape': [10],
            'layers': []
        },
        'performance': {
            'test_accuracy': float(test_accuracy),
            'expected_inference_ms': 20,
            'model_size_kb': None  # Will be filled
        },
        'labels': [str(i) for i in range(10)],
        'quantization': {
            'enabled': True,
            'dtype': 'int8'
        }
    }

    # Add layer information
    for layer in model.layers:
        layer_meta = {
            'name': layer.name,
            'type': layer.__class__.__name__,
            'output_shape': str(layer.output_shape)
        }
        metadata['architecture']['layers'].append(layer_meta)

    # Calculate model size
    model_path = output_dir / 'model_full.keras'
    if model_path.exists():
        size_kb = model_path.stat().st_size / 1024
        metadata['performance']['model_size_kb'] = round(size_kb, 2)

    # Save metadata
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {output_dir / 'metadata.json'}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description='Train tiny MNIST model')
    parser.add_argument(
        '--output',
        type=str,
        default='models/mnist-tiny',
        help='Output directory for model files'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Training batch size'
    )
    parser.add_argument(
        '--ground-truth-samples',
        type=int,
        default=1000,
        help='Number of ground truth samples to generate'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    # Load data
    x_train, y_train, x_val, y_val, x_test, y_test = load_and_preprocess_data()

    # Build and train model
    mnist_model = TinyMNISTModel()
    model = mnist_model.build()

    logger.info("Model architecture:")
    model.summary()

    mnist_model.compile()

    history = mnist_model.train(
        x_train, y_train,
        x_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # Evaluate
    results = mnist_model.evaluate(x_test, y_test)
    logger.info(f"Test accuracy: {results['accuracy']:.4f}")
    logger.info(f"Test loss: {results['loss']:.4f}")

    # Save model shards
    shard_info = save_model_shards(model, output_dir)

    # Compute ground truth
    ground_truth = compute_ground_truth(
        model, x_test, y_test,
        output_dir,
        num_samples=args.ground_truth_samples
    )

    # Export metadata
    metadata = export_metadata(model, output_dir, results['accuracy'])

    # Save layer info
    layer_info = mnist_model.get_layer_info()
    with open(output_dir / 'layer_info.json', 'w') as f:
        json.dump(layer_info, f, indent=2)

    logger.info("=" * 50)
    logger.info("Training complete!")
    logger.info(f"Model saved to: {output_dir}")
    logger.info(f"Test accuracy: {results['accuracy']:.2%}")
    logger.info(f"Model size: {metadata['performance']['model_size_kb']:.2f} KB")
    logger.info(f"Number of shards: {len(shard_info['shards'])}")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()

