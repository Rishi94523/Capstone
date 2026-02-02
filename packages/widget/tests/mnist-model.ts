/**
 * MNIST Digit Classification Model for Testing
 * 
 * A minimal neural network trained on MNIST digits 0-9
 * Architecture: 784 inputs (28x28) -> 128 hidden (ReLU) -> 10 outputs (Softmax)
 */

import type { NeuralLayerConfig, ModelShard } from '../src/types';

/**
 * Create a simple MNIST model with random weights for testing
 * In production, this would be loaded from a pre-trained model file
 */
export function createTestMNISTModel(): { layers: NeuralLayerConfig[]; shards: ModelShard[] } {
  const inputSize = 784; // 28x28 pixels
  const hiddenSize = 128;
  const outputSize = 10; // digits 0-9

  // Layer 1: Input -> Hidden (Dense + ReLU)
  const layer1Weights = new Array(inputSize * hiddenSize)
    .fill(0)
    .map(() => (Math.random() - 0.5) * 0.1);
  const layer1Biases = new Array(hiddenSize).fill(0).map(() => Math.random() * 0.1);

  const layer1: NeuralLayerConfig = {
    name: 'dense_hidden',
    type: 'dense',
    weights: layer1Weights,
    biases: layer1Biases,
    inputShape: [1, inputSize],
    outputShape: [1, hiddenSize],
    activation: 'relu',
  };

  // Layer 2: Hidden -> Output (Dense + Softmax)
  const layer2Weights = new Array(hiddenSize * outputSize)
    .fill(0)
    .map(() => (Math.random() - 0.5) * 0.1);
  const layer2Biases = new Array(outputSize).fill(0).map(() => Math.random() * 0.1);

  const layer2: NeuralLayerConfig = {
    name: 'dense_output',
    type: 'dense',
    weights: layer2Weights,
    biases: layer2Biases,
    inputShape: [1, hiddenSize],
    outputShape: [1, outputSize],
    activation: 'softmax',
  };

  // Create shards - can be split across layers for distributed computation
  const shard1: ModelShard = {
    index: 0,
    name: 'hidden_layer',
    layerType: 'dense',
    inputShape: [1, inputSize],
    outputShape: [1, hiddenSize],
    layers: [layer1],
  };

  const shard2: ModelShard = {
    index: 1,
    name: 'output_layer',
    layerType: 'dense',
    inputShape: [1, hiddenSize],
    outputShape: [1, outputSize],
    layers: [layer2],
  };

  return {
    layers: [layer1, layer2],
    shards: [shard1, shard2],
  };
}

/**
 * Generate test input data (simulated 28x28 grayscale image)
 */
export function createTestMNISTInput(digit: number = 5): Float32Array {
  // Create a simple pattern that represents a digit
  const input = new Float32Array(784);
  
  // Create a simple vertical/horizontal pattern based on digit
  for (let i = 0; i < 28; i++) {
    for (let j = 0; j < 28; j++) {
      const idx = i * 28 + j;
      
      // Simple patterns for different digits
      if (digit === 0) {
        // Circle pattern
        const dx = j - 14;
        const dy = i - 14;
        input[idx] = Math.abs(Math.sqrt(dx * dx + dy * dy) - 10) < 2 ? 1.0 : 0.0;
      } else if (digit === 1) {
        // Vertical line
        input[idx] = Math.abs(j - 14) < 2 ? 1.0 : 0.0;
      } else {
        // Random noise for other digits
        input[idx] = Math.random() * 0.5;
      }
    }
  }
  
  return input;
}

/**
 * Convert Float32Array to base64 string for network transmission
 */
export function inputToBase64(input: Float32Array): string {
  const bytes = new Uint8Array(input.buffer);
  let binary = '';
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

/**
 * MNIST digit labels
 */
export const MNIST_LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];

/**
 * Sample test cases with expected outputs
 */
export const MNIST_TEST_CASES = [
  { digit: 0, description: 'Circle pattern' },
  { digit: 1, description: 'Vertical line' },
  { digit: 2, description: 'Diagonal pattern' },
  { digit: 3, description: 'Curved pattern' },
  { digit: 4, description: 'Cross pattern' },
  { digit: 5, description: 'Horizontal pattern' },
  { digit: 6, description: 'Loop pattern' },
  { digit: 7, description: 'Angle pattern' },
  { digit: 8, description: 'Figure eight' },
  { digit: 9, description: 'Round pattern' },
];
