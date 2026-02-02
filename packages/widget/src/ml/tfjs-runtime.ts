/**
 * TensorFlow.js Runtime for PoUW CAPTCHA
 */

import * as tf from '@tensorflow/tfjs';
import type { MLModel, ModelMeta, Prediction } from '../types';
import type { ModelConfig } from './engine';
import { Config } from '../core/config';

/**
 * TensorFlow.js Runtime implementation
 */
export class TFJSRuntime {
  private config: Config;
  private model: tf.LayersModel | tf.GraphModel | null = null;
  private isInitialized = false;

  constructor(config: Config) {
    this.config = config;
  }

  /**
   * Initialize TensorFlow.js backend
   */
  private async initialize(): Promise<void> {
    if (this.isInitialized) return;

    try {
      // Try WebGL first
      await tf.setBackend('webgl');
      this.config.debug('TF.js using WebGL backend');
    } catch {
      try {
        // Fallback to WebAssembly
        await tf.setBackend('wasm');
        this.config.debug('TF.js using WASM backend');
      } catch {
        // Fallback to CPU
        await tf.setBackend('cpu');
        this.config.debug('TF.js using CPU backend');
      }
    }

    await tf.ready();
    this.isInitialized = true;

    this.config.debug('TF.js initialized', {
      backend: tf.getBackend(),
      memory: tf.memory(),
    });
  }

  /**
   * Load a TensorFlow.js model
   */
  async loadModel(config: ModelConfig): Promise<MLModel> {
    await this.initialize();

    const modelUrl = config.url;

    try {
      // Try loading as LayersModel first
      this.model = await tf.loadLayersModel(modelUrl);
      this.config.debug('Loaded as LayersModel');
    } catch {
      // Try loading as GraphModel
      this.model = await tf.loadGraphModel(modelUrl);
      this.config.debug('Loaded as GraphModel');
    }

    return this.createModelInterface(config.meta);
  }

  /**
   * Create MLModel interface
   */
  private createModelInterface(meta: ModelMeta): MLModel {
    return {
      id: `tfjs-${meta.name}-${meta.version}`,
      runtime: 'tfjs',
      meta,
      predict: async (input) => this.runInference(input, meta),
      computeGradient: async (input, target) =>
        this.computeGradient(input, target, meta),
      dispose: () => this.disposeModel(),
    };
  }

  /**
   * Run inference
   */
  private async runInference(
    input: Float32Array | ImageData | string,
    meta: ModelMeta
  ): Promise<Prediction> {
    if (!this.model) {
      throw new Error('Model not loaded');
    }

    return tf.tidy(() => {
      // Preprocess input
      const inputTensor = this.preprocessInput(input, meta);

      // Run inference
      const output = this.model!.predict(inputTensor) as tf.Tensor;

      // Get predictions
      const predictions = output.dataSync() as Float32Array;

      // Apply softmax if needed
      const probabilities = this.softmax(Array.from(predictions));

      // Get top-k predictions
      const topK = this.getTopK(probabilities, meta.labels, 5);

      return {
        label: topK[0].label,
        confidence: topK[0].confidence,
        topK,
        rawOutput: Array.from(predictions),
      };
    });
  }

  /**
   * Preprocess input for model
   */
  private preprocessInput(
    input: Float32Array | ImageData | string,
    meta: ModelMeta
  ): tf.Tensor {
    if (input instanceof ImageData) {
      // Image input
      const [height, width, channels] = meta.inputShape.slice(1);
      let tensor = tf.browser.fromPixels(input, channels);

      // Resize if needed
      if (input.width !== width || input.height !== height) {
        tensor = tf.image.resizeBilinear(tensor, [height, width]);
      }

      // Normalize to [0, 1]
      tensor = tensor.toFloat().div(255);

      // Add batch dimension
      return tensor.expandDims(0);
    } else if (input instanceof Float32Array) {
      // Raw tensor input
      return tf.tensor(input, meta.inputShape);
    } else {
      // Text input - tokenize (simplified)
      const tokens = this.tokenizeText(input, meta.inputShape[1]);
      return tf.tensor2d([tokens], [1, tokens.length]);
    }
  }

  /**
   * Simple tokenization (placeholder)
   */
  private tokenizeText(text: string, maxLength: number): number[] {
    // Simple character-level tokenization
    const tokens: number[] = [];
    for (let i = 0; i < Math.min(text.length, maxLength); i++) {
      tokens.push(text.charCodeAt(i));
    }
    // Pad to maxLength
    while (tokens.length < maxLength) {
      tokens.push(0);
    }
    return tokens;
  }

  /**
   * Softmax function
   */
  private softmax(arr: number[]): number[] {
    const max = Math.max(...arr);
    const exps = arr.map((x) => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((x) => x / sum);
  }

  /**
   * Get top-K predictions
   */
  private getTopK(
    probabilities: number[],
    labels: string[],
    k: number
  ): Array<{ label: string; confidence: number }> {
    const indexed = probabilities.map((prob, idx) => ({
      label: labels[idx] || `class_${idx}`,
      confidence: prob,
    }));

    indexed.sort((a, b) => b.confidence - a.confidence);

    return indexed.slice(0, k);
  }

  /**
   * Compute gradient for training
   */
  private async computeGradient(
    input: Float32Array | ImageData | string,
    target: number,
    meta: ModelMeta
  ): Promise<Float32Array> {
    if (!this.model || !(this.model instanceof tf.LayersModel)) {
      throw new Error('Gradient computation requires a LayersModel');
    }

    const model = this.model as tf.LayersModel;

    return tf.tidy(() => {
      const inputTensor = this.preprocessInput(input, meta);
      const targetTensor = tf.oneHot([target], meta.labels.length);

      // Compute gradients
      const gradientFn = tf.valueAndGrads(
        (inputs: tf.Tensor[]) => {
          const pred = model.predict(inputs[0]) as tf.Tensor;
          return tf.losses.softmaxCrossEntropy(targetTensor, pred);
        }
      );

      const { grads } = gradientFn([inputTensor]);

      // Flatten all gradients
      const flatGradients: number[] = [];
      for (const tensor of Object.values(grads)) {
        if (tensor) {
          flatGradients.push(...Array.from(tensor.dataSync()));
        }
      }

      return new Float32Array(flatGradients);
    });
  }

  /**
   * Dispose model
   */
  private disposeModel(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
  }

  /**
   * Dispose runtime
   */
  dispose(): void {
    this.disposeModel();
    // Note: Don't dispose TF.js entirely as it may be used elsewhere
  }
}
