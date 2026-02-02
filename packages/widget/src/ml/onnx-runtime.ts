/**
 * ONNX Runtime Web for PoUW CAPTCHA
 */

import * as ort from 'onnxruntime-web';
import type { MLModel, ModelMeta, Prediction } from '../types';
import type { ModelConfig } from './engine';
import { Config } from '../core/config';

/**
 * ONNX Runtime Web implementation
 */
export class ONNXRuntime {
  private config: Config;
  private session: ort.InferenceSession | null = null;
  private isInitialized = false;

  constructor(config: Config) {
    this.config = config;
  }

  /**
   * Initialize ONNX Runtime
   */
  private async initialize(): Promise<void> {
    if (this.isInitialized) return;

    // Configure ONNX Runtime
    ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
    ort.env.wasm.simd = true;

    this.isInitialized = true;
    this.config.debug('ONNX Runtime initialized');
  }

  /**
   * Load an ONNX model
   */
  async loadModel(config: ModelConfig): Promise<MLModel> {
    await this.initialize();

    const modelUrl = config.url;

    try {
      // Create session options
      const options: ort.InferenceSession.SessionOptions = {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      };

      // Try to use WebGL if available
      try {
        this.session = await ort.InferenceSession.create(modelUrl, {
          ...options,
          executionProviders: ['webgl', 'wasm'],
        });
        this.config.debug('ONNX using WebGL provider');
      } catch {
        // Fallback to WASM only
        this.session = await ort.InferenceSession.create(modelUrl, options);
        this.config.debug('ONNX using WASM provider');
      }

      this.config.debug('ONNX model loaded', {
        inputNames: this.session.inputNames,
        outputNames: this.session.outputNames,
      });

      return this.createModelInterface(config.meta);
    } catch (error) {
      this.config.error('Failed to load ONNX model:', error);
      throw error;
    }
  }

  /**
   * Create MLModel interface
   */
  private createModelInterface(meta: ModelMeta): MLModel {
    return {
      id: `onnx-${meta.name}-${meta.version}`,
      runtime: 'onnx',
      meta,
      predict: async (input) => this.runInference(input, meta),
      dispose: () => this.disposeSession(),
    };
  }

  /**
   * Run inference
   */
  private async runInference(
    input: Float32Array | ImageData | string,
    meta: ModelMeta
  ): Promise<Prediction> {
    if (!this.session) {
      throw new Error('ONNX session not loaded');
    }

    // Preprocess input
    const inputTensor = this.preprocessInput(input, meta);

    // Get input name
    const inputName = this.session.inputNames[0];

    // Create feeds
    const feeds: ort.InferenceSession.FeedsType = {
      [inputName]: inputTensor,
    };

    // Run inference
    const results = await this.session.run(feeds);

    // Get output
    const outputName = this.session.outputNames[0];
    const output = results[outputName];
    const predictions = output.data as Float32Array;

    // Apply softmax
    const probabilities = this.softmax(Array.from(predictions));

    // Get top-k predictions
    const topK = this.getTopK(probabilities, meta.labels, 5);

    return {
      label: topK[0].label,
      confidence: topK[0].confidence,
      topK,
      rawOutput: Array.from(predictions),
    };
  }

  /**
   * Preprocess input for ONNX model
   */
  private preprocessInput(
    input: Float32Array | ImageData | string,
    meta: ModelMeta
  ): ort.Tensor {
    if (input instanceof ImageData) {
      // Image input
      const [batch, channels, height, width] = meta.inputShape;
      const data = this.imageDataToTensor(input, height, width, channels);
      return new ort.Tensor('float32', data, [batch || 1, channels, height, width]);
    } else if (input instanceof Float32Array) {
      // Raw tensor input
      return new ort.Tensor('float32', input, meta.inputShape);
    } else {
      // Text input - tokenize
      const maxLength = meta.inputShape[1];
      const tokens = this.tokenizeText(input, maxLength);
      return new ort.Tensor('int64', BigInt64Array.from(tokens.map(BigInt)), [
        1,
        maxLength,
      ]);
    }
  }

  /**
   * Convert ImageData to tensor data
   */
  private imageDataToTensor(
    imageData: ImageData,
    targetHeight: number,
    targetWidth: number,
    channels: number
  ): Float32Array {
    // Create canvas for resizing
    const canvas = document.createElement('canvas');
    canvas.width = targetWidth;
    canvas.height = targetHeight;

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Failed to get canvas context');
    }

    // Create temp canvas with original image
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = imageData.width;
    tempCanvas.height = imageData.height;
    const tempCtx = tempCanvas.getContext('2d');
    if (!tempCtx) {
      throw new Error('Failed to get temp canvas context');
    }
    tempCtx.putImageData(imageData, 0, 0);

    // Draw resized
    ctx.drawImage(tempCanvas, 0, 0, targetWidth, targetHeight);
    const resizedData = ctx.getImageData(0, 0, targetWidth, targetHeight);

    // Convert to CHW format (channels first) and normalize
    const tensorData = new Float32Array(channels * targetHeight * targetWidth);

    for (let c = 0; c < channels; c++) {
      for (let h = 0; h < targetHeight; h++) {
        for (let w = 0; w < targetWidth; w++) {
          const srcIdx = (h * targetWidth + w) * 4 + c;
          const dstIdx = c * targetHeight * targetWidth + h * targetWidth + w;
          tensorData[dstIdx] = resizedData.data[srcIdx] / 255.0;
        }
      }
    }

    return tensorData;
  }

  /**
   * Simple tokenization
   */
  private tokenizeText(text: string, maxLength: number): number[] {
    const tokens: number[] = [];
    for (let i = 0; i < Math.min(text.length, maxLength); i++) {
      tokens.push(text.charCodeAt(i));
    }
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
   * Dispose session
   */
  private async disposeSession(): Promise<void> {
    if (this.session) {
      await this.session.release();
      this.session = null;
    }
  }

  /**
   * Dispose runtime
   */
  dispose(): void {
    this.disposeSession();
  }
}
