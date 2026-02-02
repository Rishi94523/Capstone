/**
 * Unified ML Engine for PoUW CAPTCHA
 *
 * Provides a unified interface for TensorFlow.js and ONNX Runtime Web,
 * with automatic fallback between runtimes. Also includes shard-based
 * inference for adaptive ML CAPTCHA verification.
 */

import type {
  MLModel,
  ModelMeta,
  Prediction,
  RuntimeType,
  ShardTask,
} from '../types';
import type { ShardExecutionResult } from './shard-engine';
import { ShardInferenceEngine } from './shard-engine';
import { Config } from '../core/config';
import { TFJSRuntime } from './tfjs-runtime';
import { ONNXRuntime } from './onnx-runtime';

/**
 * Model configuration for loading
 */
export interface ModelConfig {
  url: string;
  meta: ModelMeta;
  preferredRuntime?: RuntimeType;
}

/**
 * ML Engine - Unified interface for browser ML inference
 */
export class MLEngine {
  private config: Config;
  private tfjsRuntime: TFJSRuntime | null = null;
  private onnxRuntime: ONNXRuntime | null = null;
  private shardEngine: ShardInferenceEngine;
  private currentModel: MLModel | null = null;
  private availableRuntimes: RuntimeType[] = [];

  constructor(config: Config) {
    this.config = config;
    this.shardEngine = new ShardInferenceEngine(config);
    this.detectAvailableRuntimes();
  }

  /**
   * Detect which ML runtimes are available in this browser
   */
  private async detectAvailableRuntimes(): Promise<void> {
    this.availableRuntimes = [];

    // Check TensorFlow.js
    if (await this.checkTFJSAvailable()) {
      this.availableRuntimes.push('tfjs');
      this.config.debug('TensorFlow.js runtime available');
    }

    // Check ONNX Runtime
    if (await this.checkONNXAvailable()) {
      this.availableRuntimes.push('onnx');
      this.config.debug('ONNX Runtime Web available');
    }

    if (this.availableRuntimes.length === 0) {
      this.config.error('No ML runtimes available');
    }
  }

  /**
   * Check if TensorFlow.js is available
   */
  private async checkTFJSAvailable(): Promise<boolean> {
    try {
      // Check for WebGL support
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');

      if (!gl) {
        this.config.debug('WebGL not available, TF.js will use CPU backend');
      }

      return true;
    } catch {
      return false;
    }
  }

  /**
   * Check if ONNX Runtime is available
   */
  private async checkONNXAvailable(): Promise<boolean> {
    try {
      // Check for WebAssembly support
      if (typeof WebAssembly !== 'object') {
        return false;
      }

      return true;
    } catch {
      return false;
    }
  }

  /**
   * Load a model with automatic runtime selection
   */
  async loadModel(config: ModelConfig): Promise<MLModel> {
    const runtime = this.selectRuntime(config);

    this.config.debug(`Loading model with ${runtime} runtime`, {
      url: config.url,
      model: config.meta.name,
    });

    try {
      if (runtime === 'tfjs') {
        this.tfjsRuntime = new TFJSRuntime(this.config);
        this.currentModel = await this.tfjsRuntime.loadModel(config);
      } else {
        this.onnxRuntime = new ONNXRuntime(this.config);
        this.currentModel = await this.onnxRuntime.loadModel(config);
      }

      this.config.debug('Model loaded successfully', {
        id: this.currentModel.id,
        runtime: this.currentModel.runtime,
      });

      return this.currentModel;
    } catch (error) {
      this.config.error(`Failed to load model with ${runtime}:`, error);

      // Try fallback runtime
      const fallbackRuntime = this.getFallbackRuntime(runtime);
      if (fallbackRuntime) {
        this.config.debug(`Attempting fallback to ${fallbackRuntime}`);
        return this.loadModelWithRuntime(config, fallbackRuntime);
      }

      throw error;
    }
  }

  /**
   * Load model with specific runtime
   */
  private async loadModelWithRuntime(
    config: ModelConfig,
    runtime: RuntimeType
  ): Promise<MLModel> {
    if (runtime === 'tfjs') {
      this.tfjsRuntime = new TFJSRuntime(this.config);
      this.currentModel = await this.tfjsRuntime.loadModel(config);
    } else {
      this.onnxRuntime = new ONNXRuntime(this.config);
      this.currentModel = await this.onnxRuntime.loadModel(config);
    }

    return this.currentModel;
  }

  /**
   * Select the best runtime for this model
   */
  private selectRuntime(config: ModelConfig): RuntimeType {
    // Use preferred runtime if specified and available
    if (
      config.preferredRuntime &&
      this.availableRuntimes.includes(config.preferredRuntime)
    ) {
      return config.preferredRuntime;
    }

    // Determine from model URL
    if (config.url.endsWith('.onnx')) {
      if (this.availableRuntimes.includes('onnx')) {
        return 'onnx';
      }
    }

    // Default to TF.js if available
    if (this.availableRuntimes.includes('tfjs')) {
      return 'tfjs';
    }

    // Fallback to ONNX
    if (this.availableRuntimes.includes('onnx')) {
      return 'onnx';
    }

    throw new Error('No ML runtime available');
  }

  /**
   * Get fallback runtime
   */
  private getFallbackRuntime(currentRuntime: RuntimeType): RuntimeType | null {
    const fallback = currentRuntime === 'tfjs' ? 'onnx' : 'tfjs';
    return this.availableRuntimes.includes(fallback) ? fallback : null;
  }

  /**
   * Run inference on current model
   */
  async predict(input: Float32Array | ImageData | string): Promise<Prediction> {
    if (!this.currentModel) {
      throw new Error('No model loaded');
    }

    return this.currentModel.predict(input);
  }

  /**
   * Compute gradient (for training tasks)
   */
  async computeGradient(
    input: Float32Array | ImageData | string,
    target: number
  ): Promise<Float32Array> {
    if (!this.currentModel) {
      throw new Error('No model loaded');
    }

    if (!this.currentModel.computeGradient) {
      throw new Error('Current model does not support gradient computation');
    }

    return this.currentModel.computeGradient(input, target);
  }

  /**
   * Execute a shard task using the ShardInferenceEngine
   *
   * This is the main entry point for shard-based CAPTCHA verification.
   * Replaces full model inference with partial layer-wise computation.
   *
   * @param task - The shard task containing input data and model shards
   * @returns The execution result with computation proof and timing
   */
  async executeShardTask(task: ShardTask): Promise<ShardExecutionResult> {
    this.config.debug('Executing shard task', {
      taskId: task.taskId,
      modelName: task.modelName,
      shardCount: task.shards.length,
    });

    try {
      const result = await this.shardEngine.executeShards(task);

      this.config.debug('Shard task execution complete', {
        taskId: task.taskId,
        layerCount: result.layerOutputs.length,
        totalTimeMs: result.timing.totalMs,
      });

      return result;
    } catch (error) {
      this.config.error('Shard task execution failed:', error);
      throw error;
    }
  }

  /**
   * Check if shard engine is supported in this browser
   */
  isShardEngineSupported(): boolean {
    return (
      typeof Float32Array !== 'undefined' &&
      typeof atob !== 'undefined' &&
      typeof crypto !== 'undefined' &&
      typeof crypto.subtle !== 'undefined'
    );
  }

  /**
   * Dispose all resources
   */
  dispose(): void {
    this.shardEngine.dispose();
    this.currentModel?.dispose();
    this.currentModel = null;
    this.tfjsRuntime?.dispose();
    this.tfjsRuntime = null;
    this.onnxRuntime?.dispose();
    this.onnxRuntime = null;
  }
}
