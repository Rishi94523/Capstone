/**
 * Federated Inference Shard Engine
 *
 * Executes model shards (partial layer-wise computation) for CAPTCHA verification.
 * Replaces simulated proof-of-work with actual ML inference computation.
 */

import type {
  ModelShard,
  ShardTask,
  InferenceProof,
  Prediction,
} from '../types';
import { Config } from '../core/config';
import { hashData } from '../utils/crypto';

/**
 * Result from executing a model shard
 */
export interface ShardExecutionResult {
  /** Layer outputs for each executed layer */
  layerOutputs: Float32Array[];
  /** Final prediction if applicable */
  prediction?: Prediction;
  /** Proof of computation */
  proof: InferenceProof;
  /** Execution timing */
  timing: {
    totalMs: number;
    layerMs: number[];
  };
}

/**
 * Simple neural network layer implementation for shard execution
 * Uses pure JavaScript for maximum compatibility
 */
class NeuralLayer {
  public readonly name: string;
  public readonly type: string;
  public readonly weights: Float32Array;
  public readonly biases: Float32Array;
  public readonly inputShape: number[];
  public readonly outputShape: number[];
  public readonly activation: string;

  constructor(config: ModelShard['layers'][0]) {
    this.name = config.name;
    this.type = config.type;
    this.weights = new Float32Array(config.weights);
    this.biases = new Float32Array(config.biases);
    this.inputShape = config.inputShape;
    this.outputShape = config.outputShape;
    this.activation = config.activation;
  }

  /**
   * Execute forward pass through this layer
   */
  forward(input: Float32Array): Float32Array {
    const startTime = performance.now();
    let output: Float32Array;

    switch (this.type) {
      case 'conv2d':
        output = this.conv2dForward(input);
        break;
      case 'dense':
      case 'fully_connected':
        output = this.denseForward(input);
        break;
      case 'maxpool2d':
        output = this.maxPoolForward(input);
        break;
      case 'flatten':
        output = this.flattenForward(input);
        break;
      default:
        throw new Error(`Unsupported layer type: ${this.type}`);
    }

    // Apply activation
    output = this.applyActivation(output);

    const endTime = performance.now();
    this.lastExecutionTime = endTime - startTime;

    return output;
  }

  private lastExecutionTime = 0;

  getExecutionTime(): number {
    return this.lastExecutionTime;
  }

  /**
   * 2D Convolution forward pass (simplified)
   */
  private conv2dForward(input: Float32Array): Float32Array {
    const [batch, inHeight, inWidth, inChannels] = this.inputShape;
    const [_, outHeight, outWidth, outChannels] = this.outputShape;
    const kernelSize = Math.sqrt(
      this.weights.length / (outChannels * inChannels)
    );
    const stride = 1;
    const padding = 0;

    const output = new Float32Array(batch * outHeight * outWidth * outChannels);

    // Simplified convolution - in production use TensorFlow.js or ONNX
    for (let b = 0; b < batch; b++) {
      for (let oc = 0; oc < outChannels; oc++) {
        for (let oh = 0; oh < outHeight; oh++) {
          for (let ow = 0; ow < outWidth; ow++) {
            let sum = this.biases[oc];
            for (let ic = 0; ic < inChannels; ic++) {
              for (let kh = 0; kh < kernelSize; kh++) {
                for (let kw = 0; kw < kernelSize; kw++) {
                  const ih = oh * stride + kh - padding;
                  const iw = ow * stride + kw - padding;
                  if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                    const inputIdx =
                      ((b * inHeight + ih) * inWidth + iw) * inChannels + ic;
                    const weightIdx =
                      ((oc * inChannels + ic) * kernelSize + kh) * kernelSize +
                      kw;
                    sum += input[inputIdx] * this.weights[weightIdx];
                  }
                }
              }
            }
            const outIdx =
              ((b * outHeight + oh) * outWidth + ow) * outChannels + oc;
            output[outIdx] = sum;
          }
        }
      }
    }

    return output;
  }

  /**
   * Dense/Fully Connected forward pass
   */
  private denseForward(input: Float32Array): Float32Array {
    const inputSize = this.inputShape[this.inputShape.length - 1];
    const outputSize = this.outputShape[this.outputShape.length - 1];
    const output = new Float32Array(outputSize);

    for (let o = 0; o < outputSize; o++) {
      let sum = this.biases[o];
      for (let i = 0; i < inputSize; i++) {
        sum += input[i] * this.weights[o * inputSize + i];
      }
      output[o] = sum;
    }

    return output;
  }

  /**
   * Max Pooling forward pass
   */
  private maxPoolForward(input: Float32Array): Float32Array {
    const [batch, inHeight, inWidth, channels] = this.inputShape;
    const poolSize = 2;
    const stride = 2;
    const outHeight = Math.floor(inHeight / stride);
    const outWidth = Math.floor(inWidth / stride);

    const output = new Float32Array(batch * outHeight * outWidth * channels);

    for (let b = 0; b < batch; b++) {
      for (let c = 0; c < channels; c++) {
        for (let oh = 0; oh < outHeight; oh++) {
          for (let ow = 0; ow < outWidth; ow++) {
            let maxVal = -Infinity;
            for (let ph = 0; ph < poolSize; ph++) {
              for (let pw = 0; pw < poolSize; pw++) {
                const ih = oh * stride + ph;
                const iw = ow * stride + pw;
                const idx = ((b * inHeight + ih) * inWidth + iw) * channels + c;
                maxVal = Math.max(maxVal, input[idx]);
              }
            }
            const outIdx =
              ((b * outHeight + oh) * outWidth + ow) * channels + c;
            output[outIdx] = maxVal;
          }
        }
      }
    }

    return output;
  }

  /**
   * Flatten layer
   */
  private flattenForward(input: Float32Array): Float32Array {
    // Just reshape - data stays the same
    return input;
  }

  /**
   * Apply activation function
   */
  private applyActivation(data: Float32Array): Float32Array {
    switch (this.activation) {
      case 'relu':
        return data.map((x) => Math.max(0, x));
      case 'softmax':
        return this.softmax(data);
      case 'sigmoid':
        return data.map((x) => 1 / (1 + Math.exp(-x)));
      case 'tanh':
        return data.map((x) => Math.tanh(x));
      case 'linear':
      default:
        return data;
    }
  }

  /**
   * Softmax activation
   */
  private softmax(data: Float32Array): Float32Array {
    let maxVal = data[0];
    for (let i = 1; i < data.length; i++) {
      if (data[i] > maxVal) maxVal = data[i];
    }
    const expData = new Float32Array(data.length);
    let sumExp = 0;
    for (let i = 0; i < data.length; i++) {
      expData[i] = Math.exp(data[i] - maxVal);
      sumExp += expData[i];
    }
    const result = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) {
      result[i] = expData[i] / sumExp;
    }
    return result;
  }
}

/**
 * Shard Inference Engine
 *
 * Executes partial model computations to prove actual ML work
 */
export class ShardInferenceEngine {
  private config: Config;
  private layers: NeuralLayer[] = [];

  constructor(config: Config) {
    this.config = config;
  }

  /**
   * Load model shards for execution
   */
  loadShards(shards: ModelShard[]): void {
    this.layers = [];
    for (const shard of shards) {
      for (const layerConfig of shard.layers) {
        this.layers.push(new NeuralLayer(layerConfig));
      }
    }
    this.config.debug(`Loaded ${this.layers.length} layers from shards`);
  }

  /**
   * Execute model shards on input data
   */
  async executeShards(task: ShardTask): Promise<ShardExecutionResult> {
    const totalStartTime = performance.now();

    // Load shards
    this.loadShards(task.shards);

    // Decode input data
    const input = this.decodeInput(task.inputData, task.inputShape);

    // Execute layers
    const layerOutputs: Float32Array[] = [];
    const layerTimes: number[] = [];
    let currentOutput = input;

    for (let i = 0; i < this.layers.length; i++) {
      const layer = this.layers[i];
      this.config.debug(
        `Executing layer ${i + 1}/${this.layers.length}: ${layer.name}`
      );

      const layerStartTime = performance.now();
      currentOutput = layer.forward(currentOutput);
      const layerEndTime = performance.now();

      layerOutputs.push(currentOutput);
      layerTimes.push(layerEndTime - layerStartTime);

      // Report progress if callback provided
      if (task.onProgress) {
        task.onProgress((i + 1) / this.layers.length);
      }
    }

    // Generate prediction from final output
    const prediction = this.generatePrediction(
      currentOutput,
      task.labels || []
    );

    // Generate proof of computation
    const proof = await this.generateProof(
      task.taskId,
      task.sampleId,
      task.expectedLayers,
      layerOutputs,
      prediction
    );

    const totalEndTime = performance.now();

    return {
      layerOutputs,
      prediction,
      proof,
      timing: {
        totalMs: totalEndTime - totalStartTime,
        layerMs: layerTimes,
      },
    };
  }

  /**
   * Decode base64 input data to Float32Array
   */
  private decodeInput(data: string, _shape: number[]): Float32Array {
    // Remove data URL prefix if present
    const base64Data = data.replace(/^data:.*;base64,/, '');

    // Decode base64
    const binaryString = atob(base64Data);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }

    // Convert to float32 (assuming normalized [0, 1] or [-1, 1] values)
    return new Float32Array(bytes.buffer);
  }

  /**
   * Generate prediction from output logits
   */
  private generatePrediction(
    output: Float32Array,
    labels: string[]
  ): Prediction {
    // Find top prediction
    let maxIdx = 0;
    let maxVal = output[0];
    for (let i = 1; i < output.length; i++) {
      if (output[i] > maxVal) {
        maxVal = output[i];
        maxIdx = i;
      }
    }

    // Get top-k predictions
    const topK: { label: string; confidence: number }[] = [];
    const entries: { val: number; idx: number }[] = [];
    for (let i = 0; i < output.length; i++) {
      entries.push({ val: output[i], idx: i });
    }
    entries.sort((a, b) => b.val - a.val);
    const topEntries = entries.slice(0, Math.min(5, output.length));

    for (let i = 0; i < topEntries.length; i++) {
      const { val, idx } = topEntries[i];
      topK.push({
        label: labels[idx] || `class_${idx}`,
        confidence: Math.round(val * 1000) / 1000,
      });
    }

    return {
      label: labels[maxIdx] || `class_${maxIdx}`,
      confidence: Math.round(maxVal * 1000) / 1000,
      topK,
    };
  }

  /**
   * Generate cryptographic proof of inference computation
   */
  private async generateProof(
    taskId: string,
    sampleId: string,
    expectedLayers: number,
    layerOutputs: Float32Array[],
    prediction: Prediction
  ): Promise<InferenceProof> {
    // Hash each layer output
    const outputHashes: string[] = [];
    for (const output of layerOutputs) {
      const hash = await this.hashTensor(output);
      outputHashes.push(hash);
    }

    // Generate combined proof hash
    const proofData = [
      taskId,
      sampleId,
      expectedLayers.toString(),
      ...outputHashes,
      JSON.stringify(prediction),
    ].join(':');

    const proofHash = await hashData(proofData);

    return {
      taskId,
      sampleId,
      layerCount: layerOutputs.length,
      outputHashes,
      predictionHash: await hashData(JSON.stringify(prediction)),
      proofHash,
      timestamp: Date.now(),
    };
  }

  /**
   * Hash a tensor (Float32Array)
   */
  private async hashTensor(tensor: Float32Array): Promise<string> {
    // Convert to array buffer for hashing
    const arrayBuffer = tensor.buffer.slice(
      tensor.byteOffset,
      tensor.byteOffset + tensor.byteLength
    );
    return await hashData(arrayBuffer as ArrayBuffer);
  }

  /**
   * Dispose resources
   */
  dispose(): void {
    this.layers = [];
  }
}

/**
 * Utility to check if shard engine is supported
 */
export function isShardEngineSupported(): boolean {
  return (
    typeof Float32Array !== 'undefined' &&
    typeof atob !== 'undefined' &&
    typeof crypto !== 'undefined' &&
    typeof crypto.subtle !== 'undefined'
  );
}
