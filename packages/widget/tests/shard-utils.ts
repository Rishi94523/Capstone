/**
 * Model Sharding Utility
 * 
 * Splits ML models into shards for distributed client-side computation.
 * Used by the server to prepare shard tasks for CAPTCHA clients.
 */

import type { ModelShard, NeuralLayerConfig, ShardTask } from '../src/types';

/**
 * Configuration for model sharding
 */
export interface ShardConfig {
  /** Maximum layers per shard */
  maxLayersPerShard: number;
  /** Minimum computation time per shard (ms) */
  minComputationTimeMs: number;
  /** Target difficulty level */
  difficulty: 'easy' | 'medium' | 'hard';
}

/**
 * Shard a complete model into partial computations
 */
export function shardModel(
  layers: NeuralLayerConfig[],
  config: ShardConfig
): ModelShard[] {
  const shards: ModelShard[] = [];
  let currentShardLayers: NeuralLayerConfig[] = [];
  let shardIndex = 0;

  for (let i = 0; i < layers.length; i++) {
    currentShardLayers.push(layers[i]);

    // Check if we should create a shard
    const shouldCreateShard =
      currentShardLayers.length >= config.maxLayersPerShard ||
      i === layers.length - 1;

    if (shouldCreateShard && currentShardLayers.length > 0) {
      const firstLayer = currentShardLayers[0];
      const lastLayer = currentShardLayers[currentShardLayers.length - 1];

      const shard: ModelShard = {
        index: shardIndex,
        name: `shard_${shardIndex}_${firstLayer.name}_to_${lastLayer.name}`,
        layerType: firstLayer.type,
        inputShape: firstLayer.inputShape,
        outputShape: lastLayer.outputShape,
        layers: [...currentShardLayers],
      };

      shards.push(shard);
      currentShardLayers = [];
      shardIndex++;
    }
  }

  return shards;
}

/**
 * Create a shard task from model shards
 */
export function createShardTask(
  modelName: string,
  modelVersion: string,
  allShards: ModelShard[],
  inputData: Float32Array,
  labels: string[],
  difficulty: 'easy' | 'medium' | 'hard',
  clientCapability: 'low' | 'medium' | 'high'
): ShardTask {
  // Determine how many layers to assign based on difficulty and capability
  const layerAllocation = getLayerAllocation(difficulty, clientCapability);
  
  // Select shards to assign (simple strategy: consecutive from beginning)
  const assignedShards: ModelShard[] = [];
  let totalLayers = 0;
  
  for (const shard of allShards) {
    if (totalLayers >= layerAllocation.maxLayers) break;
    assignedShards.push(shard);
    totalLayers += shard.layers.length;
  }

  // Convert input to base64
  const inputBase64 = float32ArrayToBase64(inputData);

  return {
    taskId: generateTaskId(),
    sampleId: generateSampleId(),
    modelName,
    modelVersion,
    shards: assignedShards,
    inputData: inputBase64,
    inputShape: assignedShards[0]?.inputShape || [1, 784],
    expectedLayers: totalLayers,
    difficulty,
    expectedTimeMs: layerAllocation.expectedTimeMs,
    groundTruthKey: generateGroundTruthKey(),
    labels,
  };
}

/**
 * Get layer allocation based on difficulty and client capability
 */
function getLayerAllocation(
  difficulty: 'easy' | 'medium' | 'hard',
  capability: 'low' | 'medium' | 'high'
): { maxLayers: number; expectedTimeMs: number } {
  const allocationTable: Record<string, { maxLayers: number; expectedTimeMs: number }> = {
    'easy-low': { maxLayers: 1, expectedTimeMs: 200 },
    'easy-medium': { maxLayers: 2, expectedTimeMs: 300 },
    'easy-high': { maxLayers: 3, expectedTimeMs: 400 },
    'medium-low': { maxLayers: 2, expectedTimeMs: 400 },
    'medium-medium': { maxLayers: 4, expectedTimeMs: 600 },
    'medium-high': { maxLayers: 6, expectedTimeMs: 800 },
    'hard-low': { maxLayers: 3, expectedTimeMs: 600 },
    'hard-medium': { maxLayers: 6, expectedTimeMs: 1000 },
    'hard-high': { maxLayers: 10, expectedTimeMs: 1500 },
  };

  return allocationTable[`${difficulty}-${capability}`] || { maxLayers: 2, expectedTimeMs: 500 };
}

/**
 * Estimate computation time for a shard
 */
export function estimateShardComputationTime(shard: ModelShard): number {
  let totalOperations = 0;

  for (const layer of shard.layers) {
    const inputSize = layer.inputShape.reduce((a, b) => a * b, 1);
    const outputSize = layer.outputShape.reduce((a, b) => a * b, 1);
    
    switch (layer.type) {
      case 'dense':
      case 'fully_connected':
        // Matrix multiplication: input_size * output_size
        totalOperations += inputSize * outputSize;
        break;
      case 'conv2d':
        // Convolution: kernel operations
        const kernelSize = Math.sqrt(layer.weights.length / (inputSize * outputSize));
        totalOperations += inputSize * outputSize * kernelSize * kernelSize;
        break;
      default:
        // Other layers are cheaper
        totalOperations += inputSize;
    }
  }

  // Rough estimate: ~1ms per 1000 operations on modern CPU
  return Math.max(10, totalOperations / 1000);
}

/**
 * Validate shard integrity
 */
export function validateShard(shard: ModelShard): boolean {
  // Check required fields
  if (!shard.name || !shard.layers || shard.layers.length === 0) {
    return false;
  }

  // Validate layer connectivity
  for (let i = 1; i < shard.layers.length; i++) {
    const prevLayer = shard.layers[i - 1];
    const currLayer = shard.layers[i];

    // Output shape of previous should match input shape of current
    const prevOutput = prevLayer.outputShape.slice(-1)[0];
    const currInput = currLayer.inputShape.slice(-1)[0];

    if (prevOutput !== currInput) {
      return false;
    }
  }

  return true;
}

/**
 * Merge multiple shards back into a complete model
 */
export function mergeShards(shards: ModelShard[]): NeuralLayerConfig[] {
  const sortedShards = [...shards].sort((a, b) => a.index - b.index);
  const allLayers: NeuralLayerConfig[] = [];

  for (const shard of sortedShards) {
    allLayers.push(...shard.layers);
  }

  return allLayers;
}

/**
 * Convert Float32Array to base64 string
 */
function float32ArrayToBase64(arr: Float32Array): string {
  const bytes = new Uint8Array(arr.buffer);
  let binary = '';
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

/**
 * Generate unique task ID
 */
function generateTaskId(): string {
  return `task_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Generate unique sample ID
 */
function generateSampleId(): string {
  return `sample_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Generate ground truth key
 */
function generateGroundTruthKey(): string {
  return `gt_${Math.random().toString(36).substr(2, 16)}`;
}
