/**
 * Tests for Shard Inference Engine
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { ShardInferenceEngine, isShardEngineSupported, ShardExecutionResult } from '../src/ml/shard-engine';
import { Config } from '../src/core/config';
import type { ShardTask, ModelShard, NeuralLayerConfig } from '../src/types';

describe('ShardInferenceEngine', () => {
  let engine: ShardInferenceEngine;
  let mockConfig: Config;

  beforeEach(() => {
    mockConfig = new Config({
      siteKey: 'test-key',
      apiUrl: 'http://localhost:8000/api/v1',
      container: document.createElement('div'),
      debug: false,
    });
    engine = new ShardInferenceEngine(mockConfig);
  });

  describe('isShardEngineSupported', () => {
    it('should return true when required APIs are available', () => {
      expect(isShardEngineSupported()).toBe(true);
    });
  });

  describe('executeShards', () => {
    it('should execute a simple dense layer shard', async () => {
      const layerConfig: NeuralLayerConfig = {
        name: 'dense_1',
        type: 'dense',
        weights: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        biases: [0.1, 0.1],
        inputShape: [1, 4],
        outputShape: [1, 2],
        activation: 'relu',
      };

      const shard: ModelShard = {
        index: 0,
        name: 'shard_0',
        layerType: 'dense',
        inputShape: [1, 4],
        outputShape: [1, 2],
        layers: [layerConfig],
      };

      const task: ShardTask = {
        taskId: 'test-task-1',
        sampleId: 'sample-1',
        modelName: 'test-model',
        modelVersion: '1.0',
        shards: [shard],
        inputData: btoa(String.fromCharCode(...new Uint8Array(new Float32Array([1, 2, 3, 4]).buffer))),
        inputShape: [1, 4],
        expectedLayers: 1,
        difficulty: 'easy',
        expectedTimeMs: 100,
        groundTruthKey: 'gt-key',
        labels: ['zero', 'one', 'two', 'three'],
      };

      const result = await engine.executeShards(task);

      expect(result).toHaveProperty('layerOutputs');
      expect(result).toHaveProperty('prediction');
      expect(result).toHaveProperty('proof');
      expect(result).toHaveProperty('timing');
      expect(result.layerOutputs).toHaveLength(1);
      expect(result.prediction).toHaveProperty('label');
      expect(result.proof).toHaveProperty('proofHash');
    });

    it('should handle multiple layers', async () => {
      const layers: NeuralLayerConfig[] = [
        {
          name: 'dense_1',
          type: 'dense',
          weights: new Array(784 * 128).fill(0.01),
          biases: new Array(128).fill(0.1),
          inputShape: [1, 784],
          outputShape: [1, 128],
          activation: 'relu',
        },
        {
          name: 'dense_2',
          type: 'dense',
          weights: new Array(128 * 10).fill(0.01),
          biases: new Array(10).fill(0.1),
          inputShape: [1, 128],
          outputShape: [1, 10],
          activation: 'softmax',
        },
      ];

      const shard: ModelShard = {
        index: 0,
        name: 'full_model',
        layerType: 'dense',
        inputShape: [1, 784],
        outputShape: [1, 10],
        layers,
      };

      const task: ShardTask = {
        taskId: 'test-task-2',
        sampleId: 'sample-2',
        modelName: 'mnist-tiny',
        modelVersion: '1.0',
        shards: [shard],
        inputData: btoa(String.fromCharCode(...new Uint8Array(new Float32Array(784).fill(0.5).buffer))),
        inputShape: [1, 784],
        expectedLayers: 2,
        difficulty: 'medium',
        expectedTimeMs: 500,
        groundTruthKey: 'gt-key-2',
        labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
      };

      const result = await engine.executeShards(task);

      expect(result.layerOutputs).toHaveLength(2);
      expect(result.prediction!.topK).toHaveLength(5);
    });

    it('should call progress callback', async () => {
      const progressCallback = vi.fn();

      const layer: NeuralLayerConfig = {
        name: 'dense_1',
        type: 'dense',
        weights: [0.1, 0.2, 0.3, 0.4],
        biases: [0.1, 0.1],
        inputShape: [1, 2],
        outputShape: [1, 2],
        activation: 'relu',
      };

      const shard: ModelShard = {
        index: 0,
        name: 'shard_0',
        layerType: 'dense',
        inputShape: [1, 2],
        outputShape: [1, 2],
        layers: [layer],
      };

      const task: ShardTask = {
        taskId: 'test-task-3',
        sampleId: 'sample-3',
        modelName: 'test',
        modelVersion: '1.0',
        shards: [shard],
        inputData: btoa(String.fromCharCode(...new Uint8Array(new Float32Array([1, 2]).buffer))),
        inputShape: [1, 2],
        expectedLayers: 1,
        difficulty: 'easy',
        expectedTimeMs: 50,
        groundTruthKey: 'gt-key',
        labels: ['a', 'b'],
        onProgress: progressCallback,
      };

      await engine.executeShards(task);

      expect(progressCallback).toHaveBeenCalledWith(1);
    });
  });

  describe('generatePrediction', () => {
    it('should generate correct top-k predictions', async () => {
      // Create a task with known output
      const output = new Float32Array([0.1, 0.5, 0.3, 0.05, 0.05]);
      const labels = ['A', 'B', 'C', 'D', 'E'];

      // Use reflection to test private method
      const engineAny = engine as any;
      const prediction = engineAny.generatePrediction(output, labels);

      expect(prediction.label).toBe('B');
      // Confidence is rounded to 3 decimal places
      expect(prediction.topK[0]).toEqual({ label: 'B', confidence: 0.5 });
    });
  });

  describe('proof generation', () => {
    it('should generate deterministic proof hashes', async () => {
      const layer: NeuralLayerConfig = {
        name: 'dense_1',
        type: 'dense',
        weights: [0.1, 0.2],
        biases: [0.1],
        inputShape: [1, 2],
        outputShape: [1, 1],
        activation: 'linear',
      };

      const shard: ModelShard = {
        index: 0,
        name: 'shard_0',
        layerType: 'dense',
        inputShape: [1, 2],
        outputShape: [1, 1],
        layers: [layer],
      };

      const task: ShardTask = {
        taskId: 'same-task-id',
        sampleId: 'same-sample',
        modelName: 'test',
        modelVersion: '1.0',
        shards: [shard],
        inputData: btoa(String.fromCharCode(...new Uint8Array(new Float32Array([1, 2]).buffer))),
        inputShape: [1, 2],
        expectedLayers: 1,
        difficulty: 'easy',
        expectedTimeMs: 50,
        groundTruthKey: 'gt',
        labels: ['x'],
      };

      const result1 = await engine.executeShards(task);
      
      // Create new engine instance for second run
      const engine2 = new ShardInferenceEngine(mockConfig);
      const result2 = await engine2.executeShards(task);

      // Same inputs should produce same proof hash
      expect(result1.proof.proofHash).toBe(result2.proof.proofHash);
    });
  });
});
