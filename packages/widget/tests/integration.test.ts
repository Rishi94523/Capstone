/**
 * Integration Tests for ML Inference CAPTCHA Flow
 * Tests the complete flow: init -> shard task -> inference -> proof -> submit
 */
import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { PoUWCaptcha } from '../src/core/captcha';
import type { ShardTask, InitResponse, InferenceProof, CaptchaConfig } from '../src/types';
import { createTestMNISTModel, createTestMNISTInput, inputToBase64, MNIST_LABELS } from './mnist-model';

// Mock fetch
global.fetch = vi.fn();

describe('ML Inference CAPTCHA Integration', () => {
  let container: HTMLDivElement;
  let config: CaptchaConfig;
  let captcha: PoUWCaptcha;

  beforeEach(() => {
    container = document.createElement('div');
    document.body.appendChild(container);

    // Create config with all required fields
    config = {
      siteKey: 'pk_test_integration',
      container,
      apiUrl: 'http://localhost:8000/api/v1',
    };

    // @ts-ignore - Config constructor expects CaptchaConfig
    captcha = new PoUWCaptcha(config);
  });

  afterEach(() => {
    captcha.dispose();
    container.remove();
    vi.clearAllMocks();
  });

  describe('Shard Task Flow', () => {
    it('should complete full shard task verification flow', async () => {
      const { shards } = createTestMNISTModel();
      const testInput = createTestMNISTInput(5);

      const shardTask: ShardTask = {
        taskId: 'task-integration-1',
        sampleId: 'sample-123',
        modelName: 'mnist-tiny',
        modelVersion: '1.0',
        shards,
        inputData: inputToBase64(testInput),
        inputShape: [1, 784],
        expectedLayers: 2,
        difficulty: 'easy',
        expectedTimeMs: 500,
        groundTruthKey: 'gt-key-abc',
        labels: MNIST_LABELS,
      };

      // Mock init response with shard task
      const initResponse: InitResponse = {
        sessionId: 'session-test-123',
        challengeToken: 'challenge-abc',
        task: shardTask,
        difficulty: 'normal',
        expiresAt: new Date(Date.now() + 60000).toISOString(),
      };

      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(initResponse),
      });

      // Mock proof submission response
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          success: true,
          captchaToken: 'captcha-token-xyz',
          expiresAt: new Date(Date.now() + 3600000).toISOString(),
        }),
      });

      // Execute CAPTCHA
      const resultPromise = captcha.execute();

      // Wait for completion
      const result = await resultPromise;

      expect(result.token).toBe('captcha-token-xyz');
      expect(result.verificationPerformed).toBe(true);
    });

    it('should handle inference with progress callbacks', async () => {
      const progressUpdates: number[] = [];
      
      const { shards } = createTestMNISTModel();
      const testInput = createTestMNISTInput(3);

      const shardTask: ShardTask = {
        taskId: 'task-progress-test',
        sampleId: 'sample-456',
        modelName: 'mnist-tiny',
        modelVersion: '1.0',
        shards,
        inputData: inputToBase64(testInput),
        inputShape: [1, 784],
        expectedLayers: 2,
        difficulty: 'medium',
        expectedTimeMs: 800,
        groundTruthKey: 'gt-key-def',
        labels: MNIST_LABELS,
        onProgress: (progress: number) => {
          progressUpdates.push(progress);
        },
      };

      const initResponse: InitResponse = {
        sessionId: 'session-test-123',
        challengeToken: 'challenge-abc',
        task: shardTask,
        difficulty: 'normal',
        expiresAt: new Date(Date.now() + 60000).toISOString(),
      };

      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(initResponse),
      });

      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          success: true,
          captchaToken: 'token-progress',
          expiresAt: new Date(Date.now() + 3600000).toISOString(),
        }),
      });

      await captcha.execute();

      // Verify progress was reported
      expect(progressUpdates.length).toBeGreaterThan(0);
      expect(progressUpdates[progressUpdates.length - 1]).toBe(1);
    });
  });

  describe('Proof Verification', () => {
    it('should generate valid inference proof', async () => {
      const { shards } = createTestMNISTModel();
      const testInput = createTestMNISTInput(7);

      const shardTask: ShardTask = {
        taskId: 'task-proof-test',
        sampleId: 'sample-proof',
        modelName: 'mnist-tiny',
        modelVersion: '1.0',
        shards,
        inputData: inputToBase64(testInput),
        inputShape: [1, 784],
        expectedLayers: 2,
        difficulty: 'hard',
        expectedTimeMs: 1000,
        groundTruthKey: 'gt-proof-key',
        labels: MNIST_LABELS,
      };

      const initResponse: InitResponse = {
        sessionId: 'session-proof-123',
        challengeToken: 'challenge-def',
        task: shardTask,
        difficulty: 'normal',
        expiresAt: new Date(Date.now() + 60000).toISOString(),
      };

      let capturedProof: InferenceProof | null = null;

      (global.fetch as ReturnType<typeof vi.fn>).mockImplementation((url: string, options?: RequestInit) => {
        if (url.includes('/init')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(initResponse),
          });
        }
        
        if (url.includes('/submit-proof')) {
          // Capture the proof being submitted
          if (options?.body) {
            const body = JSON.parse(options.body as string);
            capturedProof = body.proof;
          }
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
              success: true,
              captchaToken: 'token-from-proof',
              expiresAt: new Date(Date.now() + 3600000).toISOString(),
            }),
          });
        }

        return Promise.resolve({ ok: false });
      });

      await captcha.execute();

      expect(capturedProof).not.toBeNull();
      expect(capturedProof!.taskId).toBe('task-proof-test');
      expect(capturedProof!.sampleId).toBe('sample-proof');
      expect(capturedProof!.layerCount).toBe(2);
      expect(capturedProof!.outputHashes).toHaveLength(2);
      expect(capturedProof!.proofHash).toBeTruthy();
      expect(capturedProof!.timestamp).toBeGreaterThan(0);
    });
  });

  describe('Error Handling', () => {
    it('should handle network errors during init', async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error('Network error')
      );

      await expect(captcha.execute()).rejects.toThrow();
    });

    it('should handle server errors during proof submission', async () => {
      const { shards } = createTestMNISTModel();
      const testInput = createTestMNISTInput(2);

      const shardTask: ShardTask = {
        taskId: 'task-error-test',
        sampleId: 'sample-error',
        modelName: 'mnist-tiny',
        modelVersion: '1.0',
        shards,
        inputData: inputToBase64(testInput),
        inputShape: [1, 784],
        expectedLayers: 2,
        difficulty: 'easy',
        expectedTimeMs: 500,
        groundTruthKey: 'gt-error',
        labels: MNIST_LABELS,
      };

      const initResponse: InitResponse = {
        sessionId: 'session-error-123',
        challengeToken: 'challenge-ghi',
        task: shardTask,
        difficulty: 'normal',
        expiresAt: new Date(Date.now() + 60000).toISOString(),
      };

      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(initResponse),
      });

      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: () => Promise.resolve({ error: 'Server error' }),
      });

      await expect(captcha.execute()).rejects.toThrow();
    });
  });

  describe('Performance', () => {
    it('should complete inference within expected time', async () => {
      const { shards } = createTestMNISTModel();
      const testInput = createTestMNISTInput(9);

      const shardTask: ShardTask = {
        taskId: 'task-perf-test',
        sampleId: 'sample-perf',
        modelName: 'mnist-tiny',
        modelVersion: '1.0',
        shards,
        inputData: inputToBase64(testInput),
        inputShape: [1, 784],
        expectedLayers: 2,
        difficulty: 'easy',
        expectedTimeMs: 500,
        groundTruthKey: 'gt-perf',
        labels: MNIST_LABELS,
      };

      const initResponse: InitResponse = {
        sessionId: 'session-perf-123',
        challengeToken: 'challenge-jkl',
        task: shardTask,
        difficulty: 'normal',
        expiresAt: new Date(Date.now() + 60000).toISOString(),
      };

      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(initResponse),
      });

      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          success: true,
          captchaToken: 'token-perf',
          expiresAt: new Date(Date.now() + 3600000).toISOString(),
        }),
      });

      const startTime = performance.now();
      await captcha.execute();
      const endTime = performance.now();

      const duration = endTime - startTime;
      expect(duration).toBeLessThan(5000); // Should complete within 5 seconds
    });
  });
});
