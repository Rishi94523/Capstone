/**
 * Tests for cryptographic utilities
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { generatePoWHash, verifyPoWHash, generateToken } from '../src/utils/crypto';

describe('Crypto Utilities', () => {
  describe('generatePoWHash', () => {
    it('should generate a valid proof of work', async () => {
      const input = {
        modelChecksum: 'abc123',
        inputData: 'test input data',
        outputData: '{"label": "cat", "confidence": 0.95}',
        sessionId: 'session-123',
      };

      const pow = await generatePoWHash(input);

      expect(pow).toHaveProperty('hash');
      expect(pow).toHaveProperty('nonce');
      expect(pow).toHaveProperty('modelChecksum', input.modelChecksum);
      expect(pow).toHaveProperty('inputHash');
      expect(pow).toHaveProperty('outputHash');
      expect(typeof pow.hash).toBe('string');
      expect(typeof pow.nonce).toBe('number');
      expect(pow.nonce).toBeGreaterThanOrEqual(0);
    });

    it('should generate hash starting with required prefix', async () => {
      const input = {
        modelChecksum: 'test',
        inputData: 'data',
        outputData: 'output',
        sessionId: 'session',
      };

      const pow = await generatePoWHash(input);

      // Hash should start with '0' (difficulty = 1)
      expect(pow.hash.startsWith('0')).toBe(true);
    });
  });

  describe('generateToken', () => {
    it('should generate a token of the specified length', () => {
      const token = generateToken(32);
      expect(token.length).toBe(64); // 32 bytes = 64 hex chars
    });

    it('should generate unique tokens', () => {
      const token1 = generateToken();
      const token2 = generateToken();
      expect(token1).not.toBe(token2);
    });

    it('should use default length when not specified', () => {
      const token = generateToken();
      expect(token.length).toBe(64); // Default 32 bytes = 64 hex chars
    });
  });
});

describe('Performance Timer', () => {
  const { PerformanceTimer } = require('../src/utils/timing');

  it('should track timing correctly', () => {
    const timer = new PerformanceTimer();
    
    timer.start('test');
    // Simulate some work
    const start = Date.now();
    while (Date.now() - start < 10) {}
    timer.end('test');

    expect(timer.get('test')).toBeGreaterThanOrEqual(0);
  });

  it('should handle multiple timers', () => {
    const timer = new PerformanceTimer();
    
    timer.start('a');
    timer.start('b');
    timer.end('a');
    timer.end('b');

    const results = timer.getAll();
    expect(results).toHaveProperty('a');
    expect(results).toHaveProperty('b');
  });

  it('should reset timers', () => {
    const timer = new PerformanceTimer();
    
    timer.start('test');
    timer.end('test');
    timer.reset();

    expect(timer.getAll()).toEqual({});
  });
});
