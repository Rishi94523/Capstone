/**
 * Cryptographic utilities for Proof-of-Work
 */

import type { ProofOfWork } from '../types';

/**
 * Generate SHA-256 hash
 */
async function sha256(data: string): Promise<string> {
  const encoder = new TextEncoder();
  const buffer = encoder.encode(data);
  const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Generate hash of arbitrary data
 */
async function hashData(data: ArrayBuffer | string): Promise<string> {
  if (typeof data === 'string') {
    return sha256(data);
  }

  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Input for PoW hash generation
 */
export interface PoWInput {
  modelChecksum: string;
  inputData: string;
  outputData: string;
  sessionId: string;
}

/**
 * Generate Proof-of-Work hash
 *
 * Creates a cryptographic proof that the client performed actual computation
 * by hashing together the model checksum, input, output, and a nonce.
 */
export async function generatePoWHash(input: PoWInput): Promise<ProofOfWork> {
  const inputHash = await sha256(input.inputData);
  const outputHash = await sha256(input.outputData);

  // Find a nonce that produces a valid hash (simple PoW)
  let nonce = 0;
  let hash = '';

  // Simple difficulty: hash must start with '0'
  // This is intentionally easy since we don't want to burden users
  const difficulty = 1;
  const prefix = '0'.repeat(difficulty);

  do {
    const payload = [
      input.modelChecksum,
      inputHash,
      outputHash,
      input.sessionId,
      nonce.toString(),
    ].join(':');

    hash = await sha256(payload);
    nonce++;
  } while (!hash.startsWith(prefix) && nonce < 1000000);

  return {
    hash,
    nonce: nonce - 1,
    modelChecksum: input.modelChecksum,
    inputHash,
    outputHash,
  };
}

/**
 * Verify a Proof-of-Work hash
 */
export async function verifyPoWHash(
  pow: ProofOfWork,
  sessionId: string
): Promise<boolean> {
  const payload = [
    pow.modelChecksum,
    pow.inputHash,
    pow.outputHash,
    sessionId,
    pow.nonce.toString(),
  ].join(':');

  const computedHash = await sha256(payload);
  return computedHash === pow.hash && pow.hash.startsWith('0');
}

/**
 * Generate a secure random token
 */
export function generateToken(length: number = 32): string {
  const array = new Uint8Array(length);
  crypto.getRandomValues(array);
  return Array.from(array, (b) => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Generate anonymous fingerprint (privacy-preserving)
 *
 * Creates a hash of non-identifying browser characteristics
 * that changes over time to prevent tracking.
 */
export async function generateAnonymousFingerprint(): Promise<string> {
  // Use only coarse-grained, non-identifying characteristics
  const components = [
    navigator.language,
    new Date().getTimezoneOffset().toString(),
    screen.colorDepth.toString(),
    navigator.hardwareConcurrency?.toString() || '0',
    // Add time-based salt (changes daily) to prevent long-term tracking
    Math.floor(Date.now() / (24 * 60 * 60 * 1000)).toString(),
  ];

  return sha256(components.join('|'));
}

export { sha256, hashData };
