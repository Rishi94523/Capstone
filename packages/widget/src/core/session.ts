/**
 * Session management for PoUW CAPTCHA
 */

import type {
  CaptchaTask,
  InitResponse,
  Prediction,
  ProofOfWork,
  TimingData,
  VerificationData,
  WidgetState,
  ShardTask,
  InferenceProof,
} from '../types';

/**
 * Manages the state and lifecycle of a CAPTCHA session
 */
export class CaptchaSession {
  /** Unique session ID */
  public readonly sessionId: string;
  /** Challenge token for validation */
  public readonly challengeToken: string;
  /** Assigned task (can be CaptchaTask or ShardTask) */
  public readonly task: CaptchaTask | ShardTask;
  /** Difficulty level */
  public readonly difficulty: 'normal' | 'suspicious' | 'bot_like';
  /** Session expiry time */
  public readonly expiresAt: Date;
  /** Creation timestamp */
  public readonly createdAt: Date;

  /** Current state */
  private _state: WidgetState = 'idle';
  /** Prediction result */
  private _prediction: Prediction | null = null;
  /** Timing data */
  private _timing: TimingData | null = null;
  /** Proof of work (legacy) */
  private _proofOfWork: ProofOfWork | null = null;
  /** Inference proof (new shard-based) */
  private _inferenceProof: InferenceProof | null = null;
  /** Verification data */
  private _verificationData: VerificationData | null = null;
  /** Final token */
  private _captchaToken: string | null = null;
  /** Error if any */
  private _error: Error | null = null;

  /** State change listeners */
  private stateListeners: Set<(state: WidgetState) => void> = new Set();

  constructor(response: InitResponse) {
    this.sessionId = response.sessionId;
    this.challengeToken = response.challengeToken;
    this.task = response.task;
    this.difficulty = response.difficulty;
    this.expiresAt = new Date(response.expiresAt);
    this.createdAt = new Date();
  }

  /**
   * Get current state
   */
  get state(): WidgetState {
    return this._state;
  }

  /**
   * Set state and notify listeners
   */
  set state(newState: WidgetState) {
    if (this._state !== newState) {
      this._state = newState;
      this.notifyStateChange();
    }
  }

  /**
   * Get prediction
   */
  get prediction(): Prediction | null {
    return this._prediction;
  }

  /**
   * Set prediction
   */
  set prediction(value: Prediction | null) {
    this._prediction = value;
  }

  /**
   * Get timing data
   */
  get timing(): TimingData | null {
    return this._timing;
  }

  /**
   * Set timing data
   */
  set timing(value: TimingData | null) {
    this._timing = value;
  }

  /**
   * Get proof of work (legacy)
   */
  get proofOfWork(): ProofOfWork | null {
    return this._proofOfWork;
  }

  /**
   * Set proof of work (legacy)
   */
  set proofOfWork(value: ProofOfWork | null) {
    this._proofOfWork = value;
  }

  /**
   * Get inference proof (shard-based)
   */
  get inferenceProof(): InferenceProof | null {
    return this._inferenceProof;
  }

  /**
   * Set inference proof (shard-based)
   */
  set inferenceProof(value: InferenceProof | null) {
    this._inferenceProof = value;
  }

  /**
   * Get verification data
   */
  get verificationData(): VerificationData | null {
    return this._verificationData;
  }

  /**
   * Set verification data
   */
  set verificationData(value: VerificationData | null) {
    this._verificationData = value;
  }

  /**
   * Get CAPTCHA token
   */
  get captchaToken(): string | null {
    return this._captchaToken;
  }

  /**
   * Set CAPTCHA token
   */
  set captchaToken(value: string | null) {
    this._captchaToken = value;
  }

  /**
   * Get error
   */
  get error(): Error | null {
    return this._error;
  }

  /**
   * Set error
   */
  set error(value: Error | null) {
    this._error = value;
    if (value) {
      this.state = 'error';
    }
  }

  /**
   * Check if session is expired
   */
  isExpired(): boolean {
    return new Date() >= this.expiresAt;
  }

  /**
   * Check if session is completed
   */
  isCompleted(): boolean {
    return this._state === 'success' && this._captchaToken !== null;
  }

  /**
   * Check if verification is required
   */
  requiresVerification(): boolean {
    return this._state === 'verifying' && this._verificationData !== null;
  }

  /**
   * Get time remaining in milliseconds
   */
  getTimeRemaining(): number {
    return Math.max(0, this.expiresAt.getTime() - Date.now());
  }

  /**
   * Subscribe to state changes
   */
  onStateChange(callback: (state: WidgetState) => void): () => void {
    this.stateListeners.add(callback);
    return () => {
      this.stateListeners.delete(callback);
    };
  }

  /**
   * Notify all state listeners
   */
  private notifyStateChange(): void {
    for (const listener of this.stateListeners) {
      try {
        listener(this._state);
      } catch (error) {
        console.error('[PoUW Session] State listener error:', error);
      }
    }
  }

  /**
   * Convert session to result object
   */
  toResult(): SessionResult {
    return {
      sessionId: this.sessionId,
      state: this._state,
      token: this._captchaToken,
      expiresAt: this.expiresAt,
      prediction: this._prediction,
      timing: this._timing,
      error: this._error?.message,
    };
  }

  /**
   * Check if session has a shard task
   */
  isShardTask(): boolean {
    return 'shards' in this.task;
  }

  /**
   * Get shard task (if applicable)
   */
  getShardTask(): ShardTask | null {
    return this.isShardTask() ? (this.task as ShardTask) : null;
  }

  /**
   * Clean up session
   */
  dispose(): void {
    this.stateListeners.clear();
  }
}

/**
 * Session result summary
 */
export interface SessionResult {
  sessionId: string;
  state: WidgetState;
  token: string | null;
  expiresAt: Date;
  prediction: Prediction | null;
  timing: TimingData | null;
  error?: string;
}
