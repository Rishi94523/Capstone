/**
 * API Client for communicating with PoUW CAPTCHA server
 */

import type {
  CaptchaTask,
  InitResponse,
  Prediction,
  ProofOfWork,
  SubmitResponse,
  TimingData,
  VerificationResponse,
  VerifyResponse,
} from '../types';
import { Config } from './config';

/**
 * API Client for PoUW CAPTCHA server communication
 */
export class ApiClient {
  private config: Config;
  private baseUrl: string;

  constructor(config: Config) {
    this.config = config;
    this.baseUrl = config.get('apiUrl') || 'https://api.pouw.dev/v1';
  }

  /**
   * Initialize a new CAPTCHA session
   */
  async initSession(metadata: ClientMetadata): Promise<InitResponse> {
    const response = await this.request<InitResponse>('/captcha/init', {
      method: 'POST',
      body: JSON.stringify({
        site_key: this.config.get('siteKey'),
        client_metadata: {
          user_agent: metadata.userAgent,
          language: metadata.language,
          timezone: metadata.timezone,
          screen_width: metadata.screenWidth,
          screen_height: metadata.screenHeight,
        },
      }),
    });

    return response;
  }

  /**
   * Submit prediction result
   */
  async submitPrediction(
    sessionId: string,
    taskId: string,
    prediction: Prediction,
    proofOfWork: ProofOfWork,
    timing: TimingData
  ): Promise<SubmitResponse> {
    const response = await this.request<SubmitResponse>('/captcha/submit', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        task_id: taskId,
        prediction: {
          label: prediction.label,
          confidence: prediction.confidence,
          top_k: prediction.topK,
        },
        proof_of_work: {
          hash: proofOfWork.hash,
          nonce: proofOfWork.nonce,
          model_checksum: proofOfWork.modelChecksum,
          input_hash: proofOfWork.inputHash,
          output_hash: proofOfWork.outputHash,
        },
        timing: {
          model_load_ms: timing.modelLoadMs,
          inference_ms: timing.inferenceMs,
          total_ms: timing.totalMs,
          started_at: timing.startedAt,
          completed_at: timing.completedAt,
        },
      }),
    });

    return response;
  }

  /**
   * Submit human verification response
   */
  async submitVerification(
    sessionId: string,
    verificationId: string,
    response: VerificationResponse
  ): Promise<VerifyResponse> {
    const result = await this.request<VerifyResponse>('/captcha/verify', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        verification_id: verificationId,
        response: response.responseType,
        corrected_label: response.correctedLabel,
        response_time_ms: response.responseTimeMs,
      }),
    });

    return result;
  }

  /**
   * Fetch sample data
   */
  async fetchSample(url: string): Promise<ArrayBuffer> {
    const response = await fetch(url, {
      method: 'GET',
      mode: 'cors',
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch sample: ${response.status}`);
    }

    return response.arrayBuffer();
  }

  /**
   * Fetch model
   */
  async fetchModel(url: string): Promise<Response> {
    const response = await fetch(url, {
      method: 'GET',
      mode: 'cors',
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch model: ${response.status}`);
    }

    return response;
  }

  /**
   * Make an authenticated request
   */
  private async request<T>(endpoint: string, options: RequestInit): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const timeout = this.config.get('timeout') || 30000;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
      this.config.debug(`API Request: ${endpoint}`, options.body);

      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          'X-POUW-Site-Key': this.config.get('siteKey') || '',
          ...options.headers,
        },
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new ApiError(
          response.status,
          errorData.message || `HTTP ${response.status}`,
          errorData
        );
      }

      const data = await response.json();
      this.config.debug(`API Response: ${endpoint}`, data);

      return data as T;
    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof ApiError) {
        throw error;
      }

      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw new ApiError(0, 'Request timeout', { timeout });
        }
        throw new ApiError(0, error.message, { originalError: error.message });
      }

      throw new ApiError(0, 'Unknown error', {});
    }
  }
}

/**
 * Client metadata for session initialization
 */
export interface ClientMetadata {
  userAgent: string;
  language: string;
  timezone: string;
  screenWidth: number;
  screenHeight: number;
}

/**
 * API Error class
 */
export class ApiError extends Error {
  public readonly status: number;
  public readonly details: Record<string, unknown>;

  constructor(status: number, message: string, details: Record<string, unknown>) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.details = details;
  }
}

/**
 * Get client metadata
 */
export function getClientMetadata(): ClientMetadata {
  return {
    userAgent: navigator.userAgent,
    language: navigator.language,
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    screenWidth: window.screen.width,
    screenHeight: window.screen.height,
  };
}
