/**
 * Type definitions for PoUW CAPTCHA Widget
 */

/**
 * Model shard configuration from server
 */
export interface ModelShard {
  /** Shard index (layer depth) */
  index: number;
  /** Layer name */
  name: string;
  /** Layer type (Conv2D, Dense, etc.) */
  layerType: string;
  /** Serialized weights (if provided) */
  weights?: Record<string, number[]>;
  /** Expected input shape */
  inputShape: number[];
  /** Expected output shape */
  outputShape: number[];
  /** Activation function */
  activation?: string;
  /** Layer configurations for shard execution */
  layers: NeuralLayerConfig[];
}

/**
 * Neural layer configuration for shard execution
 */
export interface NeuralLayerConfig {
  /** Layer name */
  name: string;
  /** Layer type (conv2d, dense, maxpool2d, flatten) */
  type: string;
  /** Layer weights as flat array */
  weights: number[];
  /** Layer biases as flat array */
  biases: number[];
  /** Input shape */
  inputShape: number[];
  /** Output shape */
  outputShape: number[];
  /** Activation function (relu, softmax, sigmoid, tanh, linear) */
  activation: string;
}

/**
 * Shard-based task assignment from server
 */
export interface ShardTask {
  /** Unique task ID */
  taskId: string;
  /** Sample ID for this task */
  sampleId: string;
  /** Model name */
  modelName: string;
  /** Model version */
  modelVersion: string;
  /** Assigned shards (layers) to compute */
  shards: ModelShard[];
  /** Input data as base64 string */
  inputData: string;
  /** Input shape */
  inputShape: number[];
  /** Number of layers client should compute */
  expectedLayers: number;
  /** Difficulty level */
  difficulty: 'easy' | 'medium' | 'hard';
  /** Expected computation time in ms */
  expectedTimeMs: number;
  /** Ground truth key for validation */
  groundTruthKey: string;
  /** Class labels for prediction */
  labels: string[];
  /** Progress callback */
  onProgress?: (progress: number) => void;
}

/**
 * Proof of inference computation (replaces PoW)
 */
export interface InferenceProof {
  /** Task ID */
  taskId: string;
  /** Sample ID */
  sampleId: string;
  /** Number of layers computed */
  layerCount: number;
  /** Hashes of each layer output */
  outputHashes: string[];
  /** Hash of prediction */
  predictionHash: string;
  /** Combined proof hash */
  proofHash: string;
  /** Timestamp when proof was generated */
  timestamp: number;
}

/**
 * Configuration options for the CAPTCHA widget
 */
export interface CaptchaConfig {
  /** Server API endpoint */
  apiUrl: string;
  /** Site API key (public) */
  siteKey: string;
  /** Container element or selector */
  container: HTMLElement | string;
  /** Theme: 'light' | 'dark' | 'auto' */
  theme?: 'light' | 'dark' | 'auto';
  /** Language code (e.g., 'en', 'es', 'fr') */
  language?: string;
  /** Callback when CAPTCHA is solved */
  onSuccess?: (token: string) => void;
  /** Callback when CAPTCHA fails */
  onError?: (error: CaptchaError) => void;
  /** Callback when CAPTCHA expires */
  onExpire?: () => void;
  /** Callback for verification UI */
  onVerificationRequired?: (data: VerificationData) => void;
  /** Enable debug logging */
  debug?: boolean;
  /** Custom model URL override */
  modelUrl?: string;
  /** Timeout in milliseconds */
  timeout?: number;
  /** Invisible mode (no UI) */
  invisible?: boolean;
}

/**
 * Result returned after successful CAPTCHA completion
 */
export interface CaptchaResult {
  /** JWT token to validate on server */
  token: string;
  /** When the token expires */
  expiresAt: Date;
  /** Session ID for tracking */
  sessionId: string;
  /** Whether human verification was performed */
  verificationPerformed: boolean;
}

/**
 * ML task assigned by the server
 */
export interface CaptchaTask {
  /** Unique task ID */
  taskId: string;
  /** URL to load the model from */
  modelUrl: string;
  /** Sample data (base64 or URL) */
  sampleData: string;
  /** Sample URL (alternative to inline data) */
  sampleUrl?: string;
  /** Type of sample */
  sampleType: 'image' | 'text';
  /** Task type */
  taskType: 'inference' | 'gradient' | 'training';
  /** Expected completion time in ms */
  expectedTimeMs: number;
  /** Model metadata */
  modelMeta: ModelMeta;
}

/**
 * Model metadata
 */
export interface ModelMeta {
  /** Model name */
  name: string;
  /** Model version */
  version: string;
  /** Expected input shape */
  inputShape: number[];
  /** Output labels */
  labels: string[];
  /** Checksum for integrity */
  checksum: string;
}

/**
 * Prediction result from ML inference
 */
export interface Prediction {
  /** Predicted label */
  label: string;
  /** Confidence score (0-1) */
  confidence: number;
  /** Top-K predictions */
  topK: Array<{
    label: string;
    confidence: number;
  }>;
  /** Raw output tensor values */
  rawOutput?: number[];
}

/**
 * Data for human verification UI
 */
export interface VerificationData {
  /** Verification ID */
  verificationId: string;
  /** Type of display */
  displayType: 'image' | 'text';
  /** URL or content to display */
  displayContent: string;
  /** Predicted label to verify */
  predictedLabel: string;
  /** Human-readable prompt */
  prompt: string;
  /** Available options */
  options: VerificationOption[];
}

/**
 * Verification option
 */
export interface VerificationOption {
  /** Option ID */
  id: string;
  /** Display label */
  label: string;
  /** Option type */
  type: 'confirm' | 'reject' | 'correct';
}

/**
 * Response from human verification
 */
export interface VerificationResponse {
  /** Response type */
  responseType: 'confirm' | 'reject' | 'correct';
  /** Corrected label if type is 'correct' */
  correctedLabel?: string;
  /** Time taken to respond in ms */
  responseTimeMs: number;
}

/**
 * Proof-of-Work data
 */
export interface ProofOfWork {
  /** SHA-256 hash */
  hash: string;
  /** Nonce used */
  nonce: number;
  /** Model checksum included */
  modelChecksum: string;
  /** Input data hash */
  inputHash: string;
  /** Output hash */
  outputHash: string;
}

/**
 * Timing data for performance tracking
 */
export interface TimingData {
  /** Model load time in ms */
  modelLoadMs: number;
  /** Inference time in ms */
  inferenceMs: number;
  /** Total time in ms */
  totalMs: number;
  /** Timestamp when started */
  startedAt: number;
  /** Timestamp when completed */
  completedAt: number;
}

/**
 * ML Model interface
 */
export interface MLModel {
  /** Model identifier */
  id: string;
  /** Runtime type */
  runtime: RuntimeType;
  /** Run inference on input */
  predict(input: Float32Array | ImageData | string): Promise<Prediction>;
  /** Compute gradient (for training tasks) */
  computeGradient?(
    input: Float32Array | ImageData | string,
    target: number
  ): Promise<Float32Array>;
  /** Dispose model resources */
  dispose(): void;
  /** Model metadata */
  meta: ModelMeta;
}

/**
 * Runtime types supported
 */
export type RuntimeType = 'tfjs' | 'onnx';

/**
 * CAPTCHA error
 */
export interface CaptchaError {
  /** Error code */
  code: CaptchaErrorCode;
  /** Human-readable message */
  message: string;
  /** Additional details */
  details?: Record<string, unknown>;
}

/**
 * Error codes
 */
export type CaptchaErrorCode =
  | 'NETWORK_ERROR'
  | 'MODEL_LOAD_ERROR'
  | 'INFERENCE_ERROR'
  | 'TIMEOUT_ERROR'
  | 'VALIDATION_ERROR'
  | 'SESSION_EXPIRED'
  | 'INVALID_CONFIG'
  | 'WEBGL_NOT_SUPPORTED'
  | 'UNKNOWN_ERROR';

/**
 * Widget state
 */
export type WidgetState =
  | 'idle'
  | 'loading'
  | 'processing'
  | 'verifying'
  | 'success'
  | 'error';

/**
 * API response types
 */
export interface InitResponse {
  sessionId: string;
  challengeToken: string;
  task: CaptchaTask | ShardTask;
  difficulty: 'normal' | 'suspicious' | 'bot_like';
  expiresAt: string;
}

export interface SubmitResponse {
  success: boolean;
  requiresVerification: boolean;
  verification?: VerificationData;
  captchaToken?: string;
  expiresAt?: string;
}

export interface VerifyResponse {
  success: boolean;
  captchaToken: string;
  expiresAt: string;
}

