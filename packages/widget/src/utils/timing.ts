/**
 * Performance timing utilities
 */

/**
 * Performance timer for tracking execution times
 */
export class PerformanceTimer {
  private timers: Map<string, { start: number; end?: number }> = new Map();

  /**
   * Start a timer
   */
  start(name: string): void {
    this.timers.set(name, { start: performance.now() });
  }

  /**
   * End a timer
   */
  end(name: string): number {
    const timer = this.timers.get(name);
    if (!timer) {
      throw new Error(`Timer '${name}' not started`);
    }

    timer.end = performance.now();
    return timer.end - timer.start;
  }

  /**
   * Get elapsed time for a timer
   */
  get(name: string): number {
    const timer = this.timers.get(name);
    if (!timer) {
      return 0;
    }

    if (timer.end) {
      return Math.round(timer.end - timer.start);
    }

    return Math.round(performance.now() - timer.start);
  }

  /**
   * Get start time for a timer
   */
  getStartTime(name: string): number {
    const timer = this.timers.get(name);
    if (!timer) {
      return 0;
    }

    return Math.round(timer.start);
  }

  /**
   * Get all timer results
   */
  getAll(): Record<string, number> {
    const results: Record<string, number> = {};
    for (const [name] of this.timers) {
      results[name] = this.get(name);
    }
    return results;
  }

  /**
   * Reset all timers
   */
  reset(): void {
    this.timers.clear();
  }

  /**
   * Create a formatted summary
   */
  summary(): string {
    const results = this.getAll();
    return Object.entries(results)
      .map(([name, time]) => `${name}: ${time}ms`)
      .join(', ');
  }
}

/**
 * Measure async function execution time
 */
export async function measureAsync<T>(
  fn: () => Promise<T>
): Promise<{ result: T; timeMs: number }> {
  const start = performance.now();
  const result = await fn();
  const timeMs = Math.round(performance.now() - start);
  return { result, timeMs };
}

/**
 * Create a delay promise
 */
export function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Timeout wrapper for promises
 */
export async function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  errorMessage = 'Operation timed out'
): Promise<T> {
  let timeoutId: number;

  const timeoutPromise = new Promise<never>((_, reject) => {
    timeoutId = window.setTimeout(() => {
      reject(new Error(errorMessage));
    }, timeoutMs);
  });

  try {
    const result = await Promise.race([promise, timeoutPromise]);
    clearTimeout(timeoutId!);
    return result;
  } catch (error) {
    clearTimeout(timeoutId!);
    throw error;
  }
}

/**
 * Debounce function
 */
export function debounce<T extends (...args: unknown[]) => unknown>(
  fn: T,
  delayMs: number
): (...args: Parameters<T>) => void {
  let timeoutId: number | undefined;

  return (...args: Parameters<T>) => {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }

    timeoutId = window.setTimeout(() => {
      fn(...args);
    }, delayMs);
  };
}

/**
 * Throttle function
 */
export function throttle<T extends (...args: unknown[]) => unknown>(
  fn: T,
  limitMs: number
): (...args: Parameters<T>) => void {
  let lastRun = 0;
  let timeoutId: number | undefined;

  return (...args: Parameters<T>) => {
    const now = Date.now();

    if (now - lastRun >= limitMs) {
      fn(...args);
      lastRun = now;
    } else {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }

      timeoutId = window.setTimeout(() => {
        fn(...args);
        lastRun = Date.now();
      }, limitMs - (now - lastRun));
    }
  };
}
