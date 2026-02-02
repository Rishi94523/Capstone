/**
 * Model Loader with caching support
 */

import { Config } from '../core/config';

/**
 * Cached model entry
 */
interface CachedModel {
  url: string;
  checksum: string;
  timestamp: number;
  data: ArrayBuffer;
}

/**
 * Model Loader with IndexedDB caching
 */
export class ModelLoader {
  private config: Config;
  private dbName = 'pouw-captcha-models';
  private storeName = 'models';
  private db: IDBDatabase | null = null;
  private cacheMaxAge = 7 * 24 * 60 * 60 * 1000; // 7 days

  constructor(config: Config) {
    this.config = config;
    this.initDB();
  }

  /**
   * Initialize IndexedDB
   */
  private async initDB(): Promise<void> {
    if (typeof indexedDB === 'undefined') {
      this.config.debug('IndexedDB not available, caching disabled');
      return;
    }

    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, 1);

      request.onerror = () => {
        this.config.error('Failed to open IndexedDB:', request.error);
        reject(request.error);
      };

      request.onsuccess = () => {
        this.db = request.result;
        this.config.debug('IndexedDB initialized');
        resolve();
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        if (!db.objectStoreNames.contains(this.storeName)) {
          db.createObjectStore(this.storeName, { keyPath: 'url' });
        }
      };
    });
  }

  /**
   * Load model from cache or network
   */
  async loadModel(
    url: string,
    checksum: string
  ): Promise<ArrayBuffer> {
    // Try cache first
    const cached = await this.getFromCache(url);
    if (cached && cached.checksum === checksum && !this.isExpired(cached)) {
      this.config.debug('Model loaded from cache', { url });
      return cached.data;
    }

    // Fetch from network
    this.config.debug('Fetching model from network', { url });
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`Failed to fetch model: ${response.status}`);
    }

    const data = await response.arrayBuffer();

    // Store in cache
    await this.saveToCache({
      url,
      checksum,
      timestamp: Date.now(),
      data,
    });

    return data;
  }

  /**
   * Get model from cache
   */
  private async getFromCache(url: string): Promise<CachedModel | null> {
    if (!this.db) return null;

    return new Promise((resolve) => {
      const transaction = this.db!.transaction(this.storeName, 'readonly');
      const store = transaction.objectStore(this.storeName);
      const request = store.get(url);

      request.onerror = () => {
        this.config.debug('Cache read error:', request.error);
        resolve(null);
      };

      request.onsuccess = () => {
        resolve(request.result || null);
      };
    });
  }

  /**
   * Save model to cache
   */
  private async saveToCache(model: CachedModel): Promise<void> {
    if (!this.db) return;

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(this.storeName, 'readwrite');
      const store = transaction.objectStore(this.storeName);
      const request = store.put(model);

      request.onerror = () => {
        this.config.debug('Cache write error:', request.error);
        reject(request.error);
      };

      request.onsuccess = () => {
        this.config.debug('Model cached', { url: model.url });
        resolve();
      };
    });
  }

  /**
   * Check if cached model is expired
   */
  private isExpired(cached: CachedModel): boolean {
    return Date.now() - cached.timestamp > this.cacheMaxAge;
  }

  /**
   * Clear model cache
   */
  async clearCache(): Promise<void> {
    if (!this.db) return;

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(this.storeName, 'readwrite');
      const store = transaction.objectStore(this.storeName);
      const request = store.clear();

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        this.config.debug('Model cache cleared');
        resolve();
      };
    });
  }

  /**
   * Get cache statistics
   */
  async getCacheStats(): Promise<{ count: number; totalSize: number }> {
    if (!this.db) return { count: 0, totalSize: 0 };

    return new Promise((resolve) => {
      const transaction = this.db!.transaction(this.storeName, 'readonly');
      const store = transaction.objectStore(this.storeName);
      const request = store.getAll();

      request.onerror = () => resolve({ count: 0, totalSize: 0 });

      request.onsuccess = () => {
        const models = request.result as CachedModel[];
        const totalSize = models.reduce((sum, m) => sum + m.data.byteLength, 0);
        resolve({ count: models.length, totalSize });
      };
    });
  }
}
