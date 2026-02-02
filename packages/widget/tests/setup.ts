/**
 * Test setup for widget tests
 */

import { vi } from 'vitest';

// Mock browser APIs
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation((query) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

// Mock crypto.subtle
Object.defineProperty(global, 'crypto', {
  value: {
    subtle: {
      digest: vi.fn().mockImplementation(async (algorithm, data) => {
        // Simple mock hash
        const mockHash = new Uint8Array(32);
        for (let i = 0; i < 32; i++) {
          mockHash[i] = i;
        }
        return mockHash.buffer;
      }),
    },
    getRandomValues: vi.fn().mockImplementation((array) => {
      for (let i = 0; i < array.length; i++) {
        array[i] = Math.floor(Math.random() * 256);
      }
      return array;
    }),
  },
});

// Mock IndexedDB
const mockIndexedDB = {
  open: vi.fn().mockReturnValue({
    onerror: null,
    onsuccess: null,
    onupgradeneeded: null,
    result: {
      objectStoreNames: { contains: () => false },
      createObjectStore: vi.fn(),
      transaction: vi.fn().mockReturnValue({
        objectStore: vi.fn().mockReturnValue({
          get: vi.fn().mockReturnValue({ onsuccess: null, onerror: null }),
          put: vi.fn().mockReturnValue({ onsuccess: null, onerror: null }),
          clear: vi.fn().mockReturnValue({ onsuccess: null, onerror: null }),
          getAll: vi.fn().mockReturnValue({ onsuccess: null, onerror: null }),
        }),
      }),
    },
  }),
};

Object.defineProperty(window, 'indexedDB', {
  value: mockIndexedDB,
  writable: true,
});

// Mock performance
if (!window.performance) {
  Object.defineProperty(window, 'performance', {
    value: {
      now: vi.fn(() => Date.now()),
    },
  });
}

// Mock ResizeObserver
class MockResizeObserver {
  observe = vi.fn();
  unobserve = vi.fn();
  disconnect = vi.fn();
}

Object.defineProperty(window, 'ResizeObserver', {
  value: MockResizeObserver,
  writable: true,
});

// Mock IntersectionObserver
class MockIntersectionObserver {
  observe = vi.fn();
  unobserve = vi.fn();
  disconnect = vi.fn();
}

Object.defineProperty(window, 'IntersectionObserver', {
  value: MockIntersectionObserver,
  writable: true,
});

// Console spy
vi.spyOn(console, 'log').mockImplementation(() => {});
vi.spyOn(console, 'warn').mockImplementation(() => {});
vi.spyOn(console, 'error').mockImplementation(() => {});
