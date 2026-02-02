/**
 * Vanilla JS wrapper for PoUW CAPTCHA
 */

import { PoUWCaptcha, CaptchaConfig, CaptchaResult } from '@pouw/widget';

/**
 * Create a PoUW CAPTCHA instance
 */
export function createCaptcha(config: CaptchaConfig): {
  execute: () => Promise<CaptchaResult>;
  reset: () => void;
  destroy: () => void;
} {
  const instance = new PoUWCaptcha(config);

  return {
    execute: () => instance.execute(),
    reset: () => instance.reset(),
    destroy: () => instance.destroy(),
  };
}

/**
 * Auto-initialize from data attributes
 */
export function autoInit(): void {
  if (typeof document === 'undefined') return;

  document.querySelectorAll('[data-pouw-captcha]').forEach((element) => {
    const siteKey = element.getAttribute('data-site-key');
    const theme = element.getAttribute('data-theme') as
      | 'light'
      | 'dark'
      | 'auto';
    const callbackName = element.getAttribute('data-callback');

    if (!siteKey) {
      console.error('[PoUW] Missing data-site-key attribute');
      return;
    }

    const captcha = createCaptcha({
      siteKey,
      container: element as HTMLElement,
      theme,
      onSuccess: (token) => {
        if (callbackName && typeof window[callbackName as any] === 'function') {
          (window as any)[callbackName](token);
        }
      },
    });

    // Store instance on element
    (element as any).__pouw_captcha = captcha;
  });
}

// Auto-initialize on DOMContentLoaded
if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', autoInit);
  } else {
    autoInit();
  }
}
