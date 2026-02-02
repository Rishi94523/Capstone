/**
 * CSS Styles for PoUW CAPTCHA Widget
 *
 * All styles are scoped within Shadow DOM to prevent conflicts
 */

import { prefersReducedMotion, prefersHighContrast } from '../utils/accessibility';

/**
 * Get widget styles as a string
 */
export function getWidgetStyles(): string {
  const reducedMotion = prefersReducedMotion();
  const highContrast = prefersHighContrast();

  return `
    :host {
      --pouw-primary: #6366f1;
      --pouw-primary-hover: #4f46e5;
      --pouw-primary-glow: rgba(99, 102, 241, 0.3);
      --pouw-success: #10b981;
      --pouw-success-glow: rgba(16, 185, 129, 0.3);
      --pouw-error: #ef4444;
      --pouw-error-glow: rgba(239, 68, 68, 0.3);
      --pouw-warning: #f59e0b;
      --pouw-bg: rgba(255, 255, 255, 0.85);
      --pouw-bg-glass: rgba(255, 255, 255, 0.7);
      --pouw-bg-secondary: rgba(243, 244, 246, 0.6);
      --pouw-text: #1f2937;
      --pouw-text-secondary: #6b7280;
      --pouw-border: rgba(229, 231, 235, 0.5);
      --pouw-border-radius: 16px;
      --pouw-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
      --pouw-shadow-lg: 0 20px 60px rgba(0, 0, 0, 0.12);
      --pouw-font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 
                          'Helvetica Neue', Arial, sans-serif;
      --pouw-transition: ${reducedMotion ? 'none' : 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)'};

      display: block;
      font-family: var(--pouw-font-family);
      font-size: 14px;
      line-height: 1.5;
      color: var(--pouw-text);
    }

    :host(.dark) {
      --pouw-bg: rgba(31, 41, 55, 0.9);
      --pouw-bg-glass: rgba(31, 41, 55, 0.7);
      --pouw-bg-secondary: rgba(55, 65, 81, 0.5);
      --pouw-text: #f9fafb;
      --pouw-text-secondary: #9ca3af;
      --pouw-border: rgba(75, 85, 99, 0.5);
    }

    ${highContrast ? `
    :host {
      --pouw-primary: #0000ee;
      --pouw-success: #008000;
      --pouw-error: #ff0000;
      --pouw-border: #000000;
    }
    ` : ''}

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    .pouw-container {
      background: var(--pouw-bg-glass);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border: 1px solid var(--pouw-border);
      border-radius: var(--pouw-border-radius);
      box-shadow: var(--pouw-shadow-lg);
      padding: 24px;
      min-width: 320px;
      max-width: 420px;
      position: relative;
      overflow: hidden;
    }

    .pouw-container::before {
      content: '';
      position: absolute;
      top: -50%;
      left: -50%;
      width: 200%;
      height: 200%;
      background: radial-gradient(circle, var(--pouw-primary-glow) 0%, transparent 70%);
      opacity: 0.5;
      pointer-events: none;
    }

    .pouw-header {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 12px;
    }

    .pouw-logo {
      width: 24px;
      height: 24px;
      flex-shrink: 0;
    }

    .pouw-title {
      font-size: 14px;
      font-weight: 500;
      color: var(--pouw-text);
    }

    .pouw-content {
      min-height: 60px;
      display: flex;
      flex-direction: column;
      justify-content: center;
    }

    /* Loading State */
    .pouw-loading {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .pouw-spinner {
      width: 24px;
      height: 24px;
      border: 3px solid transparent;
      border-top-color: var(--pouw-primary);
      border-right-color: var(--pouw-primary);
      border-radius: 50%;
      animation: ${reducedMotion ? 'none' : 'pouw-spin 0.8s linear infinite'};
      box-shadow: 0 0 10px var(--pouw-primary-glow);
    }

    @keyframes pouw-spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }

    .pouw-message {
      color: var(--pouw-text-secondary);
      font-size: 14px;
      font-weight: 500;
    }

    /* Progress Bar */
    .pouw-progress-container {
      margin-top: 8px;
    }

    .pouw-progress-bar {
      height: 4px;
      background: var(--pouw-bg-secondary);
      border-radius: 2px;
      overflow: hidden;
    }

    .pouw-progress-fill {
      height: 100%;
      background: var(--pouw-primary);
      transition: width var(--pouw-transition);
      width: 0%;
    }

    .pouw-progress-text {
      font-size: 11px;
      color: var(--pouw-text-secondary);
      margin-top: 4px;
      text-align: right;
    }

    /* Success State */
    .pouw-success {
      display: flex;
      align-items: center;
      gap: 12px;
      color: var(--pouw-success);
    }

    .pouw-success-icon {
      width: 24px;
      height: 24px;
      background: var(--pouw-success);
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .pouw-success-icon svg {
      width: 14px;
      height: 14px;
      stroke: white;
      stroke-width: 3;
    }

    .pouw-success-text {
      font-weight: 500;
    }

    /* Error State */
    .pouw-error {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .pouw-error-message {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--pouw-error);
      font-size: 13px;
    }

    .pouw-error-icon {
      width: 16px;
      height: 16px;
      flex-shrink: 0;
    }

    /* Buttons */
    .pouw-button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 8px 16px;
      font-size: 13px;
      font-weight: 500;
      border-radius: 6px;
      border: none;
      cursor: pointer;
      transition: background-color var(--pouw-transition), 
                  transform var(--pouw-transition);
      font-family: inherit;
    }

    .pouw-button:focus {
      outline: 2px solid var(--pouw-primary);
      outline-offset: 2px;
    }

    .pouw-button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    .pouw-button--primary {
      background: var(--pouw-primary);
      color: white;
    }

    .pouw-button--primary:hover:not(:disabled) {
      background: var(--pouw-primary-hover);
    }

    .pouw-button--primary:active:not(:disabled) {
      transform: ${reducedMotion ? 'none' : 'scale(0.98)'};
    }

    .pouw-button--secondary {
      background: var(--pouw-bg-secondary);
      color: var(--pouw-text);
      border: 1px solid var(--pouw-border);
    }

    .pouw-button--secondary:hover:not(:disabled) {
      background: var(--pouw-border);
    }

    /* Verification UI */
    .pouw-verification {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .pouw-verification-prompt {
      font-size: 14px;
      font-weight: 500;
      color: var(--pouw-text);
    }

    .pouw-verification-content {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 12px;
      padding: 12px;
      background: var(--pouw-bg-secondary);
      border-radius: 6px;
    }

    .pouw-verification-image {
      max-width: 100%;
      max-height: 150px;
      border-radius: 4px;
    }

    .pouw-verification-text {
      padding: 12px;
      background: var(--pouw-bg);
      border: 1px solid var(--pouw-border);
      border-radius: 4px;
      font-style: italic;
      max-height: 100px;
      overflow-y: auto;
    }

    .pouw-verification-label {
      font-size: 16px;
      font-weight: 600;
      color: var(--pouw-primary);
    }

    .pouw-verification-actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }

    .pouw-verification-actions .pouw-button {
      flex: 1;
      min-width: 80px;
    }

    /* Correction Select */
    .pouw-correction {
      display: none;
      flex-direction: column;
      gap: 8px;
      margin-top: 8px;
    }

    .pouw-correction.visible {
      display: flex;
    }

    .pouw-correction-label {
      font-size: 13px;
      color: var(--pouw-text-secondary);
    }

    .pouw-correction-select {
      padding: 8px 12px;
      font-size: 14px;
      border: 1px solid var(--pouw-border);
      border-radius: 6px;
      background: var(--pouw-bg);
      color: var(--pouw-text);
      font-family: inherit;
    }

    .pouw-correction-select:focus {
      outline: 2px solid var(--pouw-primary);
      outline-offset: 2px;
    }

    /* Footer */
    .pouw-footer {
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px solid var(--pouw-border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 11px;
      color: var(--pouw-text-secondary);
    }

    .pouw-footer a {
      color: var(--pouw-text-secondary);
      text-decoration: none;
    }

    .pouw-footer a:hover {
      text-decoration: underline;
    }

    .pouw-footer a:focus {
      outline: 2px solid var(--pouw-primary);
      outline-offset: 2px;
    }

    /* Screen reader only */
    .sr-only {
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
    }

    /* Focus visible */
    .pouw-button:focus-visible,
    .pouw-correction-select:focus-visible,
    .pouw-footer a:focus-visible {
      outline: 2px solid var(--pouw-primary);
      outline-offset: 2px;
    }
  `;
}

/**
 * Create and inject styles into Shadow DOM
 */
export function createStyleSheet(): HTMLStyleElement {
  const style = document.createElement('style');
  style.textContent = getWidgetStyles();
  return style;
}
