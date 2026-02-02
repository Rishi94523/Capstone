/**
 * Main CAPTCHA Widget UI Component
 */

import type { VerificationData, VerificationResponse, WidgetState } from '../types';
import { Config } from '../core/config';
import { createStyleSheet } from './styles';
import { ProgressIndicator } from './progress';
import { VerificationUI } from './verification';
import { announceToScreenReader } from '../utils/accessibility';

/**
 * Widget event handlers
 */
export interface WidgetHandlers {
  onRetry: () => void;
  onVerificationSubmit: (response: VerificationResponse) => void;
}

/**
 * Main CAPTCHA Widget Component
 *
 * Uses Shadow DOM for style isolation
 */
export class CaptchaWidget {
  private config: Config;
  private handlers: WidgetHandlers;
  private container: HTMLElement;
  private shadowRoot: ShadowRoot;
  private widgetElement: HTMLElement;
  private contentElement: HTMLElement;
  private progressIndicator: ProgressIndicator | null = null;
  private verificationUI: VerificationUI | null = null;
  private currentState: WidgetState = 'idle';

  constructor(config: Config, handlers: WidgetHandlers) {
    this.config = config;
    this.handlers = handlers;
    this.container = config.getContainer();

    // Create Shadow DOM
    this.shadowRoot = this.container.attachShadow({ mode: 'closed' });

    // Add styles
    this.shadowRoot.appendChild(createStyleSheet());

    // Create widget structure
    this.widgetElement = this.createWidgetElement();
    this.contentElement = this.widgetElement.querySelector('.pouw-content')!;
    this.shadowRoot.appendChild(this.widgetElement);

    // Apply theme
    const theme = config.get('theme');
    if (theme === 'dark') {
      (this.shadowRoot.host as HTMLElement).classList.add('dark');
    }

    // Initial state
    this.renderIdleState();
  }

  /**
   * Create the main widget element
   */
  private createWidgetElement(): HTMLElement {
    const widget = document.createElement('div');
    widget.className = 'pouw-container';
    widget.setAttribute('role', 'region');
    widget.setAttribute('aria-label', 'Security verification');

    widget.innerHTML = `
      <div class="pouw-header">
        <svg class="pouw-logo" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
          <path d="M9 12l2 2 4-4"/>
        </svg>
        <span class="pouw-title">Security Check</span>
      </div>
      <div class="pouw-content" aria-live="polite">
        <!-- Dynamic content -->
      </div>
      <div class="pouw-footer">
        <span>Protected by PoUW CAPTCHA</span>
        <a href="https://pouw.dev/privacy" target="_blank" rel="noopener">Privacy</a>
      </div>
    `;

    return widget;
  }

  /**
   * Set widget state
   */
  setState(state: WidgetState): void {
    if (this.currentState === state) return;

    this.currentState = state;
    this.config.debug('Widget state changed:', state);

    switch (state) {
      case 'idle':
        this.renderIdleState();
        break;
      case 'loading':
        this.renderLoadingState();
        break;
      case 'processing':
        this.renderProcessingState();
        break;
      case 'verifying':
        // Handled by showVerification
        break;
      case 'success':
        // Handled by showSuccess
        break;
      case 'error':
        // Handled by showError
        break;
    }
  }

  /**
   * Set status message
   */
  setMessage(message: string): void {
    const msgElement = this.contentElement.querySelector('.pouw-message');
    if (msgElement) {
      msgElement.textContent = message;
    }
  }

  /**
   * Set progress value
   */
  setProgress(value: number): void {
    if (this.progressIndicator) {
      this.progressIndicator.setProgress(value);
    }
  }

  /**
   * Render idle state
   */
  private renderIdleState(): void {
    this.contentElement.innerHTML = `
      <div class="pouw-loading">
        <div class="pouw-message">Click to verify you're human</div>
      </div>
    `;

    // Make clickable
    this.contentElement.style.cursor = 'pointer';
    this.contentElement.setAttribute('tabindex', '0');
    this.contentElement.setAttribute('role', 'button');
  }

  /**
   * Render loading state
   */
  private renderLoadingState(): void {
    this.contentElement.innerHTML = `
      <div class="pouw-loading">
        <div class="pouw-spinner" aria-hidden="true"></div>
        <div class="pouw-message">Initializing...</div>
      </div>
    `;

    this.contentElement.style.cursor = 'default';
    this.contentElement.removeAttribute('tabindex');
    this.contentElement.removeAttribute('role');

    announceToScreenReader('Security check initializing');
  }

  /**
   * Render processing state
   */
  private renderProcessingState(): void {
    this.contentElement.innerHTML = `
      <div class="pouw-loading">
        <div class="pouw-spinner" aria-hidden="true"></div>
        <div class="pouw-message">Running security check...</div>
      </div>
      <div class="pouw-progress-wrapper"></div>
    `;

    const progressWrapper = this.contentElement.querySelector('.pouw-progress-wrapper');
    if (progressWrapper) {
      this.progressIndicator = new ProgressIndicator(progressWrapper as HTMLElement);
    }

    announceToScreenReader('Running security check using useful computation');
  }

  /**
   * Show verification UI
   */
  showVerification(data: VerificationData): void {
    this.currentState = 'verifying';

    // Clean up progress indicator
    if (this.progressIndicator) {
      this.progressIndicator.destroy();
      this.progressIndicator = null;
    }

    // Create verification UI
    this.verificationUI = new VerificationUI(
      this.contentElement,
      data,
      this.handlers.onVerificationSubmit
    );
    this.verificationUI.render();
  }

  /**
   * Show success state
   */
  showSuccess(): void {
    this.currentState = 'success';

    // Clean up
    if (this.progressIndicator) {
      this.progressIndicator.destroy();
      this.progressIndicator = null;
    }
    if (this.verificationUI) {
      this.verificationUI.destroy();
      this.verificationUI = null;
    }

    this.contentElement.innerHTML = `
      <div class="pouw-success">
        <div class="pouw-success-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <polyline points="20 6 9 17 4 12"></polyline>
          </svg>
        </div>
        <span class="pouw-success-text">Verification complete</span>
      </div>
    `;

    announceToScreenReader('Security verification complete', 'polite');
  }

  /**
   * Show error state
   */
  showError(message: string): void {
    this.currentState = 'error';

    // Clean up
    if (this.progressIndicator) {
      this.progressIndicator.destroy();
      this.progressIndicator = null;
    }
    if (this.verificationUI) {
      this.verificationUI.destroy();
      this.verificationUI = null;
    }

    this.contentElement.innerHTML = `
      <div class="pouw-error">
        <div class="pouw-error-message">
          <svg class="pouw-error-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/>
            <line x1="12" y1="8" x2="12" y2="12"/>
            <line x1="12" y1="16" x2="12.01" y2="16"/>
          </svg>
          <span>${this.escapeHtml(message)}</span>
        </div>
        <button class="pouw-button pouw-button--secondary pouw-retry-btn">
          Try again
        </button>
      </div>
    `;

    // Add retry handler
    const retryBtn = this.contentElement.querySelector('.pouw-retry-btn');
    if (retryBtn) {
      retryBtn.addEventListener('click', () => {
        this.handlers.onRetry();
      });
    }

    announceToScreenReader(`Error: ${message}. Press the try again button to retry.`, 'assertive');
  }

  /**
   * Reset widget to initial state
   */
  reset(): void {
    if (this.progressIndicator) {
      this.progressIndicator.destroy();
      this.progressIndicator = null;
    }
    if (this.verificationUI) {
      this.verificationUI.destroy();
      this.verificationUI = null;
    }

    this.currentState = 'idle';
    this.renderIdleState();
  }

  /**
   * Destroy the widget
   */
  destroy(): void {
    if (this.progressIndicator) {
      this.progressIndicator.destroy();
    }
    if (this.verificationUI) {
      this.verificationUI.destroy();
    }

    // Clear shadow DOM
    while (this.shadowRoot.firstChild) {
      this.shadowRoot.removeChild(this.shadowRoot.firstChild);
    }
  }

  /**
   * Escape HTML to prevent XSS
   */
  private escapeHtml(text: string): string {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  /**
   * Get current state
   */
  getState(): WidgetState {
    return this.currentState;
  }
}
