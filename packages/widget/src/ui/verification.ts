/**
 * Human Verification UI Component
 */

import type { VerificationData, VerificationResponse } from '../types';
import {
  announceToScreenReader,
  trapFocus,
  handleEscapeKey,
  createAccessibleButton,
} from '../utils/accessibility';
import { PerformanceTimer } from '../utils/timing';

/**
 * Verification UI Component
 */
export class VerificationUI {
  private container: HTMLElement;
  private data: VerificationData;
  private onSubmit: (response: VerificationResponse) => void;
  private releaseFocusTrap: (() => void) | null = null;
  private releaseEscapeHandler: (() => void) | null = null;
  private timer: PerformanceTimer;
  private element: HTMLElement | null = null;

  constructor(
    container: HTMLElement,
    data: VerificationData,
    onSubmit: (response: VerificationResponse) => void
  ) {
    this.container = container;
    this.data = data;
    this.onSubmit = onSubmit;
    this.timer = new PerformanceTimer();
  }

  /**
   * Render the verification UI
   */
  render(): void {
    this.timer.start('verification');

    this.element = document.createElement('div');
    this.element.className = 'pouw-verification';
    this.element.setAttribute('role', 'dialog');
    this.element.setAttribute('aria-modal', 'true');
    this.element.setAttribute('aria-labelledby', 'pouw-verification-title');

    // Prompt
    const prompt = document.createElement('p');
    prompt.id = 'pouw-verification-title';
    prompt.className = 'pouw-verification-prompt';
    prompt.textContent = this.data.prompt;

    // Content (image or text)
    const content = this.createContent();

    // Label display
    const label = document.createElement('div');
    label.className = 'pouw-verification-label';
    label.textContent = `"${this.data.predictedLabel}"`;

    // Action buttons
    const actions = this.createActions();

    // Correction UI (hidden initially)
    const correction = this.createCorrectionUI();

    this.element.appendChild(prompt);
    this.element.appendChild(content);
    this.element.appendChild(label);
    this.element.appendChild(actions);
    this.element.appendChild(correction);

    // Clear container and add verification UI
    this.container.innerHTML = '';
    this.container.appendChild(this.element);

    // Set up accessibility
    this.releaseFocusTrap = trapFocus(this.element);

    // Announce to screen readers
    announceToScreenReader(
      `Verification required: ${this.data.prompt} The predicted answer is ${this.data.predictedLabel}`,
      'polite'
    );
  }

  /**
   * Create content display (image or text)
   */
  private createContent(): HTMLElement {
    const content = document.createElement('div');
    content.className = 'pouw-verification-content';

    if (this.data.displayType === 'image') {
      const img = document.createElement('img');
      img.className = 'pouw-verification-image';
      img.src = this.data.displayContent;
      img.alt = 'Image to verify';
      img.loading = 'eager';
      content.appendChild(img);
    } else {
      const text = document.createElement('div');
      text.className = 'pouw-verification-text';
      text.textContent = this.data.displayContent;
      content.appendChild(text);
    }

    return content;
  }

  /**
   * Create action buttons
   */
  private createActions(): HTMLElement {
    const actions = document.createElement('div');
    actions.className = 'pouw-verification-actions';

    // Yes/Confirm button
    const confirmBtn = createAccessibleButton({
      text: 'Yes, correct',
      ariaLabel: `Confirm that the answer "${this.data.predictedLabel}" is correct`,
      onClick: () => this.handleConfirm(),
      className: 'pouw-button pouw-button--primary',
    });

    // No/Reject button
    const rejectBtn = createAccessibleButton({
      text: 'No, wrong',
      ariaLabel: `Indicate that "${this.data.predictedLabel}" is incorrect`,
      onClick: () => this.handleReject(),
      className: 'pouw-button pouw-button--secondary',
    });

    actions.appendChild(confirmBtn);
    actions.appendChild(rejectBtn);

    return actions;
  }

  /**
   * Create correction UI
   */
  private createCorrectionUI(): HTMLElement {
    const correction = document.createElement('div');
    correction.className = 'pouw-correction';
    correction.id = 'pouw-correction';

    const label = document.createElement('label');
    label.className = 'pouw-correction-label';
    label.htmlFor = 'pouw-correction-select';
    label.textContent = 'What is the correct answer?';

    const select = document.createElement('select');
    select.className = 'pouw-correction-select';
    select.id = 'pouw-correction-select';

    // Add placeholder option
    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = 'Select the correct label...';
    placeholder.disabled = true;
    placeholder.selected = true;
    select.appendChild(placeholder);

    // Add label options (would come from model metadata)
    // For now, using generic options
    const labels = this.getCorrectionLabels();
    for (const labelText of labels) {
      const option = document.createElement('option');
      option.value = labelText;
      option.textContent = labelText;
      select.appendChild(option);
    }

    const submitBtn = createAccessibleButton({
      text: 'Submit correction',
      onClick: () => this.handleCorrection(select.value),
      className: 'pouw-button pouw-button--primary',
    });

    correction.appendChild(label);
    correction.appendChild(select);
    correction.appendChild(submitBtn);

    return correction;
  }

  /**
   * Get available correction labels
   */
  private getCorrectionLabels(): string[] {
    // This would typically come from the verification data
    // Using CIFAR-10 labels as example
    return [
      'airplane',
      'automobile',
      'bird',
      'cat',
      'deer',
      'dog',
      'frog',
      'horse',
      'ship',
      'truck',
    ].filter((l) => l !== this.data.predictedLabel);
  }

  /**
   * Handle confirm action
   */
  private handleConfirm(): void {
    this.timer.end('verification');

    const response: VerificationResponse = {
      responseType: 'confirm',
      responseTimeMs: this.timer.get('verification'),
    };

    this.cleanup();
    this.onSubmit(response);
  }

  /**
   * Handle reject action
   */
  private handleReject(): void {
    // Show correction UI
    const correction = this.element?.querySelector('.pouw-correction');
    if (correction) {
      correction.classList.add('visible');
      const select = correction.querySelector('select');
      select?.focus();

      announceToScreenReader(
        'Please select the correct label from the dropdown',
        'polite'
      );
    }
  }

  /**
   * Handle correction submission
   */
  private handleCorrection(correctedLabel: string): void {
    if (!correctedLabel) {
      announceToScreenReader('Please select a label', 'assertive');
      return;
    }

    this.timer.end('verification');

    const response: VerificationResponse = {
      responseType: 'correct',
      correctedLabel,
      responseTimeMs: this.timer.get('verification'),
    };

    this.cleanup();
    this.onSubmit(response);
  }

  /**
   * Clean up event listeners
   */
  private cleanup(): void {
    if (this.releaseFocusTrap) {
      this.releaseFocusTrap();
      this.releaseFocusTrap = null;
    }

    if (this.releaseEscapeHandler) {
      this.releaseEscapeHandler();
      this.releaseEscapeHandler = null;
    }
  }

  /**
   * Destroy the component
   */
  destroy(): void {
    this.cleanup();
    if (this.element) {
      this.element.remove();
      this.element = null;
    }
  }
}
