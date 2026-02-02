/**
 * Progress Indicator Component
 */

/**
 * Progress indicator for processing state
 */
export class ProgressIndicator {
  private container: HTMLElement;
  private progressBar: HTMLElement;
  private progressText: HTMLElement;
  private currentProgress = 0;
  private animationFrame: number | null = null;

  constructor(container: HTMLElement) {
    this.container = container;
    this.progressBar = this.createProgressBar();
    this.progressText = this.createProgressText();
    this.render();
  }

  /**
   * Create progress bar element
   */
  private createProgressBar(): HTMLElement {
    const wrapper = document.createElement('div');
    wrapper.className = 'pouw-progress-container';
    wrapper.setAttribute('role', 'progressbar');
    wrapper.setAttribute('aria-valuemin', '0');
    wrapper.setAttribute('aria-valuemax', '100');
    wrapper.setAttribute('aria-valuenow', '0');
    wrapper.setAttribute('aria-label', 'Security check progress');

    const bar = document.createElement('div');
    bar.className = 'pouw-progress-bar';

    const fill = document.createElement('div');
    fill.className = 'pouw-progress-fill';

    bar.appendChild(fill);
    wrapper.appendChild(bar);

    return wrapper;
  }

  /**
   * Create progress text element
   */
  private createProgressText(): HTMLElement {
    const text = document.createElement('div');
    text.className = 'pouw-progress-text';
    text.textContent = '0%';
    return text;
  }

  /**
   * Render progress indicator
   */
  private render(): void {
    this.container.appendChild(this.progressBar);
    this.container.appendChild(this.progressText);
  }

  /**
   * Set progress value (0-100)
   */
  setProgress(value: number): void {
    const clampedValue = Math.max(0, Math.min(100, value));

    // Cancel any ongoing animation
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
    }

    // Animate to new value
    this.animateTo(clampedValue);
  }

  /**
   * Animate progress to target value
   */
  private animateTo(targetValue: number): void {
    const startValue = this.currentProgress;
    const startTime = performance.now();
    const duration = 300; // ms

    const animate = (currentTime: number) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);

      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      this.currentProgress = startValue + (targetValue - startValue) * eased;

      this.updateDisplay();

      if (progress < 1) {
        this.animationFrame = requestAnimationFrame(animate);
      }
    };

    this.animationFrame = requestAnimationFrame(animate);
  }

  /**
   * Update visual display
   */
  private updateDisplay(): void {
    const fill = this.progressBar.querySelector('.pouw-progress-fill') as HTMLElement;
    if (fill) {
      fill.style.width = `${this.currentProgress}%`;
    }

    this.progressText.textContent = `${Math.round(this.currentProgress)}%`;
    this.progressBar.setAttribute('aria-valuenow', String(Math.round(this.currentProgress)));
  }

  /**
   * Show indeterminate state
   */
  setIndeterminate(): void {
    const fill = this.progressBar.querySelector('.pouw-progress-fill') as HTMLElement;
    if (fill) {
      fill.style.width = '100%';
      fill.style.animation = 'pouw-indeterminate 1.5s ease-in-out infinite';
    }
    this.progressText.textContent = '';
  }

  /**
   * Reset progress
   */
  reset(): void {
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
    }

    this.currentProgress = 0;
    this.updateDisplay();

    const fill = this.progressBar.querySelector('.pouw-progress-fill') as HTMLElement;
    if (fill) {
      fill.style.animation = '';
    }
  }

  /**
   * Destroy component
   */
  destroy(): void {
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
    }

    this.progressBar.remove();
    this.progressText.remove();
  }
}
