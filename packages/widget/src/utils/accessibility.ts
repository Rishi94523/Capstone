/**
 * Accessibility utilities for PoUW CAPTCHA
 */

/**
 * Announce message to screen readers
 */
export function announceToScreenReader(
  message: string,
  priority: 'polite' | 'assertive' = 'polite'
): void {
  // Find or create live region
  let liveRegion = document.getElementById('pouw-live-region');

  if (!liveRegion) {
    liveRegion = document.createElement('div');
    liveRegion.id = 'pouw-live-region';
    liveRegion.setAttribute('aria-live', priority);
    liveRegion.setAttribute('aria-atomic', 'true');
    liveRegion.setAttribute('role', 'status');
    liveRegion.style.cssText = `
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
    `;
    document.body.appendChild(liveRegion);
  }

  // Clear and set message (triggers announcement)
  liveRegion.textContent = '';
  setTimeout(() => {
    liveRegion!.textContent = message;
  }, 100);
}

/**
 * Trap focus within an element
 */
export function trapFocus(container: HTMLElement): () => void {
  const focusableElements = getFocusableElements(container);

  if (focusableElements.length === 0) {
    return () => {};
  }

  const firstElement = focusableElements[0];
  const lastElement = focusableElements[focusableElements.length - 1];

  // Store previously focused element
  const previouslyFocused = document.activeElement as HTMLElement;

  // Focus first element
  firstElement.focus();

  const handleKeyDown = (event: KeyboardEvent) => {
    if (event.key !== 'Tab') return;

    if (event.shiftKey) {
      // Shift + Tab
      if (document.activeElement === firstElement) {
        event.preventDefault();
        lastElement.focus();
      }
    } else {
      // Tab
      if (document.activeElement === lastElement) {
        event.preventDefault();
        firstElement.focus();
      }
    }
  };

  container.addEventListener('keydown', handleKeyDown);

  // Return cleanup function
  return () => {
    container.removeEventListener('keydown', handleKeyDown);
    previouslyFocused?.focus();
  };
}

/**
 * Get all focusable elements within a container
 */
export function getFocusableElements(container: HTMLElement): HTMLElement[] {
  const selector = [
    'button:not([disabled])',
    'input:not([disabled])',
    'select:not([disabled])',
    'textarea:not([disabled])',
    'a[href]',
    '[tabindex]:not([tabindex="-1"])',
  ].join(', ');

  return Array.from(container.querySelectorAll<HTMLElement>(selector)).filter(
    (el) => !el.hasAttribute('disabled') && el.offsetParent !== null
  );
}

/**
 * Handle Escape key to close dialogs
 */
export function handleEscapeKey(
  onEscape: () => void
): () => void {
  const handleKeyDown = (event: KeyboardEvent) => {
    if (event.key === 'Escape') {
      event.preventDefault();
      onEscape();
    }
  };

  document.addEventListener('keydown', handleKeyDown);

  return () => {
    document.removeEventListener('keydown', handleKeyDown);
  };
}

/**
 * Check if user prefers reduced motion
 */
export function prefersReducedMotion(): boolean {
  if (typeof window === 'undefined') return false;
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/**
 * Check if user prefers high contrast
 */
export function prefersHighContrast(): boolean {
  if (typeof window === 'undefined') return false;
  return window.matchMedia('(prefers-contrast: more)').matches;
}

/**
 * Create accessible button
 */
export function createAccessibleButton(options: {
  text: string;
  ariaLabel?: string;
  onClick: () => void;
  className?: string;
  disabled?: boolean;
}): HTMLButtonElement {
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = options.text;
  button.className = options.className || '';
  button.disabled = options.disabled || false;

  if (options.ariaLabel) {
    button.setAttribute('aria-label', options.ariaLabel);
  }

  button.addEventListener('click', options.onClick);

  // Handle keyboard activation
  button.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      options.onClick();
    }
  });

  return button;
}

/**
 * Generate unique ID for accessibility
 */
export function generateA11yId(prefix: string = 'pouw'): string {
  return `${prefix}-${Math.random().toString(36).substring(2, 9)}`;
}

/**
 * Set up ARIA relationships
 */
export function setupAriaRelationship(
  controller: HTMLElement,
  controlled: HTMLElement,
  type: 'controls' | 'describedby' | 'labelledby' | 'owns'
): void {
  if (!controlled.id) {
    controlled.id = generateA11yId();
  }

  const attrName = `aria-${type}`;
  const existing = controller.getAttribute(attrName);
  const newValue = existing
    ? `${existing} ${controlled.id}`
    : controlled.id;

  controller.setAttribute(attrName, newValue);
}
