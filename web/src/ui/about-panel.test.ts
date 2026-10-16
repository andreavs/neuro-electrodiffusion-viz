// @vitest-environment happy-dom
import { describe, expect, it } from 'vitest';
import { createAboutPanel } from './about';

function deferred() {
  let resolve!: (html: string) => void;
  let reject!: (err: unknown) => void;
  const promise = new Promise<string>((res, rej) => ((resolve = res), (reject = rej)));
  return { promise, resolve, reject };
}

const flush = () => new Promise((r) => setTimeout(r, 0));

describe('createAboutPanel', () => {
  it('starts collapsed with a focusable summary', () => {
    const panel = createAboutPanel('knp', () => new Promise(() => {}));
    expect(panel.open).toBe(false);
    expect(panel.querySelector('summary')?.textContent).toBe('About this model');
  });

  it('opens only after the body is rendered', async () => {
    const d = deferred();
    const panel = createAboutPanel('knp', () => d.promise);
    document.body.append(panel);
    const summary = panel.querySelector('summary')!;

    summary.click();
    expect(panel.open).toBe(false);
    expect(summary.getAttribute('aria-busy')).toBe('true');

    d.resolve('<p>content</p>');
    await flush();
    expect(panel.open).toBe(true);
    expect(summary.hasAttribute('aria-busy')).toBe(false);
    expect(panel.querySelector('.about-body')!.innerHTML).toBe('<p>content</p>');
    panel.remove();
  });

  it('prefetches once on focus and hover', async () => {
    let calls = 0;
    const panel = createAboutPanel('emi', async () => {
      calls++;
      return '<p>x</p>';
    });
    const summary = panel.querySelector('summary')!;
    summary.dispatchEvent(new Event('focus'));
    summary.dispatchEvent(new Event('pointerenter'));
    await flush();
    expect(calls).toBe(1);
  });

  it('shows an error and retries on the next open after a failed load', async () => {
    let attempt = 0;
    const panel = createAboutPanel('particles', async () => {
      attempt++;
      if (attempt === 1) throw new Error('offline');
      return '<p>ok</p>';
    });
    document.body.append(panel);
    const summary = panel.querySelector('summary')!;
    const body = panel.querySelector('.about-body')!;

    summary.click();
    await flush();
    expect(panel.open).toBe(true);
    expect(body.textContent).toContain('offline');

    panel.open = false;
    await flush();
    summary.click();
    await flush();
    expect(attempt).toBe(2);
    expect(body.innerHTML).toBe('<p>ok</p>');
    expect(panel.open).toBe(true);
    panel.remove();
  });
});
