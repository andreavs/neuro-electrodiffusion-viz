// @vitest-environment happy-dom
import { describe, expect, it } from 'vitest';
import type { Params } from '../../sim/protocol';
import { createTour, TOUR } from './tour';

describe('createTour', () => {
  it('applies each step, releases when nothing changed and stops at the ends', () => {
    const applied: Params[] = [];
    let releases = 0;
    let changed = true;
    const el = createTour({ apply: (p) => (applied.push(p), changed), release: () => void releases++ });
    const click = (which: string) => el.querySelector<HTMLButtonElement>(`[data-tour="${which}"]`)!.click();
    click('next');
    expect(applied).toEqual([TOUR[0]!.params]);
    expect(el.querySelector('h3')!.textContent).toBe(TOUR[0]!.title);
    expect(el.querySelector<HTMLButtonElement>('[data-tour="prev"]')!.disabled).toBe(true);
    changed = false;
    click('next');
    expect(releases).toBe(1);
    for (let i = 0; i < TOUR.length; i++) if (!el.querySelector<HTMLButtonElement>('[data-tour="next"]')!.disabled) click('next');
    expect(el.textContent).toContain(`${TOUR.length} of ${TOUR.length}`);
    click('prev');
    expect(el.querySelector('h3')!.textContent).toBe(TOUR[TOUR.length - 2]!.title);
  });
});
