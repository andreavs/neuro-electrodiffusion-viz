// @vitest-environment happy-dom
import { describe, expect, it } from 'vitest';
import type { Params } from '../../sim/protocol';
import { createTour, TOUR } from './tour';

describe('createTour', () => {
  it('applies each step, releases when nothing changed, stops at the ends and closes', () => {
    const applied: Params[] = [];
    let releases = 0;
    let changed = true;
    const el = createTour({ apply: (p) => (applied.push(p), changed), release: () => void releases++ });
    const button = (which: string) => el.querySelector<HTMLButtonElement>(`[data-tour="${which}"]`)!;
    const title = () => el.querySelector('h3')!.textContent;
    expect(el.textContent).toContain(`A ${TOUR.length}-step tour`);
    button('next').click();
    expect(applied).toEqual([TOUR[0]!.params]);
    expect(title()).toBe(TOUR[0]!.title);
    expect(button('prev').disabled).toBe(true);
    changed = false;
    button('next').click();
    expect([title(), releases]).toEqual([TOUR[1]!.title, 1]);
    for (let i = 2; i < TOUR.length; i++) button('next').click();
    expect(title()).toBe(TOUR[TOUR.length - 1]!.title);
    expect(button('next').disabled).toBe(true);
    button('prev').click();
    expect(title()).toBe(TOUR[TOUR.length - 2]!.title);
    button('close').click();
    expect([button('prev').hidden, button('next').textContent]).toEqual([true, 'Start the tour']);
    expect(applied).toHaveLength(TOUR.length + 1);
  });
});
