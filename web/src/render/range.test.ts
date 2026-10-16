import { describe, expect, it } from 'vitest';
import { ExpandingRange, autoRange, restRange } from './range';
import { niceTicks } from './ticks';

describe('autoRange', () => {
  it('finds min and max, ignoring non-finite values', () => {
    expect(autoRange([3, -1, NaN, 7, Infinity])).toEqual({ min: -1, max: 7 });
  });
  it('centres on zero when symmetric', () => {
    expect(autoRange(new Float32Array([-2, 5]), true)).toEqual({ min: -5, max: 5 });
  });
  it('pads constant data and defaults on empty input', () => {
    const r = autoRange([4, 4, 4]);
    expect(r.min).toBeLessThan(4);
    expect(r.max).toBeGreaterThan(4);
    expect(autoRange([])).toEqual({ min: 0, max: 1 });
  });
});

describe('niceTicks', () => {
  it('uses 1/2/5 steps inside the range', () => {
    expect(niceTicks(0, 10, 5)).toEqual([0, 2, 4, 6, 8, 10]);
    expect(niceTicks(-70, 30, 4)).toEqual([-60, -40, -20, 0, 20]);
    const t = niceTicks(0.013, 0.087, 4);
    expect(t[0]).toBeGreaterThanOrEqual(0.013);
    expect(t[t.length - 1]).toBeLessThanOrEqual(0.087);
  });
});

describe('ExpandingRange', () => {
  it('fits the first data exactly, then only grows (with headroom)', () => {
    const r = new ExpandingRange(0.1);
    expect(r.empty).toBe(true);
    expect(r.include(0, 10)).toBe(true);
    expect([r.min, r.max]).toEqual([0, 10]);
    expect(r.include(2, 8)).toBe(false);
    expect([r.min, r.max]).toEqual([0, 10]);
    expect(r.include(0, 20)).toBe(true);
    expect(r.min).toBe(0);
    expect(r.max).toBeCloseTo(22);
    expect(r.include(-10, 5)).toBe(true);
    expect(r.min).toBeCloseTo(-13.2);
    expect(r.max).toBeCloseTo(22);
  });

  it('ignores non-finite input and starts over after reset', () => {
    const r = new ExpandingRange();
    expect(r.include(NaN, 1)).toBe(false);
    expect(r.include(0, Infinity)).toBe(false);
    r.include(1, 2);
    r.reset();
    expect(r.empty).toBe(true);
    r.include(5, 6);
    expect([r.min, r.max]).toEqual([5, 6]);
  });
});

describe('restRange', () => {
  it('centres on the rest value, ignores masked cells and keeps a minimum width', () => {
    const field = new Float32Array([3, 3, 3.2, 2.9, 50, 3]);
    const mask = new Uint8Array([0, 0, 0, 0, 1, 0]);
    const [lo, hi] = restRange(field, mask, 3, 1, 1, new Float32Array(6));
    expect(lo).toBeCloseTo(2.8);
    expect(hi).toBeCloseTo(3.2);
    expect(restRange(new Float32Array([3, 3]), new Uint8Array(2), 3, 1, 1, new Float32Array(2))).toEqual([
      3 - 3e-3,
      3 + 3e-3,
    ]);
  });
});
