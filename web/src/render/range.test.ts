import { describe, expect, it } from 'vitest';
import { autoRange } from './range';
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
