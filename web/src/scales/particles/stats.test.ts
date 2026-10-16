import { describe, expect, it } from 'vitest';
import { formatMfpt, histogramSteps, maxOf, mfptReadout } from './stats';

describe('mfptReadout', () => {
  const stats = { mfpt: 10, mfptSe: 0.5, escaped: 100, mfptCensored: 40, refEscaped: 0 };

  it('uses the censored estimate while ions remain and the sample mean once all escaped', () => {
    expect(mfptReadout(stats, '', 25)).toEqual({ mean: 40, se: 4, final: false });
    expect(mfptReadout(stats, '', 0)).toEqual({ mean: 10, se: 0.5, final: true });
  });

  it('is null before the first escape', () => {
    expect(mfptReadout(stats, 'ref', 10)).toBeNull();
    expect(mfptReadout({}, '', 10)).toBeNull();
  });
});

describe('formatMfpt', () => {
  it('marks running estimates and scales the precision', () => {
    expect(formatMfpt(null)).toBe('–');
    expect(formatMfpt({ mean: 46.71, se: 1.42, final: false })).toBe('≈ 46.7 ± 1.4 ms');
    expect(formatMfpt({ mean: 3.456, se: Number.NaN, final: true })).toBe('3.46 ms');
    expect(formatMfpt({ mean: 120.4, se: 3.6, final: true })).toBe('120 ± 4 ms');
  });
});

describe('histogramSteps', () => {
  it('outlines each bin and closes at zero on both ends', () => {
    expect([...histogramSteps([1, 3], 4)]).toEqual([0, 0, 0, 1, 2, 1, 2, 3, 4, 3, 4, 0]);
  });
});

describe('maxOf', () => {
  it('skips missing arrays and respects the floor', () => {
    expect(maxOf([[1, 5], undefined, [2]], 0)).toBe(5);
    expect(maxOf([[0.001]], 0.01)).toBe(0.01);
  });
});
