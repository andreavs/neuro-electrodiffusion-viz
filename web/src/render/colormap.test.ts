import { describe, expect, it } from 'vitest';
import { LUT_SIZE, coolwarm, lightnessAt, sampleColormap, viridis } from './colormap';

const rgb = (lut: Uint8Array, i: number) => [lut[4 * i], lut[4 * i + 1], lut[4 * i + 2]];

describe('viridis', () => {
  it('has 256 opaque entries', () => {
    expect(viridis.lut.length).toBe(LUT_SIZE * 4);
    for (let i = 0; i < LUT_SIZE; i++) expect(viridis.lut[4 * i + 3]).toBe(255);
  });

  it('matches matplotlib endpoints within 4/255', () => {
    const start = rgb(viridis.lut, 0);
    const end = rgb(viridis.lut, LUT_SIZE - 1);
    [68, 1, 84].forEach((c, k) => expect(Math.abs(start[k] - c)).toBeLessThanOrEqual(4));
    [253, 231, 37].forEach((c, k) => expect(Math.abs(end[k] - c)).toBeLessThanOrEqual(4));
  });

  it('has monotonically increasing lightness (up to 8-bit rounding)', () => {
    for (let i = 1; i < LUT_SIZE; i++)
      expect(lightnessAt(viridis.lut, i)).toBeGreaterThan(lightnessAt(viridis.lut, i - 1) - 0.1);
    for (let i = 8; i < LUT_SIZE; i += 8)
      expect(lightnessAt(viridis.lut, i)).toBeGreaterThan(lightnessAt(viridis.lut, i - 8) + 1);
  });
});

describe('coolwarm', () => {
  it('runs blue → light → red', () => {
    const [r0, , b0] = rgb(coolwarm.lut, 0);
    const [r1, , b1] = rgb(coolwarm.lut, LUT_SIZE - 1);
    expect(b0).toBeGreaterThan(r0);
    expect(r1).toBeGreaterThan(b1);
    expect(coolwarm.diverging).toBe(true);
  });

  it('has lightness rising to the centre then falling', () => {
    const mid = LUT_SIZE / 2;
    for (let i = 1; i < mid; i++) expect(lightnessAt(coolwarm.lut, i)).toBeGreaterThanOrEqual(lightnessAt(coolwarm.lut, i - 1));
    for (let i = mid + 1; i < LUT_SIZE; i++)
      expect(lightnessAt(coolwarm.lut, i)).toBeLessThanOrEqual(lightnessAt(coolwarm.lut, i - 1));
    // Ends are similarly dark, so +v and -v read with equal weight.
    expect(Math.abs(lightnessAt(coolwarm.lut, 0) - lightnessAt(coolwarm.lut, LUT_SIZE - 1))).toBeLessThan(6);
  });
});

describe('sampleColormap', () => {
  it('clamps and handles non-finite input', () => {
    expect(sampleColormap(viridis, -1)).toEqual(sampleColormap(viridis, 0));
    expect(sampleColormap(viridis, 2)).toEqual(sampleColormap(viridis, 1));
    expect(sampleColormap(viridis, NaN)).toEqual(sampleColormap(viridis, 0));
  });
});
