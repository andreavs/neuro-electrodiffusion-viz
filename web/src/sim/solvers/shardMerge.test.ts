import { describe, expect, it } from 'vitest';
import { mergeHistograms, mergeMfpt, shardCount, splitEvenly } from './shardMerge';

describe('shard helpers', () => {
  it('uses one shard per 10 000 ions, leaving two cores free', () => {
    expect([1000, 15_000, 25_000, 1e6].map((n) => shardCount(n, 5))).toEqual([1, 2, 3, 3]);
    expect(shardCount(1e6, 1)).toBe(1);
    expect(shardCount(4e6, 4)).toBe(4);
    expect(splitEvenly(10, 3)).toEqual([4, 3, 3]);
  });

  it('pools exit-time summaries exactly', () => {
    const a = [1, 2, 3, 10];
    const b = [4, 5];
    const summary = (t: number[]) => {
      const m = t.reduce((x, y) => x + y, 0) / t.length;
      const v = t.length > 1 ? t.reduce((x, y) => x + (y - m) ** 2, 0) / (t.length - 1) : NaN;
      return Float64Array.of(m, Math.sqrt(v / t.length), t.length, m);
    };
    const pooled = mergeMfpt([summary(a), summary(b), Float64Array.of(NaN, NaN, 0, NaN)]);
    const direct = summary([...a, ...b]);
    for (let i = 0; i < 4; i++) expect(pooled[i]).toBeCloseTo(direct[i]!, 12);
    expect(mergeMfpt([summary([7])])[1]).toBeNaN();
    // A shard with no exits yet still adds its ions' time inside to the censored estimate.
    const c = Float64Array.of(2, 0, 2, 3); // censored = (exits + inside) / 2 = 3
    expect(mergeMfpt([c, Float64Array.of(NaN, NaN, 0, NaN)], 10 * 100)[3]).toBe((3 * 2 + 1000) / 2);
  });

  it('weights histograms by each shard’s ions', () => {
    const h = mergeHistograms(
      [
        { histogram: Float32Array.of(1, 0), released: 3 },
        { histogram: Float32Array.of(0, 1), released: 1 },
      ],
      2,
    );
    expect([...h]).toEqual([0.75, 0.25]);
  });
});
