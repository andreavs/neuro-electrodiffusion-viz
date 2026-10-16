import { describe, expect, it } from 'vitest';
import { RateMeter, buildNeuron, clipRange, gridToWorld, spreadValues, worldToGrid } from './neuron';

describe('neuron mesh', () => {
  const size: [number, number, number] = [100, 100, 200];
  // A soma and one vertical dendrite.
  const comps = Float32Array.of(50, 50, 40, 50, 50, 60, 20, 50, 50, 60, 50, 50, 160, 2);

  it('maps between grid and world coordinates', () => {
    expect(gridToWorld([50, 50, 100], size)).toEqual([0, 0, 0]);
    const p = worldToGrid(gridToWorld([10, 20, 30], size), size);
    p.forEach((v, i) => expect(v).toBeCloseTo([10, 20, 30][i]!, 9));
  });
  it('builds one cylinder per compartment and spreads values onto it', () => {
    const { geometry, vertexComp } = buildNeuron(comps, size);
    const pos = geometry.getAttribute('position');
    expect(vertexComp.length).toBe(pos.count);
    expect(new Set(vertexComp)).toEqual(new Set([0, 1]));
    // The dendrite reaches from z = 60 µm to 160 µm (world −0.04 to 0.06).
    let zMax = -Infinity;
    for (let i = 0; i < pos.count; i++) if (vertexComp[i] === 1) zMax = Math.max(zMax, pos.getZ(i));
    expect(zMax).toBeCloseTo(0.06, 6);
    const out = spreadValues([-65, 20], vertexComp, new Float32Array(pos.count));
    expect(out[0]).toBe(-65);
    expect(out[pos.count - 1]).toBe(20);
  });
});

describe('RateMeter', () => {
  it('counts spikes per second over the window and restarts on reset', () => {
    const m = new RateMeter(1000);
    m.push(0, 0);
    expect(m.push(500, 5)).toBeCloseTo(10);
    expect(m.push(2000, 35)).toBeCloseTo(20);
    expect(m.push(10, 0)).toBe(0);
  });
});

describe('clipRange', () => {
  it('drops the extremes beyond the quantiles', () => {
    const a = Float32Array.from({ length: 101 }, (_, i) => i);
    a[100] = 1e6;
    expect(clipRange([a, Float32Array.of(NaN)], 0.98, [])).toEqual([2, 98]);
    expect(clipRange([], 0.99, [])).toEqual([0, 1]);
  });
});
