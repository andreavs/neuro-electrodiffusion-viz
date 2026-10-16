import { describe, expect, it } from 'vitest';
import { gather, membraneMesh, sliceAxisOf, sliceMesh, slicePlane, stimulatedCells } from './geometry';

// Two unit quads in the z = 0 and z = 2 planes, sharing nothing.
const nodes = Float32Array.of(0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 2, 1, 0, 2, 1, 1, 2, 0, 1, 2);
const quads = Float32Array.of(0, 1, 2, 3, 4, 5, 6, 7);

describe('membraneMesh', () => {
  it('drops quads reaching past the cut and maps vertices to membrane nodes', () => {
    const all = membraneMesh(nodes, quads, 2, 5);
    expect(all.positions).toHaveLength(24);
    expect(Array.from(all.index)).toEqual([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]);
    const cut = membraneMesh(nodes, quads, 2, 1);
    expect(Array.from(cut.source)).toEqual([0, 1, 2, 3]);
    expect(membraneMesh(nodes, quads, 2, -1).index).toHaveLength(0);
  });
});

describe('sliceMesh', () => {
  const corners = new Float32Array(24).map((_, i) => (i % 3 === 2 ? 0.7 : i));
  it('hides intracellular quads unless asked', () => {
    const tags = Float32Array.of(0, 3);
    expect(Array.from(sliceMesh(corners, tags, false).source)).toEqual([0, 1, 2, 3]);
    expect(Array.from(sliceMesh(corners, tags, true).source)).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    expect(sliceAxisOf(corners)).toBe(2);
    expect(slicePlane(corners, 2)).toBeCloseTo(0.7);
  });

  it('gathers per-corner values', () => {
    const m = sliceMesh(corners, Float32Array.of(1, 0), false);
    const values = Float32Array.of(9, 9, 9, 9, 1, 2, 3, 4);
    expect(Array.from(gather(values, m.source, new Float32Array(4)))).toEqual([1, 2, 3, 4]);
  });
});

it('decodes the stimulated mask', () => {
  expect([...stimulatedCells(0b10001, 9)]).toEqual([1, 5]);
  expect(stimulatedCells(0, 4).size).toBe(0);
});
