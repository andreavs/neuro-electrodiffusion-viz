import { describe, expect, it } from 'vitest';
import { gather, membraneMesh, sliceAxisOf, sliceMesh, slicePlane, stimulatedCells } from './geometry';

// Two triangles in the z = 0 and z = 2 planes, sharing nothing.
const nodes = Float32Array.of(0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 2, 1, 0, 2, 1, 1, 2);
const tris = Float32Array.of(0, 1, 2, 3, 4, 5);

describe('membraneMesh', () => {
  it('drops triangles reaching past the cut and maps vertices to membrane nodes', () => {
    const all = membraneMesh(nodes, tris, 2, 5);
    expect(all.positions).toHaveLength(18);
    expect(Array.from(all.index)).toEqual([0, 1, 2, 3, 4, 5]);
    const cut = membraneMesh(nodes, tris, 2, 1);
    expect(Array.from(cut.source)).toEqual([0, 1, 2]);
    expect(membraneMesh(nodes, tris, 2, -1).index).toHaveLength(0);
  });
});

describe('sliceMesh', () => {
  const corners = new Float32Array(18).map((_, i) => (i % 3 === 2 ? 0.7 : i));
  it('hides intracellular triangles unless asked', () => {
    const tags = Float32Array.of(0, 3);
    expect(Array.from(sliceMesh(corners, tags, false).source)).toEqual([0, 1, 2]);
    expect(Array.from(sliceMesh(corners, tags, true).source)).toEqual([0, 1, 2, 3, 4, 5]);
    expect(sliceAxisOf(corners)).toBe(2);
    // A first triangle that is a single point does not fool it.
    const yCut = Float32Array.of(1, 0.5, 1, 1, 0.5, 1, 1, 0.5, 1, 0, 0.5, 0, 2, 0.5, 0, 0, 0.5, 3);
    expect(sliceAxisOf(yCut)).toBe(1);
    expect(slicePlane(corners, 2)).toBeCloseTo(0.7);
  });

  it('gathers per-corner values', () => {
    const m = sliceMesh(corners, Float32Array.of(1, 0), false);
    const values = Float32Array.of(9, 9, 9, 1, 2, 3);
    expect(Array.from(gather(values, m.source, new Float32Array(3)))).toEqual([1, 2, 3]);
  });
});

it('decodes the stimulated mask', () => {
  expect([...stimulatedCells(0b10001, 9)]).toEqual([1, 5]);
  expect(stimulatedCells(0, 4).size).toBe(0);
});
