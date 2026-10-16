/** Pure helpers turning the EMI frame's triangles into indexed triangle meshes. */

export interface TriMesh {
  /** xyz per vertex, µm. */
  positions: Float32Array;
  /** One triangle per kept face. */
  index: Uint32Array;
  /** Source of each vertex's value: a membrane node, or a slice corner (3 · triangle + corner). */
  source: Uint32Array;
}

function assemble(kept: number[], corner: (q: number, c: number) => [number, number, number, number]): TriMesh {
  const positions = new Float32Array(kept.length * 9);
  const source = new Uint32Array(kept.length * 3);
  const index = new Uint32Array(kept.length * 3);
  kept.forEach((q, k) => {
    for (let c = 0; c < 3; c++) {
      const [x, y, z, src] = corner(q, c);
      positions.set([x, y, z], 9 * k + 3 * c);
      source[3 * k + c] = src;
      index[3 * k + c] = 3 * k + c;
    }
  });
  return { positions, index, source };
}

/**
 * The cell surfaces, with every triangle that reaches past `cut` along `axis` removed so the
 * slice behind it stays visible. Each triangle gets its own three vertices (flat shading).
 */
export function membraneMesh(nodes: Float32Array, tris: Float32Array, axis: number, cut: number): TriMesh {
  const kept: number[] = [];
  const eps = 1e-6 * Math.max(1, Math.abs(cut));
  for (let q = 0; q < tris.length / 3; q++) {
    let max = -Infinity;
    for (let c = 0; c < 3; c++) max = Math.max(max, nodes[3 * tris[3 * q + c]! + axis]!);
    if (max <= cut + eps) kept.push(q);
  }
  return assemble(kept, (q, c) => {
    const n = tris[3 * q + c]!;
    return [nodes[3 * n]!, nodes[3 * n + 1]!, nodes[3 * n + 2]!, n];
  });
}

/** The slice triangles, keeping intracellular ones (tag > 0) only when `inside` is set. */
export function sliceMesh(corners: Float32Array, tags: Float32Array, inside: boolean): TriMesh {
  const kept: number[] = [];
  for (let q = 0; q < tags.length; q++) if (inside || tags[q] === 0) kept.push(q);
  return assemble(kept, (q, c) => {
    const i = 9 * q + 3 * c;
    return [corners[i]!, corners[i + 1]!, corners[i + 2]!, 3 * q + c];
  });
}

/** `out[i] = values[source[i]]`. */
export function gather(values: ArrayLike<number>, source: Uint32Array, out: Float32Array): Float32Array {
  for (let i = 0; i < source.length; i++) out[i] = values[source[i]!]!;
  return out;
}

/** The slice plane's coordinate along its normal axis (the corners all share it). */
export function slicePlane(corners: Float32Array, axis: number): number {
  return corners.length ? corners[axis]! : 0;
}

/** Cell tags (1-based) whose bit is set in the `stimulated` mask. */
export function stimulatedCells(mask: number, nCells: number): Set<number> {
  const out = new Set<number>();
  for (let c = 1; c <= nCells; c++) if (mask & (1 << (c - 1))) out.add(c);
  return out;
}

/** Normal axis of the slice: the one along which the corners spread least (not at all). */
export function sliceAxisOf(corners: Float32Array): number {
  const spread = [0, 1, 2].map((a) => {
    const c = corners.filter((_, i) => i % 3 === a);
    return c.reduce((m, v) => Math.max(m, v), -Infinity) - c.reduce((m, v) => Math.min(m, v), Infinity);
  });
  return corners.length < 9 ? 2 : spread.indexOf(Math.min(...spread));
}
