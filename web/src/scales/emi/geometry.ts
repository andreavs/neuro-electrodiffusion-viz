/** Pure helpers turning the EMI frame's quads into indexed triangle meshes. */

export interface QuadMesh {
  /** xyz per vertex, µm. */
  positions: Float32Array;
  /** Two triangles per kept quad. */
  index: Uint32Array;
  /** Source of each vertex's value: a membrane node, or a slice corner (4 · quad + corner). */
  source: Uint32Array;
}

/** Same-sided triangulation of a quad whose corners are 0..3 in order. */
const TRI = [0, 1, 2, 0, 2, 3];

function assemble(kept: number[], corner: (q: number, c: number) => [number, number, number, number]): QuadMesh {
  const positions = new Float32Array(kept.length * 12);
  const source = new Uint32Array(kept.length * 4);
  const index = new Uint32Array(kept.length * 6);
  kept.forEach((q, k) => {
    for (let c = 0; c < 4; c++) {
      const [x, y, z, src] = corner(q, c);
      positions.set([x, y, z], 12 * k + 3 * c);
      source[4 * k + c] = src;
    }
    for (let t = 0; t < 6; t++) index[6 * k + t] = 4 * k + TRI[t]!;
  });
  return { positions, index, source };
}

/**
 * The axon surfaces, with every quad that reaches past `cut` along `axis` removed so the
 * slice behind it stays visible. Each quad gets its own four vertices (flat shading per face).
 */
export function membraneMesh(nodes: Float32Array, quads: Float32Array, axis: number, cut: number): QuadMesh {
  const kept: number[] = [];
  const eps = 1e-6 * Math.max(1, Math.abs(cut));
  for (let q = 0; q < quads.length / 4; q++) {
    let max = -Infinity;
    for (let c = 0; c < 4; c++) max = Math.max(max, nodes[3 * quads[4 * q + c]! + axis]!);
    if (max <= cut + eps) kept.push(q);
  }
  return assemble(kept, (q, c) => {
    const n = quads[4 * q + c]!;
    return [nodes[3 * n]!, nodes[3 * n + 1]!, nodes[3 * n + 2]!, n];
  });
}

/** The slice quads, keeping intracellular ones (tag > 0) only when `inside` is set. */
export function sliceMesh(corners: Float32Array, tags: Float32Array, inside: boolean): QuadMesh {
  const kept: number[] = [];
  for (let q = 0; q < tags.length; q++) if (inside || tags[q] === 0) kept.push(q);
  return assemble(kept, (q, c) => {
    const i = 12 * q + 3 * c;
    return [corners[i]!, corners[i + 1]!, corners[i + 2]!, 4 * q + c];
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

/** Normal axis of the slice: the one along which a quad's corners share their coordinate. */
export function sliceAxisOf(corners: Float32Array): number {
  for (let a = 0; a < 3; a++) {
    if (corners[a] === corners[3 + a] && corners[a] === corners[6 + a] && corners[a] === corners[9 + a]) return a;
  }
  return 2;
}
