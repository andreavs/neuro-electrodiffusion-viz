/**
 * Indexing for 3-D scalar fields sent by the solvers.
 *
 * A field with shape `[nx, ny, nz]` is a flat `Float32Array` with x varying
 * fastest: `index(x, y, z) = x + nx * (y + ny * z)`.
 */

export type Axis = 'x' | 'y' | 'z';
export type Shape3 = readonly [number, number, number];

export function fieldIndex(shape: Shape3, x: number, y: number, z: number): number {
  return x + shape[0] * (y + shape[1] * z);
}

/**
 * Width and height of a slice normal to `axis`, as (u, v) where
 * x-slices span (y, z), y-slices span (x, z) and z-slices span (x, y).
 */
export function sliceDims(shape: Shape3, axis: Axis): [number, number] {
  const [nx, ny, nz] = shape;
  if (axis === 'x') return [ny, nz];
  if (axis === 'y') return [nx, nz];
  return [nx, ny];
}

/**
 * Copy the plane `axis = index` of `field` into `out` (row-major, u fastest:
 * `out[u + w * v]`). `index` is clamped to the valid range. `out` must hold at
 * least w*h values; nothing is allocated. Returns `out`.
 */
export function extractSlice(
  field: ArrayLike<number>,
  shape: Shape3,
  axis: Axis,
  index: number,
  out: Float32Array,
): Float32Array {
  const [nx, ny, nz] = shape;
  if (field.length < nx * ny * nz) {
    throw new RangeError(`field has ${field.length} values, shape needs ${nx * ny * nz}`);
  }
  const [w, h] = sliceDims(shape, axis);
  if (out.length < w * h) throw new RangeError(`out has ${out.length} values, slice needs ${w * h}`);
  const n = axis === 'x' ? nx : axis === 'y' ? ny : nz;
  const k = Math.min(n - 1, Math.max(0, Math.round(index)));
  const sxy = nx * ny;
  if (axis === 'z') {
    const base = k * sxy;
    for (let i = 0; i < sxy; i++) out[i] = field[base + i];
  } else if (axis === 'y') {
    for (let z = 0; z < nz; z++) {
      const src = nx * k + sxy * z;
      const dst = nx * z;
      for (let x = 0; x < nx; x++) out[dst + x] = field[src + x];
    }
  } else {
    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) out[y + ny * z] = field[k + nx * y + sxy * z];
    }
  }
  return out;
}
