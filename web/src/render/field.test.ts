import { describe, expect, it } from 'vitest';
import { extractSlice, fieldIndex, sliceDims, type Shape3 } from './field';

// Encode coordinates in the value so every slice entry is checkable.
const shape: Shape3 = [4, 3, 5];
const code = (x: number, y: number, z: number) => 100 * z + 10 * y + x;
const field = new Float32Array(shape[0] * shape[1] * shape[2]);
for (let z = 0; z < shape[2]; z++)
  for (let y = 0; y < shape[1]; y++) for (let x = 0; x < shape[0]; x++) field[fieldIndex(shape, x, y, z)] = code(x, y, z);

describe('extractSlice', () => {
  it('z-slice spans (x, y), x fastest', () => {
    const [w, h] = sliceDims(shape, 'z');
    expect([w, h]).toEqual([4, 3]);
    const out = extractSlice(field, shape, 'z', 2, new Float32Array(w * h));
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) expect(out[x + w * y]).toBe(code(x, y, 2));
  });

  it('y-slice spans (x, z)', () => {
    const [w, h] = sliceDims(shape, 'y');
    expect([w, h]).toEqual([4, 5]);
    const out = extractSlice(field, shape, 'y', 1, new Float32Array(w * h));
    for (let z = 0; z < h; z++) for (let x = 0; x < w; x++) expect(out[x + w * z]).toBe(code(x, 1, z));
  });

  it('x-slice spans (y, z)', () => {
    const [w, h] = sliceDims(shape, 'x');
    expect([w, h]).toEqual([3, 5]);
    const out = extractSlice(field, shape, 'x', 3, new Float32Array(w * h));
    for (let z = 0; z < h; z++) for (let y = 0; y < w; y++) expect(out[y + w * z]).toBe(code(3, y, z));
  });

  it('clamps the index and writes into the given buffer', () => {
    const out = new Float32Array(12);
    expect(extractSlice(field, shape, 'z', 99, out)).toBe(out);
    expect(out[0]).toBe(code(0, 0, 4));
    extractSlice(field, shape, 'z', -3, out);
    expect(out[0]).toBe(code(0, 0, 0));
  });

  it('rejects undersized buffers', () => {
    expect(() => extractSlice(field, shape, 'z', 0, new Float32Array(11))).toThrow(RangeError);
    expect(() => extractSlice(field.subarray(1), shape, 'z', 0, new Float32Array(12))).toThrow(RangeError);
  });
});
