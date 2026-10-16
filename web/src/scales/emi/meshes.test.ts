import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { MESH_GEOMETRIES, type MeshMeta } from './meshes';
import { sliceAxis } from './params';

describe('MESH_GEOMETRIES', () => {
  it('offers exactly the stimulus sites of each shipped mesh', () => {
    for (const g of MESH_GEOMETRIES) {
      const url = new URL(`../../../public/meshes/${g.value}.json`, import.meta.url);
      const meta = JSON.parse(readFileSync(url, 'utf8')) as MeshMeta;
      expect(g.sites.map((s) => s.value).sort()).toEqual(meta.stimuli.map((s) => s.key).sort());
      expect(meta.probes.length).toBeGreaterThan(0);
    }
  });
});

describe('sliceAxis', () => {
  it('cuts the bundle along x and the upright cells along z', () => {
    expect([sliceAxis('along'), sliceAxis('across')]).toEqual([2, 0]);
    expect([sliceAxis('along', 'neuron'), sliceAxis('across', 'neuron')]).toEqual([1, 2]);
  });
});
