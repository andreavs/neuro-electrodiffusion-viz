import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

/** World units per µm: the scene is in millimetres, centred on the tissue box. */
export const WORLD_PER_UM = 1e-3;
/** Thin dendrites are widened so they stay visible in a millimetre-sized box. */
const MIN_RADIUS_UM = 3;
const RADIUS_SCALE = 1.5;
const SIDES = 8;
export type Vec3 = [number, number, number];

/** Grid coordinates (µm, box spans [0, size]) to world coordinates. */
export function gridToWorld(p: ArrayLike<number>, size: Vec3): Vec3 {
  return [0, 1, 2].map((a) => (p[a]! - size[a]! / 2) * WORLD_PER_UM) as Vec3;
}
/** World coordinates to grid coordinates (µm). */
export function worldToGrid(p: ArrayLike<number>, size: Vec3): Vec3 {
  return [0, 1, 2].map((a) => p[a]! / WORLD_PER_UM + size[a]! / 2) as Vec3;
}
export interface NeuronMesh {
  geometry: THREE.BufferGeometry;
  /** Compartment index of every vertex, to spread per-compartment values onto the mesh. */
  vertexComp: Uint16Array;
}

/**
 * One cylinder per compartment from `compartments` (7 floats each: start xyz, end xyz in
 * grid µm, diameter µm), merged into a single geometry. The soma (index 0) is capped.
 */
export function buildNeuron(compartments: Float32Array, size: Vec3): NeuronMesh {
  const parts: THREE.BufferGeometry[] = [];
  const owner: number[] = [];
  const up = new THREE.Vector3(0, 1, 0);
  for (let c = 0; 7 * c + 6 < compartments.length; c++) {
    const a = new THREE.Vector3(...gridToWorld(compartments.subarray(7 * c, 7 * c + 3), size));
    const b = new THREE.Vector3(...gridToWorld(compartments.subarray(7 * c + 3, 7 * c + 6), size));
    const r = Math.max(MIN_RADIUS_UM, (RADIUS_SCALE * compartments[7 * c + 6]!) / 2) * WORLD_PER_UM;
    const dir = b.clone().sub(a);
    const g = new THREE.CylinderGeometry(r, r, dir.length(), SIDES, 1, c !== 0);
    g.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(up, dir.normalize()));
    g.translate((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2);
    g.deleteAttribute('uv');
    for (let i = 0; i < g.getAttribute('position').count; i++) owner.push(c);
    parts.push(g);
  }
  const geometry = mergeGeometries(parts);
  parts.forEach((g) => g.dispose());
  if (!geometry) throw new Error('could not build the neuron mesh');
  return { geometry, vertexComp: Uint16Array.from(owner) };
}
/** Writes each compartment's value to its vertices. */
export function spreadValues(values: ArrayLike<number>, vertexComp: Uint16Array, out: Float32Array): Float32Array {
  for (let i = 0; i < vertexComp.length; i++) out[i] = values[vertexComp[i]!]!;
  return out;
}
/** Firing rate (Hz) from (sim time ms, spike count) samples over the last `windowMs`. */
export class RateMeter {
  private samples: [number, number][] = [];
  constructor(private readonly windowMs = 1000) {}
  push(t: number, spikes: number): number {
    const s = this.samples;
    if (s.length && t < s[s.length - 1]![0]) s.length = 0; // reset
    s.push([t, spikes]);
    while (s.length > 2 && t - s[1]![0] >= this.windowMs) s.shift();
    const [t0, n0] = s[0]!;
    return t > t0 ? ((spikes - n0) * 1000) / (t - t0) : 0;
  }
}

/**
 * The `1 − q` and `q` quantiles of all values in `arrays`, as a two-element range for the
 * colour bar's auto mode. `scratch` is reused between calls.
 */
export function clipRange(arrays: readonly Float32Array[], q: number, scratch: number[]): [number, number] {
  scratch.length = 0;
  for (const a of arrays) for (let i = 0; i < a.length; i++) if (Number.isFinite(a[i]!)) scratch.push(a[i]!);
  if (scratch.length === 0) return [0, 1];
  scratch.sort((x, y) => x - y);
  const at = (f: number) => scratch[Math.min(scratch.length - 1, Math.max(0, Math.round(f * (scratch.length - 1))))]!;
  return [at(1 - q), at(q)];
}
