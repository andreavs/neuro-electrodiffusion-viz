import * as THREE from 'three';
import type { Colormap } from './colormap';
import { COLORMAP_GLSL, colormapUniforms, type ColormapUniforms, type Colormapped } from './gpu';
import type { Shape3 } from './field';
import type { Range } from './range';

export interface FlowOptions {
  shape: Shape3;
  /** World-space size of the whole field box (centred on the origin). */
  extent: THREE.Vector3Like;
  colormap: Colormap;
  range?: Range;
  /** Number of tracers. Default 1500. */
  count?: number;
  /** Points kept per trail. Default 12. */
  trail?: number;
  /** Speed of the fastest tracers, in grid cells per second. Default 14. */
  speed?: number;
}

/** Fraction of the gradient magnitudes below the reference gradient (stronger ones move at full speed). */
const GRAD_QUANTILE = 0.9;

/**
 * Glowing tracers that stream down the gradient of a 3-D scalar field (along
 * −∇φ, the current, for a potential; along −∇c, the diffusive flux, for a
 * concentration), drawn as additive fading trails coloured by the field.
 *
 * Tracers are born where the gradient is strong, move at a speed that grows
 * with it, and are reborn when they stall, leave the box or grow old.
 * Advection runs on the CPU in `step()`; one draw call renders every trail.
 */
export class FlowTracers implements Colormapped {
  readonly lines: THREE.LineSegments<THREE.BufferGeometry, THREE.ShaderMaterial>;
  readonly uniforms: ColormapUniforms;
  private readonly shape: Shape3;
  private readonly extent: THREE.Vector3;
  private readonly count: number;
  private readonly trail: number;
  private readonly speed: number;
  /** Trail points in grid coordinates (cell centres at integers), newest first. */
  private readonly hist: Float32Array;
  private readonly age: Float32Array;
  private readonly life: Float32Array;
  private readonly position: THREE.BufferAttribute;
  private readonly value: THREE.BufferAttribute;
  private readonly alpha: THREE.BufferAttribute;
  private field: ArrayLike<number> | null = null;
  /** Squared cell spacing per axis relative to the smallest, so tracers follow −∇f in world space. */
  private readonly spacing2: [number, number, number];
  private readonly g: [number, number, number] = [0, 0, 0];
  private readonly mags = new Float32Array(4000);
  private gradRef = 0;
  private frames = 0;
  private seed = 0x2545f491;

  constructor(opts: FlowOptions) {
    this.shape = opts.shape;
    this.extent = new THREE.Vector3().copy(opts.extent);
    this.count = opts.count ?? 1500;
    this.trail = Math.max(2, opts.trail ?? 12);
    this.speed = opts.speed ?? 14;
    const h = [this.extent.x / opts.shape[0], this.extent.y / opts.shape[1], this.extent.z / opts.shape[2]];
    const hMin = Math.min(...h);
    this.spacing2 = h.map((x) => (x / hMin) ** 2) as [number, number, number];
    this.hist = new Float32Array(this.count * this.trail * 3);
    this.age = new Float32Array(this.count);
    this.life = new Float32Array(this.count);
    for (let i = 0; i < this.count; i++) {
      this.respawn(i);
      this.age[i] = this.rand() * this.life[i]!; // stagger births
    }

    const segs = this.count * (this.trail - 1);
    const g = new THREE.BufferGeometry();
    this.position = new THREE.BufferAttribute(new Float32Array(segs * 6), 3).setUsage(THREE.DynamicDrawUsage);
    this.value = new THREE.BufferAttribute(new Float32Array(segs * 2), 1).setUsage(THREE.DynamicDrawUsage);
    this.alpha = new THREE.BufferAttribute(new Float32Array(segs * 2), 1).setUsage(THREE.DynamicDrawUsage);
    g.setAttribute('position', this.position);
    g.setAttribute('value', this.value);
    g.setAttribute('alpha', this.alpha);
    this.uniforms = colormapUniforms(opts.colormap, opts.range);
    const material = new THREE.ShaderMaterial({
      uniforms: this.uniforms,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      vertexShader: /* glsl */ `
        attribute float value;
        attribute float alpha;
        varying float vValue;
        varying float vAlpha;
        void main() {
          vValue = value;
          vAlpha = alpha;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }`,
      fragmentShader: /* glsl */ `
        ${COLORMAP_GLSL}
        varying float vValue;
        varying float vAlpha;
        void main() {
          // Tracers are bright even where the field is near rest (black in diverging glow maps).
          vec3 c = max(colormap(vValue), vec3(0.05, 0.08, 0.14));
          gl_FragColor = vec4(c * vAlpha * 0.9, 1.0);
          #include <colorspace_fragment>
        }`,
    });
    this.lines = new THREE.LineSegments(g, material);
    this.lines.frustumCulled = false;
  }

  /** Forget the gradient reference, e.g. when the followed field or simulation changes. */
  reset(): void {
    this.gradRef = 0;
    this.frames = 0;
  }

  /** The field to follow (flat, x fastest). Kept by reference; call again when it changes. */
  setField(field: ArrayLike<number>): void {
    this.field = field;
    // The reference gradient (a high quantile, so a few huge source cells don't freeze
    // everything else) changes slowly: refresh it every few fields.
    if (this.frames++ % 10 === 0 || this.gradRef === 0) this.gradRef = this.gradientQuantile(GRAD_QUANTILE);
  }

  /** Advance the tracers by `dt` seconds of wall time and refresh the trails. */
  step(dt: number): void {
    const f = this.field;
    if (!f || !(this.gradRef > 0)) return;
    const [nx, ny, nz] = this.shape;
    const { hist, trail, count } = this;
    const g = this.g;
    const [sx, sy, sz] = this.spacing2;
    const h = Math.min(dt, 0.05);
    for (let i = 0; i < count; i++) {
      const b = i * trail * 3;
      // Shift the trail back by one point.
      hist.copyWithin(b + 3, b, b + (trail - 1) * 3);
      const strength = this.gradient(f, hist[b]!, hist[b + 1]!, hist[b + 2]!, g);
      const s = Math.min(1, strength / this.gradRef);
      const k = (this.speed * s * h) / (strength || 1);
      // Grid-unit velocity of −∇f in world space: per-cell gradient over spacing².
      hist[b] = hist[b]! - (g[0] / sx) * k;
      hist[b + 1] = hist[b + 1]! - (g[1] / sy) * k;
      hist[b + 2] = hist[b + 2]! - (g[2] / sz) * k;
      this.age[i] = this.age[i]! + h;
      const x = hist[b]!;
      const y = hist[b + 1]!;
      const z = hist[b + 2]!;
      const out = x < -0.5 || y < -0.5 || z < -0.5 || x > nx - 0.5 || y > ny - 0.5 || z > nz - 0.5;
      if (out || this.age[i]! > this.life[i]! || s < 0.02) this.respawn(i);
    }
    this.writeGeometry(f);
  }

  dispose(): void {
    this.lines.removeFromParent();
    this.lines.geometry.dispose();
    this.uniforms.uCmap.value.dispose();
    this.lines.material.dispose();
  }

  private writeGeometry(f: ArrayLike<number>): void {
    const [nx, ny, nz] = this.shape;
    const e = this.extent;
    const pos = this.position.array as Float32Array;
    const val = this.value.array as Float32Array;
    const alp = this.alpha.array as Float32Array;
    const { hist, trail, count } = this;
    let v = 0;
    for (let i = 0; i < count; i++) {
      const b = i * trail * 3;
      // Fade in after birth and out before death, so respawns don't pop.
      const t = this.age[i]! / this.life[i]!;
      const life = Math.min(1, t * 6, (1 - t) * 4);
      const value = this.sample(f, hist[b]!, hist[b + 1]!, hist[b + 2]!);
      for (let k = 0; k < trail - 1; k++) {
        for (let end = 0; end < 2; end++, v++) {
          const j = b + (k + end) * 3;
          pos[3 * v] = ((hist[j]! + 0.5) / nx - 0.5) * e.x;
          pos[3 * v + 1] = ((hist[j + 1]! + 0.5) / ny - 0.5) * e.y;
          pos[3 * v + 2] = ((hist[j + 2]! + 0.5) / nz - 0.5) * e.z;
          val[v] = value;
          alp[v] = life * (1 - (k + end) / (trail - 1));
        }
      }
    }
    this.position.needsUpdate = true;
    this.value.needsUpdate = true;
    this.alpha.needsUpdate = true;
  }

  /** Rebirth tracer `i` at a random cell, preferring cells with a strong gradient. */
  private respawn(i: number): void {
    const [nx, ny, nz] = this.shape;
    const g = this.g;
    let x = 0;
    let y = 0;
    let z = 0;
    for (let tries = 0; tries < 12; tries++) {
      x = this.rand() * nx - 0.5;
      y = this.rand() * ny - 0.5;
      z = this.rand() * nz - 0.5;
      if (!this.field || !(this.gradRef > 0)) break;
      const s = Math.min(1, this.gradient(this.field, x, y, z, g) / this.gradRef);
      if (this.rand() < Math.sqrt(s)) break;
    }
    const b = i * this.trail * 3;
    for (let k = 0; k < this.trail; k++) {
      this.hist[b + 3 * k] = x;
      this.hist[b + 3 * k + 1] = y;
      this.hist[b + 3 * k + 2] = z;
    }
    this.age[i] = 0;
    this.life[i] = 1.5 + 2.5 * this.rand();
  }

  /** Central-difference gradient (per cell) at the nearest cell; returns its magnitude. */
  private gradient(f: ArrayLike<number>, x: number, y: number, z: number, out: [number, number, number]): number {
    const [nx, ny, nz] = this.shape;
    const i = Math.min(nx - 1, Math.max(0, Math.round(x)));
    const j = Math.min(ny - 1, Math.max(0, Math.round(y)));
    const k = Math.min(nz - 1, Math.max(0, Math.round(z)));
    const sy = nx;
    const sz = nx * ny;
    const c = i + sy * j + sz * k;
    const i0 = Math.max(0, i - 1);
    const i1 = Math.min(nx - 1, i + 1);
    const j0 = Math.max(0, j - 1);
    const j1 = Math.min(ny - 1, j + 1);
    const k0 = Math.max(0, k - 1);
    const k1 = Math.min(nz - 1, k + 1);
    out[0] = (f[c + i1 - i]! - f[c + i0 - i]!) / Math.max(1, i1 - i0);
    out[1] = (f[c + sy * (j1 - j)]! - f[c + sy * (j0 - j)]!) / Math.max(1, j1 - j0);
    out[2] = (f[c + sz * (k1 - k)]! - f[c + sz * (k0 - k)]!) / Math.max(1, k1 - k0);
    const m = Math.hypot(out[0], out[1], out[2]);
    return Number.isFinite(m) ? m : 0;
  }

  private sample(f: ArrayLike<number>, x: number, y: number, z: number): number {
    const [nx, ny, nz] = this.shape;
    const i = Math.min(nx - 1, Math.max(0, Math.round(x)));
    const j = Math.min(ny - 1, Math.max(0, Math.round(y)));
    const k = Math.min(nz - 1, Math.max(0, Math.round(z)));
    return f[i + nx * (j + ny * k)]!;
  }

  /** The `q` quantile of the gradient magnitude over ~4000 random cells. */
  private gradientQuantile(q: number): number {
    const f = this.field;
    if (!f) return 0;
    const [nx, ny, nz] = this.shape;
    const mags = this.mags;
    const n = mags.length;
    const g = this.g;
    for (let s = 0; s < n; s++) {
      mags[s] = this.gradient(f, this.rand() * nx - 0.5, this.rand() * ny - 0.5, this.rand() * nz - 0.5, g);
    }
    mags.sort();
    // Quantile of the non-zero gradients: most of the tissue can be exactly at rest.
    let first = 0;
    while (first < n && !(mags[first]! > 0)) first++;
    if (first === n) return 0;
    return mags[first + Math.floor(q * (n - 1 - first))]!;
  }

  /** xorshift32 in [0, 1). */
  private rand(): number {
    let s = this.seed;
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    this.seed = s;
    return (s >>> 0) / 4294967296;
  }
}
