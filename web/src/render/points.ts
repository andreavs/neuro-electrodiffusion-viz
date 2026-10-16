import * as THREE from 'three';
import { viridis, type Colormap } from './colormap';
import { COLORMAP_GLSL, colormapUniforms, type ColormapUniforms, type Colormapped } from './gpu';
import type { Range } from './range';

export interface PointCloudOptions {
  /** Maximum number of points; buffers are allocated once at this size. */
  capacity: number;
  /** Point diameter in world units (perspective-attenuated). */
  size: number;
  /** Colour by a per-point value with this map; otherwise all points use `color`. */
  colormap?: Colormap;
  range?: Range;
  color?: THREE.ColorRepresentation;
}

/**
 * Up to `capacity` round point sprites (one GL draw call) from a flat xyz
 * `Float32Array`. `update()` copies into preallocated GPU buffers in place and
 * uploads only the used range.
 */
export class PointCloud implements Colormapped {
  readonly points: THREE.Points<THREE.BufferGeometry, THREE.ShaderMaterial>;
  readonly uniforms: ColormapUniforms & {
    uSize: THREE.IUniform<number>;
    uViewportHeight: THREE.IUniform<number>;
    uColor: THREE.IUniform<THREE.Color>;
    uUseMap: THREE.IUniform<boolean>;
  };
  readonly capacity: number;
  private readonly position: THREE.BufferAttribute;
  private readonly value: THREE.BufferAttribute;
  private _count = 0;

  constructor(opts: PointCloudOptions) {
    this.capacity = opts.capacity;
    const geometry = new THREE.BufferGeometry();
    this.position = new THREE.BufferAttribute(new Float32Array(opts.capacity * 3), 3);
    this.position.setUsage(THREE.DynamicDrawUsage);
    this.value = new THREE.BufferAttribute(new Float32Array(opts.capacity), 1);
    this.value.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute('position', this.position);
    geometry.setAttribute('value', this.value);
    geometry.setDrawRange(0, 0);

    this.uniforms = {
      ...colormapUniforms(opts.colormap ?? viridis, opts.range),
      uSize: { value: opts.size },
      uViewportHeight: { value: 1 },
      uColor: { value: new THREE.Color(opts.color ?? 0xffb000) },
      uUseMap: { value: opts.colormap !== undefined },
    };
    const material = new THREE.ShaderMaterial({
      uniforms: this.uniforms,
      vertexShader: /* glsl */ `
        attribute float value;
        uniform float uSize;
        uniform float uViewportHeight;
        varying float vValue;
        void main() {
          vValue = value;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_Position = projectionMatrix * mv;
          // projectionMatrix[1][1] = 1 / tan(fov / 2): world size → pixels.
          gl_PointSize = max(1.0, uSize * projectionMatrix[1][1] * 0.5 * uViewportHeight / -mv.z);
        }`,
      fragmentShader: /* glsl */ `
        ${COLORMAP_GLSL}
        uniform vec3 uColor;
        uniform bool uUseMap;
        varying float vValue;
        void main() {
          vec2 p = gl_PointCoord * 2.0 - 1.0;
          float r2 = dot(p, p);
          if (r2 > 1.0) discard;
          vec3 base = uUseMap ? colormap(vValue) : uColor;
          gl_FragColor = vec4(base * (1.0 - 0.35 * r2), 1.0);
        }`,
    });
    this.points = new THREE.Points(geometry, material);
    this.points.frustumCulled = false;
    const size = new THREE.Vector2();
    this.points.onBeforeRender = (renderer) => {
      this.uniforms.uViewportHeight.value = renderer.getDrawingBufferSize(size).y;
    };
  }

  get count(): number {
    return this._count;
  }

  /**
   * Show the first `count` points of `xyz` (defaults to `xyz.length / 3`,
   * clamped to capacity). `values`, if given, colours each point.
   */
  update(xyz: Float32Array, count = Math.floor(xyz.length / 3), values?: Float32Array): void {
    const n = Math.min(count, this.capacity, Math.floor(xyz.length / 3));
    const pos = this.position.array as Float32Array;
    pos.set(n * 3 === xyz.length ? xyz : xyz.subarray(0, n * 3));
    this.position.clearUpdateRanges();
    this.position.addUpdateRange(0, n * 3);
    this.position.needsUpdate = true;
    if (values) {
      const m = Math.min(n, values.length);
      (this.value.array as Float32Array).set(m === values.length ? values : values.subarray(0, m));
      this.value.clearUpdateRanges();
      this.value.addUpdateRange(0, m);
      this.value.needsUpdate = true;
    }
    this._count = n;
    this.points.geometry.setDrawRange(0, n);
  }

  dispose(): void {
    this.points.removeFromParent();
    this.points.geometry.dispose();
    this.uniforms.uCmap.value.dispose();
    this.points.material.dispose();
  }
}
