import * as THREE from 'three';
import type { Colormap } from './colormap';
import { COLORMAP_GLSL, colormapUniforms, type ColormapUniforms, type Colormapped } from './gpu';
import { extractSlice, sliceDims, type Axis, type Shape3 } from './field';
import type { Range } from './range';

export interface SliceOptions {
  shape: Shape3;
  /** World-space size of the whole field box (centred on the origin). */
  extent: THREE.Vector3Like;
  axis: Axis;
  index?: number;
  colormap: Colormap;
  range?: Range;
  /** Bilinear filtering; requires `SceneHandle.floatLinear`. Default false (voxels). */
  smooth?: boolean;
  opacity?: number;
}

/**
 * One axis-aligned slice of a 3-D field shown as a colormapped plane.
 *
 * The slice values live in a single R32F `DataTexture` that is overwritten in
 * place on every `update()`; no buffers are allocated per frame.
 */
export class SlicePlane implements Colormapped {
  readonly mesh: THREE.Mesh<THREE.PlaneGeometry, THREE.ShaderMaterial>;
  readonly uniforms: ColormapUniforms & { uData: THREE.IUniform<THREE.DataTexture>; uOpacity: THREE.IUniform<number> };
  /** The slice values (row-major, u fastest), shared with the texture. */
  readonly values: Float32Array;
  readonly axis: Axis;
  private readonly shape: Shape3;
  private readonly extent: THREE.Vector3;
  private readonly texture: THREE.DataTexture;
  private _index = 0;

  constructor(opts: SliceOptions) {
    this.shape = opts.shape;
    this.axis = opts.axis;
    this.extent = new THREE.Vector3().copy(opts.extent);
    const [w, h] = sliceDims(opts.shape, opts.axis);
    this.values = new Float32Array(w * h);
    this.texture = new THREE.DataTexture(this.values, w, h, THREE.RedFormat, THREE.FloatType);
    const filter = opts.smooth ? THREE.LinearFilter : THREE.NearestFilter;
    this.texture.magFilter = filter;
    this.texture.minFilter = filter;
    this.texture.generateMipmaps = false;
    this.texture.needsUpdate = true;

    this.uniforms = {
      ...colormapUniforms(opts.colormap, opts.range),
      uData: { value: this.texture },
      uOpacity: { value: opts.opacity ?? 1 },
    };
    const material = new THREE.ShaderMaterial({
      uniforms: this.uniforms,
      side: THREE.DoubleSide,
      transparent: (opts.opacity ?? 1) < 1,
      // Let crossing translucent slices show through each other.
      depthWrite: (opts.opacity ?? 1) >= 1,
      vertexShader: /* glsl */ `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }`,
      fragmentShader: /* glsl */ `
        ${COLORMAP_GLSL}
        uniform sampler2D uData;
        uniform float uOpacity;
        varying vec2 vUv;
        void main() {
          float v = texture2D(uData, vUv).r;
          gl_FragColor = vec4(colormap(v), uOpacity);
          #include <colorspace_fragment>
        }`,
    });

    const e = this.extent;
    const [pw, ph] = opts.axis === 'x' ? [e.y, e.z] : opts.axis === 'y' ? [e.x, e.z] : [e.x, e.y];
    this.mesh = new THREE.Mesh(new THREE.PlaneGeometry(pw, ph), material);
    // Map plane-local (u, v, normal) to world axes; u/v match `extractSlice`.
    const X = new THREE.Vector3(1, 0, 0);
    const Y = new THREE.Vector3(0, 1, 0);
    const Z = new THREE.Vector3(0, 0, 1);
    const basis =
      opts.axis === 'x'
        ? new THREE.Matrix4().makeBasis(Y, Z, X)
        : opts.axis === 'y'
          ? new THREE.Matrix4().makeBasis(X, Z, Y.clone().negate())
          : new THREE.Matrix4().makeBasis(X, Y, Z);
    this.mesh.quaternion.setFromRotationMatrix(basis);
    this.index = opts.index ?? Math.floor(this.axisLength() / 2);
  }

  get index(): number {
    return this._index;
  }

  /** Move the plane to voxel layer `i` (clamped). Takes effect on the next `update()`. */
  set index(i: number) {
    const n = this.axisLength();
    this._index = Math.min(n - 1, Math.max(0, Math.round(i)));
    const len = this.axis === 'x' ? this.extent.x : this.axis === 'y' ? this.extent.y : this.extent.z;
    const pos = ((this._index + 0.5) / n - 0.5) * len;
    this.mesh.position.set(this.axis === 'x' ? pos : 0, this.axis === 'y' ? pos : 0, this.axis === 'z' ? pos : 0);
  }

  /** Copy the current layer of `field` into the texture. */
  update(field: ArrayLike<number>): void {
    extractSlice(field, this.shape, this.axis, this._index, this.values);
    this.texture.needsUpdate = true;
  }

  dispose(): void {
    this.mesh.removeFromParent();
    this.mesh.geometry.dispose();
    this.texture.dispose();
    this.uniforms.uCmap.value.dispose();
    this.mesh.material.dispose();
  }

  private axisLength(): number {
    return this.axis === 'x' ? this.shape[0] : this.axis === 'y' ? this.shape[1] : this.shape[2];
  }
}
