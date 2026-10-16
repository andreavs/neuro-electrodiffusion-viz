import * as THREE from 'three';
import type { Colormap } from './colormap';
import { COLORMAP_GLSL, colormapUniforms, type ColormapUniforms, type Colormapped } from './gpu';
import type { Shape3 } from './field';
import type { Range } from './range';

export interface VolumeOptions {
  shape: Shape3;
  /** World-space size of the whole field box (centred on the origin). */
  extent: THREE.Vector3Like;
  colormap: Colormap;
  range?: Range;
  /**
   * Opacity from value: `sequential` fades in from the low end of the range,
   * `diverging` from the centre outwards (both signs glow). Default from the colormap.
   */
  mode?: 'sequential' | 'diverging';
  /** Ray-march samples across the box diagonal. Default 160. */
  steps?: number;
  /** Trilinear filtering; requires `SceneHandle.floatLinear`. Default true. */
  smooth?: boolean;
}

/** Look settings, changeable at any time. */
export interface VolumeLook {
  /** Optical density at full strength (per box diagonal). Default 6. */
  density: number;
  /** Fraction of the range (from its low end, or from the centre when diverging) that stays clear. Default 0.15. */
  threshold: number;
  /** HDR brightness boost of the strongest values, so they bloom. Default 0.8. */
  emission: number;
}

/**
 * A 3-D scalar field drawn as glowing fog by GPU ray marching: each pixel
 * integrates colour and opacity (from the colormap and a value ramp) through
 * an R32F 3-D texture that is overwritten in place on every `update()`.
 *
 * The volume does not write depth and is drawn before opaque objects
 * (`renderOrder` -1), so surfaces inside the box are painted over it.
 */
export class VolumeField implements Colormapped {
  readonly mesh: THREE.Mesh<THREE.BoxGeometry, THREE.ShaderMaterial>;
  readonly uniforms: ColormapUniforms & {
    uData: THREE.IUniform<THREE.Data3DTexture>;
    uDensity: THREE.IUniform<number>;
    uThreshold: THREE.IUniform<number>;
    uEmission: THREE.IUniform<number>;
    uDiverging: THREE.IUniform<boolean>;
    uSteps: THREE.IUniform<number>;
  };
  private readonly texture: THREE.Data3DTexture;
  private readonly values: Float32Array;

  constructor(opts: VolumeOptions) {
    const [nx, ny, nz] = opts.shape;
    this.values = new Float32Array(nx * ny * nz);
    // Data3DTexture layout is x fastest, then y, then z: the solvers' layout.
    this.texture = new THREE.Data3DTexture(this.values, nx, ny, nz);
    this.texture.format = THREE.RedFormat;
    this.texture.type = THREE.FloatType;
    const filter = opts.smooth === false ? THREE.NearestFilter : THREE.LinearFilter;
    this.texture.magFilter = filter;
    this.texture.minFilter = filter;
    this.texture.generateMipmaps = false;
    this.texture.unpackAlignment = 1;
    this.texture.needsUpdate = true;

    this.uniforms = {
      ...colormapUniforms(opts.colormap, opts.range),
      uData: { value: this.texture },
      uDensity: { value: 6 },
      uThreshold: { value: 0.15 },
      uEmission: { value: 0.8 },
      uDiverging: { value: (opts.mode ?? (opts.colormap.diverging ? 'diverging' : 'sequential')) === 'diverging' },
      uSteps: { value: opts.steps ?? 160 },
    };
    const material = new THREE.ShaderMaterial({
      uniforms: this.uniforms,
      side: THREE.BackSide,
      depthTest: false,
      depthWrite: false,
      // Premultiplied "over" blending; set by hand so it applies in the opaque pass.
      blending: THREE.CustomBlending,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneMinusSrcAlphaFactor,
      blendSrcAlpha: THREE.OneFactor,
      blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
      vertexShader: /* glsl */ `
        varying vec3 vOrigin;
        varying vec3 vDir;
        void main() {
          // Ray in box-local coordinates, from the camera to this back-face vertex.
          vOrigin = (inverse(modelMatrix) * vec4(cameraPosition, 1.0)).xyz;
          vDir = position - vOrigin;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }`,
      fragmentShader: /* glsl */ `
        precision highp sampler3D;
        ${COLORMAP_GLSL}
        uniform sampler3D uData;
        uniform vec3 uSize;
        uniform float uDensity;
        uniform float uThreshold;
        uniform float uEmission;
        uniform bool uDiverging;
        uniform float uSteps;
        varying vec3 vOrigin;
        varying vec3 vDir;

        vec2 hitBox(vec3 o, vec3 d) {
          vec3 inv = 1.0 / d;
          vec3 t0 = (-0.5 * uSize - o) * inv;
          vec3 t1 = (0.5 * uSize - o) * inv;
          vec3 lo = min(t0, t1);
          vec3 hi = max(t0, t1);
          return vec2(max(max(lo.x, lo.y), lo.z), min(min(hi.x, hi.y), hi.z));
        }

        void main() {
          vec3 dir = normalize(vDir);
          vec2 t = hitBox(vOrigin, dir);
          t.x = max(t.x, 0.0);
          if (t.x >= t.y) discard;
          float dt = length(uSize) / uSteps;
          // Per-pixel jitter hides the banding of a fixed step.
          float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
          float span = uRange.y - uRange.x;
          vec4 acc = vec4(0.0);
          for (float s = t.x + jitter * dt; s < t.y; s += dt) {
            vec3 p = (vOrigin + dir * s) / uSize + 0.5;
            float v = texture(uData, p).r;
            if (v != v) continue;
            float u = span > 0.0 ? clamp((v - uRange.x) / span, 0.0, 1.0) : 0.5;
            float w = uDiverging ? abs(2.0 * u - 1.0) : u;
            float ramp = smoothstep(uThreshold, 1.0, w);
            if (ramp <= 0.0) continue;
            float alpha = 1.0 - exp(-uDensity * ramp * ramp * dt / length(uSize));
            vec3 c = colormap(v) * (1.0 + uEmission * ramp * ramp);
            acc.rgb += (1.0 - acc.a) * alpha * c;
            acc.a += (1.0 - acc.a) * alpha;
            if (acc.a > 0.97) break;
          }
          // Encode the un-premultiplied colour, then premultiply again for the blend.
          gl_FragColor = vec4(acc.a > 0.0 ? acc.rgb / acc.a : vec3(0.0), 1.0);
          #include <colorspace_fragment>
          gl_FragColor = vec4(gl_FragColor.rgb * acc.a, acc.a);
        }`,
    });
    material.uniforms.uSize = { value: new THREE.Vector3().copy(opts.extent) };
    this.mesh = new THREE.Mesh(new THREE.BoxGeometry(opts.extent.x, opts.extent.y, opts.extent.z), material);
    this.mesh.renderOrder = -1;
  }

  /** Change the look; omitted settings keep their values. */
  setLook(look: Partial<VolumeLook>): void {
    if (look.density !== undefined) this.uniforms.uDensity.value = look.density;
    if (look.threshold !== undefined) this.uniforms.uThreshold.value = look.threshold;
    if (look.emission !== undefined) this.uniforms.uEmission.value = look.emission;
  }

  setMode(mode: 'sequential' | 'diverging'): void {
    this.uniforms.uDiverging.value = mode === 'diverging';
  }

  /** Upload a whole field (flat, x fastest, at least nx·ny·nz values). */
  update(field: ArrayLike<number>): void {
    if (field.length < this.values.length) {
      throw new RangeError(`field has ${field.length} values, shape needs ${this.values.length}`);
    }
    if (field instanceof Float32Array) this.values.set(field.subarray(0, this.values.length));
    else for (let i = 0; i < this.values.length; i++) this.values[i] = field[i]!;
    this.texture.needsUpdate = true;
  }

  dispose(): void {
    this.mesh.removeFromParent();
    this.mesh.geometry.dispose();
    this.texture.dispose();
    this.uniforms.uCmap.value.dispose();
    this.mesh.material.dispose();
  }
}
