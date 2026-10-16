import * as THREE from 'three';
import type { Colormap } from './colormap';
import { COLORMAP_GLSL, colormapUniforms, type ColormapUniforms, type Colormapped } from './gpu';
import type { Range } from './range';

/**
 * Colormapped scalar values on a mesh surface, one value per vertex, with
 * simple headlight shading, or (`setGlow`) an emissive look with a bright
 * Fresnel rim for bloom. `update()` overwrites the vertex attribute in place.
 */
export class SurfaceField implements Colormapped {
  readonly mesh: THREE.Mesh<THREE.BufferGeometry, THREE.ShaderMaterial>;
  readonly uniforms: ColormapUniforms & { uGlow: THREE.IUniform<number> };
  private readonly attr: THREE.BufferAttribute;

  /** `geometry` is owned by the SurfaceField afterwards and freed by `dispose()`. */
  constructor(geometry: THREE.BufferGeometry, colormap: Colormap, range?: Range) {
    if (!geometry.getAttribute('normal')) geometry.computeVertexNormals();
    const n = geometry.getAttribute('position').count;
    this.attr = new THREE.BufferAttribute(new Float32Array(n), 1);
    this.attr.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute('value', this.attr);
    this.uniforms = { ...colormapUniforms(colormap, range), uGlow: { value: 0 } };
    const material = new THREE.ShaderMaterial({
      uniforms: this.uniforms,
      side: THREE.DoubleSide,
      vertexShader: /* glsl */ `
        attribute float value;
        varying float vValue;
        varying vec3 vNormal;
        varying vec3 vView;
        void main() {
          vValue = value;
          vNormal = normalize(normalMatrix * normal);
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          vView = -mv.xyz;
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: /* glsl */ `
        ${COLORMAP_GLSL}
        varying float vValue;
        varying vec3 vNormal;
        varying vec3 vView;
        uniform float uGlow;
        void main() {
          vec3 base = colormap(vValue);
          float facing = abs(dot(normalize(vNormal), normalize(vView)));
          // Shading in linear light: the ^2.2 keeps the classic look's sRGB-space contrast.
          float shade = pow(0.45 + 0.55 * abs(normalize(vNormal).z), 2.2);
          // Glow: dimmer faces, a bright rim where the surface turns away, HDR so the rim blooms.
          float rim = pow(1.0 - facing, 2.5);
          vec3 glow = base * (0.35 + 0.65 * facing) + base * rim * 2.5 + vec3(0.6) * rim * rim;
          gl_FragColor = vec4(mix(base * shade, glow, uGlow), 1.0);
          #include <colorspace_fragment>
        }`,
    });
    this.mesh = new THREE.Mesh(geometry, material);
  }

  /** Switch between plain shading and the emissive glow look. */
  setGlow(on: boolean): void {
    this.uniforms.uGlow.value = on ? 1 : 0;
  }

  get vertexCount(): number {
    return this.attr.count;
  }

  /** Per-vertex values (length >= vertex count). */
  update(values: ArrayLike<number>): void {
    const dst = this.attr.array as Float32Array;
    if (values.length < dst.length) throw new RangeError(`need ${dst.length} values, got ${values.length}`);
    if (values instanceof Float32Array) dst.set(values.length === dst.length ? values : values.subarray(0, dst.length));
    else for (let i = 0; i < dst.length; i++) dst[i] = values[i];
    this.attr.needsUpdate = true;
  }

  dispose(): void {
    this.mesh.removeFromParent();
    this.mesh.geometry.dispose();
    this.uniforms.uCmap.value.dispose();
    this.mesh.material.dispose();
  }
}
