import * as THREE from 'three';
import type { Colormap } from './colormap';
import { COLORMAP_GLSL, colormapUniforms, type ColormapUniforms, type Colormapped } from './gpu';
import type { Range } from './range';

/**
 * Colormapped scalar values on a mesh surface, one value per vertex, with
 * simple headlight shading. `update()` overwrites the vertex attribute in place.
 */
export class SurfaceField implements Colormapped {
  readonly mesh: THREE.Mesh<THREE.BufferGeometry, THREE.ShaderMaterial>;
  readonly uniforms: ColormapUniforms;
  private readonly attr: THREE.BufferAttribute;

  /** `geometry` is owned by the SurfaceField afterwards and freed by `dispose()`. */
  constructor(geometry: THREE.BufferGeometry, colormap: Colormap, range?: Range) {
    if (!geometry.getAttribute('normal')) geometry.computeVertexNormals();
    const n = geometry.getAttribute('position').count;
    this.attr = new THREE.BufferAttribute(new Float32Array(n), 1);
    this.attr.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute('value', this.attr);
    this.uniforms = colormapUniforms(colormap, range);
    const material = new THREE.ShaderMaterial({
      uniforms: this.uniforms,
      side: THREE.DoubleSide,
      vertexShader: /* glsl */ `
        attribute float value;
        varying float vValue;
        varying vec3 vNormal;
        void main() {
          vValue = value;
          vNormal = normalize(normalMatrix * normal);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }`,
      fragmentShader: /* glsl */ `
        ${COLORMAP_GLSL}
        varying float vValue;
        varying vec3 vNormal;
        void main() {
          float shade = 0.45 + 0.55 * abs(normalize(vNormal).z);
          gl_FragColor = vec4(colormap(vValue) * shade, 1.0);
        }`,
    });
    this.mesh = new THREE.Mesh(geometry, material);
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
