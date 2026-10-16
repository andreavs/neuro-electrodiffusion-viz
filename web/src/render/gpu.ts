import * as THREE from 'three';
import { LUT_SIZE, type Colormap } from './colormap';
import type { Range } from './range';

/**
 * 256×1 RGBA8 texture of a colormap, linearly filtered. Caller disposes.
 *
 * The bytes are sRGB and are sampled raw (no colour-space decode): our shader
 * materials write them straight to the sRGB framebuffer.
 */
export function createColormapTexture(cm: Colormap): THREE.DataTexture {
  const tex = new THREE.DataTexture(cm.lut, LUT_SIZE, 1, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}

export interface ColormapUniforms {
  [name: string]: THREE.IUniform;
  uCmap: THREE.IUniform<THREE.Texture>;
  uRange: THREE.IUniform<THREE.Vector2>;
}

export function colormapUniforms(cm: Colormap, range: Range = { min: 0, max: 1 }): ColormapUniforms {
  return {
    uCmap: { value: createColormapTexture(cm) },
    uRange: { value: new THREE.Vector2(range.min, range.max) },
  };
}

/** GLSL: `vec3 colormap(float v)` using uCmap/uRange; samples texel centres. */
export const COLORMAP_GLSL = /* glsl */ `
uniform sampler2D uCmap;
uniform vec2 uRange;
vec3 colormap(float v) {
  if (v != v) return vec3(0.5); // NaN: neutral grey "no data"
  float span = uRange.y - uRange.x;
  float t = span > 0.0 ? clamp((v - uRange.x) / span, 0.0, 1.0) : 0.5;
  return texture2D(uCmap, vec2(t * (255.0 / 256.0) + 0.5 / 256.0, 0.5)).rgb;
}`;

/** Common surface of the scalar-field helpers. */
export interface Colormapped {
  readonly uniforms: ColormapUniforms;
}

export function setRange(target: Colormapped, range: Range): void {
  target.uniforms.uRange.value.set(range.min, range.max);
}

export function setColormap(target: Colormapped, cm: Colormap): void {
  const tex = target.uniforms.uCmap.value as THREE.DataTexture;
  tex.image.data = cm.lut;
  tex.needsUpdate = true;
}

/** Dispose a material and every texture it references (maps or uniforms). */
export function disposeMaterial(material: THREE.Material): void {
  const seen = new Set<THREE.Texture>();
  const visit = (v: unknown) => {
    if (v instanceof THREE.Texture && !seen.has(v)) {
      seen.add(v);
      v.dispose();
    }
  };
  for (const v of Object.values(material)) visit(v);
  if (material instanceof THREE.ShaderMaterial) {
    for (const u of Object.values(material.uniforms)) visit(u.value);
  }
  material.dispose();
}

/** Dispose geometries, materials and textures of `root` and its descendants. */
export function disposeObject(root: THREE.Object3D): void {
  root.traverse((obj) => {
    const o = obj as THREE.Object3D & { geometry?: THREE.BufferGeometry; material?: THREE.Material | THREE.Material[] };
    o.geometry?.dispose();
    if (Array.isArray(o.material)) o.material.forEach(disposeMaterial);
    else if (o.material) disposeMaterial(o.material);
  });
}
