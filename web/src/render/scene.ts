import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { disposeObject } from './gpu';

export interface SceneOptions {
  /** Upper bound on devicePixelRatio, to cap fill rate on HiDPI screens. Default 2. */
  maxPixelRatio?: number;
  /** Clear colour; `null` keeps the canvas transparent. Default transparent. */
  background?: THREE.ColorRepresentation | null;
  fov?: number;
  antialias?: boolean;
}

export interface BloomOptions {
  /** Glow intensity. Default 0.9. */
  strength?: number;
  /** Glow spread, 0 to 1. Default 0.5. */
  radius?: number;
  /** Linear brightness above which pixels glow. Default 0.6. */
  threshold?: number;
}

export interface SceneHandle {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;
  /** Whether linear filtering of float textures is supported. */
  readonly floatLinear: boolean;
  /** Point the camera at a box of the given size centred on the origin. */
  frameBox(size: THREE.Vector3Like): void;
  /**
   * Turn bloom post-processing on (rendering into an HDR, multisampled buffer)
   * or off (`null`, plain rendering straight to the canvas). Stays off on GPUs
   * that cannot render to float buffers.
   */
  setBloom(opts: BloomOptions | null): void;
  /** Clear colour; `null` makes the canvas transparent. */
  setBackground(color: THREE.ColorRepresentation | null): void;
  /** Render once (controls are updated first). */
  render(): void;
  /** Start a requestAnimationFrame loop calling `onFrame(dtMs)` before each render. */
  start(onFrame?: (dtMs: number, nowMs: number) => void): void;
  stop(): void;
  /** Stop the loop, free every GPU resource in the scene and release the WebGL context. */
  dispose(): void;
}

/**
 * WebGL2 scene with orbit controls, automatic resize (via ResizeObserver on the
 * canvas, which should be sized by CSS) and a capped device pixel ratio.
 */
export function createScene(canvas: HTMLCanvasElement, opts: SceneOptions = {}): SceneHandle {
  const maxPixelRatio = opts.maxPixelRatio ?? 2;
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: opts.antialias ?? true,
    alpha: opts.background == null,
    powerPreference: 'high-performance',
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, maxPixelRatio));
  if (opts.background != null) renderer.setClearColor(opts.background);
  else renderer.setClearColor(0x000000, 0);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(opts.fov ?? 40, 1, 0.01, 1000);
  camera.position.set(0, 0, 5);
  // Z-up is the toolkit convention. Set it before creating the controls:
  // OrbitControls caches its orbit frame from camera.up at construction.
  camera.up.set(0, 0, 1);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;

  const floatLinear = renderer.extensions.has('OES_texture_float_linear');

  let raf = 0;
  let last = 0;
  let disposed = false;
  let composer: EffectComposer | null = null;
  let bloom: UnrealBloomPass | null = null;

  const render = () => {
    controls.update();
    if (composer) composer.render();
    else renderer.render(scene, camera);
  };

  let composerRatio = 0;
  const sizeComposer = () => {
    if (!composer) return;
    // setPixelRatio reallocates every buffer itself; only call it when the ratio changes.
    const ratio = renderer.getPixelRatio();
    if (ratio !== composerRatio) composer.setPixelRatio(ratio);
    composerRatio = ratio;
    composer.setSize(Math.max(1, canvas.clientWidth), Math.max(1, canvas.clientHeight));
  };
  // Bloom renders into half-float targets; without float colour buffers it would draw nothing.
  const hdrTargets =
    renderer.extensions.has('EXT_color_buffer_float') || renderer.extensions.has('EXT_color_buffer_half_float');

  const resize = () => {
    const w = Math.max(1, canvas.clientWidth);
    const h = Math.max(1, canvas.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, maxPixelRatio));
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    sizeComposer();
    // setSize clears the canvas; redraw when no loop will do it.
    if (raf === 0 && !disposed) render();
  };
  resize();
  const observer = new ResizeObserver(resize);
  observer.observe(canvas);

  // Moving to a screen with another devicePixelRatio doesn't change the CSS
  // size, so watch the DPR separately (the media query is per-ratio).
  let dprQuery: MediaQueryList | null = null;
  const onDprChange = () => {
    resize();
    watchDpr();
  };
  const watchDpr = () => {
    dprQuery?.removeEventListener('change', onDprChange);
    dprQuery = window.matchMedia(`(resolution: ${window.devicePixelRatio || 1}dppx)`);
    dprQuery.addEventListener('change', onDprChange);
  };
  watchDpr();

  const handle: SceneHandle = {
    renderer,
    scene,
    camera,
    controls,
    floatLinear,
    frameBox(size) {
      const r = 0.5 * Math.hypot(size.x, size.y, size.z);
      const dist = r / Math.sin(THREE.MathUtils.degToRad(camera.fov) / 2);
      camera.near = dist / 100;
      camera.far = dist * 100;
      camera.position.set(dist * 0.55, -dist * 0.65, dist * 0.55);
      controls.target.set(0, 0, 0);
      camera.updateProjectionMatrix();
      controls.update();
    },
    setBloom(opts) {
      if (!opts || !hdrTargets) {
        composer?.passes.forEach((p) => p.dispose());
        composer?.dispose();
        composer = null;
        bloom = null;
        if (raf === 0 && !disposed) render();
        return;
      }
      if (!composer) {
        const target = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 });
        composer = new EffectComposer(renderer, target);
        composer.renderTarget2.samples = 0; // only the scene pass needs multisampling
        composerRatio = 0;
        composer.addPass(new RenderPass(scene, camera));
        bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.9, 0.5, 0.6);
        composer.addPass(bloom);
        composer.addPass(new OutputPass());
        sizeComposer();
      }
      bloom!.strength = opts.strength ?? 0.9;
      bloom!.radius = opts.radius ?? 0.5;
      bloom!.threshold = opts.threshold ?? 0.6;
      if (raf === 0 && !disposed) render();
    },
    setBackground(color) {
      // Also as the scene background: the bloom passes clear with their own colour.
      if (color == null) renderer.setClearColor(0x000000, 0);
      else renderer.setClearColor(color, 1);
      scene.background = color == null ? null : new THREE.Color(color);
      if (raf === 0 && !disposed) render();
    },
    render,
    start(onFrame) {
      if (disposed) return;
      handle.stop();
      last = performance.now();
      const loop = (now: number) => {
        raf = requestAnimationFrame(loop);
        // rAF timestamps can precede the performance.now() taken in start().
        const dt = Math.max(0, now - last);
        last = now;
        onFrame?.(dt, now);
        render();
      };
      raf = requestAnimationFrame(loop);
    },
    stop() {
      if (raf) cancelAnimationFrame(raf);
      raf = 0;
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      handle.stop();
      observer.disconnect();
      dprQuery?.removeEventListener('change', onDprChange);
      controls.dispose();
      handle.setBloom(null);
      disposeObject(scene);
      scene.clear();
      renderer.dispose();
      // Browsers cap live WebGL contexts; release ours when switching scales.
      renderer.forceContextLoss();
    },
  };
  return handle;
}
