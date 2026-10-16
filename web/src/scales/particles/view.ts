import * as THREE from 'three';
import { ExpandingRange, PointCloud, createScene, watchRenderStyle, disposeObject, type Colormap, type SceneHandle } from '../../render';
import type { SimClient } from '../../sim/client';
import type { FrameMessage } from '../../sim/protocol';
import { createAboutPanel } from '../../ui/about';
import { ControlPanel } from '../../ui/controls';
import { startRenderLoop } from '../../ui/renderLoop';
import { getScale, type ScaleView } from '../registry';
import { PARTICLE_PARAMS, signedDrop } from './params';
import { LinePlot } from './plot';
import { formatMfpt, histogramSteps, maxOf, mfptReadout } from './stats';
import './view.css';
import { createTour } from './tour';

/** Matches the solver's DISPLAY_MAX: ions beyond it are simulated but not drawn. */
const MAX_IONS = 100_000;
/** Brightness of each glowing ion (the bloom threshold keeps a dense cloud from burning out). */
const GLOW_BRIGHTNESS = 1;
const NECK_COLOR = '#ff9f1c';
const HEAD_COLOR = '#4cc9f0';
const FIELD_COLOR = '#e4572e';
const REF_COLOR = '#8d93a5';

/** Two-colour lookup: values below 0.5 → neck colour, above → head colour. */
function regionColormap(): Colormap {
  const lut = new Uint8Array(256 * 4);
  const [a, b] = [new THREE.Color(NECK_COLOR), new THREE.Color(HEAD_COLOR)];
  for (let i = 0; i < 256; i++) {
    // LUT bytes are sRGB, like every colormap (THREE.Color holds linear components).
    const c = (i < 128 ? a : b).getRGB({ r: 0, g: 0, b: 0 }, THREE.SRGBColorSpace);
    lut.set([Math.round(c.r * 255), Math.round(c.g * 255), Math.round(c.b * 255), 255], 4 * i);
  }
  return { name: 'regions', lut, diverging: false };
}

/** Spine wall, absorbing base and a stub of the parent dendrite. Rebuilt when the shape changes. */
function buildSpine(profile: Float32Array, neckRadius: number): THREE.Group {
  const group = new THREE.Group();
  const pts: THREE.Vector2[] = [];
  for (let i = 0; i < profile.length; i += 2) pts.push(new THREE.Vector2(profile[i], profile[i + 1]));
  const wall = new THREE.Mesh(
    new THREE.LatheGeometry(pts, 64),
    new THREE.MeshStandardMaterial({
      color: 0xb8c4e0,
      roughness: 0.5,
      transparent: true,
      opacity: 0.2,
      depthWrite: false,
      side: THREE.DoubleSide,
    }),
  );
  wall.rotation.x = Math.PI / 2; // lathe axis y → z (Z-up)
  const base = new THREE.Mesh(
    new THREE.CircleGeometry(neckRadius, 48),
    new THREE.MeshBasicMaterial({ color: FIELD_COLOR, transparent: true, opacity: 0.7, side: THREE.DoubleSide }),
  );
  const shaft = new THREE.Mesh(
    new THREE.CylinderGeometry(0.45, 0.45, 2.4, 48, 1, true),
    new THREE.MeshStandardMaterial({ color: 0x8890a8, transparent: true, opacity: 0.12, depthWrite: false, side: THREE.DoubleSide }),
  );
  shaft.rotation.z = Math.PI / 2; // cylinder axis y → x
  shaft.position.z = -0.45;
  group.add(wall, base, shaft);
  return group;
}

/** Arrows showing the drift on Ca²⁺ (along −E for z > 0): down when the field pushes ions out. */
function buildArrows(kind: string, drop: number, neckTop: number, top: number): THREE.Group {
  const group = new THREE.Group();
  if (kind === 'off' || drop === 0) return group;
  const dir = new THREE.Vector3(0, 0, drop > 0 ? -1 : 1);
  const zMax = kind === 'neck' ? neckTop : top;
  const len = Math.min(0.5, 0.15 + Math.abs(drop) / 150);
  const rows = Math.max(2, Math.round(zMax / 0.5));
  for (let k = 0; k < 4; k++) {
    const a = (k * Math.PI) / 2 + Math.PI / 4;
    const r = kind === 'neck' ? 0.28 : 0.75;
    for (let j = 0; j < rows; j++) {
      const z = ((j + 0.5) / rows) * zMax - (dir.z * len) / 2;
      group.add(new THREE.ArrowHelper(dir, new THREE.Vector3(r * Math.cos(a), r * Math.sin(a), z), len, FIELD_COLOR, 0.08, 0.05));
    }
  }
  return group;
}

export function createParticlesView(): ScaleView {
  let scene: SceneHandle | null = null;
  let plots: LinePlot[] = [];
  let disposed = false;
  let stopStyle: (() => void) | null = null;

  return {
    async mount(root: HTMLElement, client: SimClient) {
      const info = getScale('particles')!;
      root.innerHTML = `
        <section class="pv">
          <header class="pv-head">
            <h2>${info.title}</h2>
            <p class="muted">Ca²⁺ ions are released in the head of a dendritic spine and random-walk until they leave
              through the base of the neck. An electric potential drop across the spine adds a drift that speeds up or
              slows down their escape. Following Publication 3 of the thesis, the mean first passage time (MFPT) is
              compared with and without the field.</p>
          </header>
          <div class="pv-main">
            <div class="pv-stage">
              <canvas class="pv-canvas" aria-label="3-D view of the spine and the ions"></canvas>
              <div class="pv-legend">
                <span><i class="pv-swatch" style="background:${HEAD_COLOR}"></i>in the head</span>
                <span><i class="pv-swatch" style="background:${NECK_COLOR}"></i>in the neck</span>
                <span><i class="pv-swatch" style="background:${FIELD_COLOR}"></i>absorbing base · drift</span>
                <label><input type="checkbox" data-arrows checked /> field arrows</label>
              </div>
            </div>
            <aside class="pv-side"></aside>
          </div>
          <div class="pv-plots">
            <figure><figcaption>Fraction of ions still in the spine</figcaption><canvas data-plot="survival"></canvas></figure>
            <figure><figcaption>Exit-time distribution (fraction per bin)</figcaption><canvas data-plot="hist"></canvas></figure>
            <dl class="pv-readout">
              <dt>Ions inside</dt><dd data-out="alive">–</dd>
              <dt><i class="pv-swatch" style="background:${FIELD_COLOR}"></i>MFPT</dt><dd data-out="mfpt">–</dd>
              <dt><i class="pv-swatch" style="background:${REF_COLOR}"></i>MFPT, no field</dt><dd data-out="ref">–</dd>
            </dl>
          </div>
        </section>`;
      const q = <T extends HTMLElement>(sel: string) => root.querySelector<T>(sel)!;
      const out = { alive: q('[data-out="alive"]'), mfpt: q('[data-out="mfpt"]'), ref: q('[data-out="ref"]') };
      q('.pv-head').append(createAboutPanel('particles'));
      const panel = new ControlPanel(q('.pv-side'), client, PARTICLE_PARAMS);

      try {
        scene = createScene(q<HTMLCanvasElement>('.pv-canvas'));
      } catch (err) {
        out.alive.textContent = `3-D view unavailable (WebGL2 needed): ${String(err)}`;
        return;
      }
      const view = scene;
      view.frameBox(new THREE.Vector3(1.4, 1.4, 2.4));
      view.camera.position.z += 0.9;
      view.controls.target.set(0, 0, 0.9);
      view.scene.add(new THREE.HemisphereLight(0xffffff, 0x445066, 1.6));
      const sun = new THREE.DirectionalLight(0xffffff, 1.2);
      sun.position.set(2, -3, 4);
      view.scene.add(sun);
      const cloud = new PointCloud({ capacity: MAX_IONS, size: 0.014, colormap: regionColormap(), range: { min: 0, max: 1 } });
      view.scene.add(cloud.points);
      let spine: THREE.Group | null = null;
      let spineKey = '';
      let arrows = new THREE.Group();
      let geom = { neckTop: 1, top: 2 };
      view.scene.add(arrows);

      const arrowToggle = q<HTMLInputElement>('[data-arrows]');
      let arrowsKey = '';
      const refreshArrows = () => {
        const p = panel.params();
        const args = [String(p.field), signedDrop(Number(p.dv), String(p.direction)), geom.neckTop, geom.top] as const;
        const key = arrowToggle.checked ? args.join() : 'hidden';
        if (key === arrowsKey) return;
        arrowsKey = key;
        disposeObject(arrows);
        view.scene.remove(arrows);
        arrows = arrowToggle.checked ? buildArrows(...args) : new THREE.Group();
        view.scene.add(arrows);
      };
      arrowToggle.addEventListener('change', refreshArrows);
      q('.pv-head').after(
        createTour({
          apply: (p) => {
            const changed = panel.setValues(p);
            refreshArrows();
            if (changed && !panel.isRunning) panel.setRunning(true);
            return changed;
          },
          release: () => {
            client.action('release');
            if (!panel.isRunning) panel.setRunning(true);
          },
        }),
      );
      const onPanel = (ev: Event) => {
        const k = (ev.target as HTMLElement).dataset?.key;
        if (k === 'field' || k === 'dv' || k === 'direction') refreshArrows();
      };
      panel.el.addEventListener('change', onPanel);
      panel.el.addEventListener('input', onPanel);

      plots = [
        new LinePlot(q('[data-plot="survival"]'), { x: 'time (ms)', y: 'S(t)' }),
        new LinePlot(q('[data-plot="hist"]'), { x: 'exit time (ms)', y: 'fraction' }),
      ];
      const [survivalPlot, histPlot] = plots as [LinePlot, LinePlot];
      // The histogram's y axis only grows (no jitter) until the run restarts.
      const histY = new ExpandingRange(0.1);
      let lastTime = -1;

      const draw = (frame: FrameMessage) => {
        const f = Object.fromEntries(frame.fields.map((x) => [x.name, x.data]));
        const stats = frame.stats ?? {};
        cloud.update(f.positions!, f.positions!.length / 3, f.regions);

        // The sharded solver's first frame after a release can arrive before the wall profile.
        const profile = f.profile ?? new Float32Array(0);
        let sum = 0;
        for (let i = 0; i < profile.length; i++) sum += profile[i]! * (i + 1);
        const key = `${profile.length}:${sum}`;
        if (key !== spineKey && profile.length >= 4) {
          spineKey = key;
          if (spine) {
            disposeObject(spine);
            view.scene.remove(spine);
          }
          spine = buildSpine(profile, stats.neckRadius ?? 0.1);
          view.scene.add(spine);
          geom = { neckTop: stats.neckTop ?? 1, top: profile[profile.length - 1]! };
          refreshArrows();
        }

        const alive = stats.alive ?? 0;
        const ref = (stats.hasReference ?? 0) > 0;

        const tEnd = Math.max(frame.timeMs, 1e-3);
        survivalPlot.draw({
          x: [0, tEnd],
          y: [0, 1],
          series: [
            ...(ref ? [{ xy: f.survivalRef!, color: REF_COLOR, dashed: true }] : []),
            { xy: f.survival!, color: FIELD_COLOR },
          ],
        });
        if (frame.timeMs < lastTime) histY.reset();
        lastTime = frame.timeMs;
        histY.include(0, maxOf([f.histogram, ref ? f.histogramRef : undefined], 0.01) * 1.1);
        const tMax = stats.histTMax ?? 1;
        histPlot.draw({
          x: [0, tMax],
          y: [0, histY.max],
          series: [
            { xy: histogramSteps(f.histogram!, tMax), color: FIELD_COLOR, fill: true },
            ...(ref ? [{ xy: histogramSteps(f.histogramRef!, tMax), color: REF_COLOR, dashed: true }] : []),
          ],
        });
        // DOM writes last, after the canvases are drawn.
        out.alive.textContent = `${alive.toLocaleString()} of ${(stats.released ?? 0).toLocaleString()}`;
        out.mfpt.textContent = formatMfpt(mfptReadout(stats, '', alive));
        const refLeft = (stats.refReleased ?? 0) - (stats.refEscaped ?? 0);
        out.ref.textContent = ref ? formatMfpt(mfptReadout(stats, 'ref', refLeft)) : 'field is off';
      };

      // Draw the empty spine scene straight away; the data arrives once the solver is ready.
      let glow = false;
      stopStyle = watchRenderStyle((style) => {
        glow = style === 'glow';
        view.setBackground(glow ? 0x05060a : null);
        view.setBloom(glow ? { strength: 0.8, radius: 0.5, threshold: 0.5 } : null);
        cloud.setGlow(glow, GLOW_BRIGHTNESS);
      });
      view.start();
      try {
        await client.ready();
      } catch (err) {
        out.alive.textContent = `Failed to load solver: ${String(err)}`;
        return;
      }
      if (disposed) return;
      startRenderLoop(client, { draw, stats: panel });
      // Re-register the scene's rAF after the frame consumer's, so each animation frame
      // renders the positions it just received.
      view.start();
      client.create('particles', panel.params());
      panel.setRunning(true);
    },
    unmount() {
      disposed = true;
      stopStyle?.();
      stopStyle = null;
      for (const p of plots) p.dispose();
      plots = [];
      scene?.dispose();
      scene = null;
    },
  };
}
