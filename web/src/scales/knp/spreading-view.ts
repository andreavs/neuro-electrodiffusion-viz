import * as THREE from 'three';
import {
  Colorbar,
  PointCloud,
  SlicePlane,
  TimeSeriesPlot,
  coolwarm,
  createScene,
  setColormap,
  setRange,
  viridis,
  watchRenderStyle,
  type RenderStyle,
  type SceneHandle,
} from '../../render';
import type { SimClient } from '../../sim/client';
import type { FrameMessage } from '../../sim/protocol';
import { createAboutPanel } from '../../ui/about';
import { ControlPanel } from '../../ui/controls';
import { startRenderLoop } from '../../ui/renderLoop';
import { type ScaleView } from '../registry';
import { WORLD_PER_UM, gridToWorld, type Vec3 } from './neuron';
import { SD_FIELDS, SD_PARAMS } from './spreading-params';
import './view.css';

/** Membrane potential colour range of the neurons, mV. */
const V_RANGE = { min: -80, max: 0 };
/** Default [K⁺]e colour range, mM: baseline to fully depolarised tissue. */
const K_RANGE = { min: 3, max: 30 };
/** Neuron sprite diameter, µm. */
const NEURON_UM = 7;

/** The 10,000-neuron slab of the spreading-depression scenario. */
export function createSpreadingView(): ScaleView {
  let scene: SceneHandle | null = null;
  let plots: TimeSeriesPlot[] = [];
  let stopStyle: (() => void) | null = null;

  return {
    async mount(root: HTMLElement, client: SimClient) {
      root.innerHTML = `
        <section class="kv">
          <header class="kv-head">
            <h2>Spreading depression from ion coupling alone</h2>
            <p class="muted">Ten thousand simplified neurons (a soma and a dendrite each) sit in a 1 × 1 × 0.2 mm
              slab of tissue. They have no synapses: they interact only through the ions they release into and take
              up from the shared extracellular space. One neuron near the centre is driven hard. Its K⁺ raises the
              [K⁺]e around it; if that depolarises the neighbours enough that they release K⁺ faster than their
              pumps and diffusion can clear it, a self-sustaining wave of depolarisation spreads through the slab.
              Lower the persistent Na⁺ current or strengthen the pump to see the threshold. The findings of a
              parameter scan are in
              <a href="https://github.com/andreavs/neuro-electrodiffusion-viz/blob/main/docs/spreading-depression.md"
                target="_blank" rel="noopener">docs/spreading-depression.md</a>.</p>
          </header>
          <div class="kv-main">
            <div class="kv-stage">
              <canvas class="kv-canvas" aria-label="3-D view of the neurons in the tissue slab"></canvas>
              <div class="kv-colorbar"></div>
              <div class="kv-legend">Neurons: <i class="kv-vbar"></i> ${V_RANGE.min} to ${V_RANGE.max} mV</div>
            </div>
            <aside class="kv-side"></aside>
          </div>
          <div class="kv-plots">
            <figure><figcaption>Neurons depolarised</figcaption><canvas data-plot="depol"></canvas></figure>
            <figure><figcaption>[K⁺]e in the soma layer</figcaption><canvas data-plot="k"></canvas></figure>
            <dl class="kv-readout">
              <dt>Depolarised</dt><dd data-out="depol">–</dd>
              <dt>Peak [K⁺]e</dt><dd data-out="peak">–</dd>
              <dt>Neuron steps</dt><dd data-out="cost">–</dd>
            </dl>
          </div>
        </section>`;
      const q = <T extends HTMLElement>(sel: string) => root.querySelector<T>(sel)!;
      const out = { depol: q('[data-out="depol"]'), peak: q('[data-out="peak"]'), cost: q('[data-out="cost"]') };
      q('.kv-head').append(createAboutPanel('knp'));
      const panel = new ControlPanel(q('.kv-side'), client, SD_PARAMS);
      try {
        scene = createScene(q<HTMLCanvasElement>('.kv-canvas'));
      } catch (err) {
        out.depol.textContent = `3-D view unavailable (WebGL2 needed): ${String(err)}`;
        return;
      }
      const view = scene;
      const depolPlot = new TimeSeriesPlot(q('[data-plot="depol"]'), {
        traces: [{ name: 'depolarised', color: '#e4572e' }],
        windowMs: 20_000,
        xLabel: 'time (ms)',
        yLabel: '%',
        yRange: [0, 100],
      });
      const kPlot = new TimeSeriesPlot(q('[data-plot="k"]'), {
        traces: [
          { name: 'max', color: '#ff9f1c' },
          { name: 'mean', color: '#4cc9f0' },
        ],
        windowMs: 20_000,
        xLabel: 'time (ms)',
        yLabel: '[K⁺]e (mM)',
      });
      plots = [depolPlot, kPlot];

      let size: Vec3 | null = null;
      let slice: SlicePlane | null = null;
      let neurons: PointCloud | null = null;
      let neuronXyz = new Float32Array(0);
      let driven: THREE.Points | null = null;
      let colorbar: Colorbar | null = null;
      let shownField = '';
      let generation = -1;
      let lastTime = -1;
      let lastField: Float32Array | null = null;
      let style: RenderStyle = 'classic';
      const glow = () => style === 'glow';
      /** Dense cloud: dim sprites so 10,000 of them don't bloom into a white sheet. */
      const NEURON_GLOW = 0.35;
      const fieldInfo = (f: string) => SD_FIELDS.find((x) => x.value === f) ?? SD_FIELDS[0];

      const rebuildColorbar = (field: string) => {
        colorbar?.dispose();
        const fi = fieldInfo(field);
        const cm = field === 'phi' ? coolwarm : viridis;
        colorbar = new Colorbar({ colormap: cm, label: fi.label, units: fi.units });
        q('.kv-colorbar').append(colorbar.el);
        // A fixed [K⁺]e range shows the wave against the baseline; auto would track the peak.
        if (field === 'k') colorbar.setRange(K_RANGE.min, K_RANGE.max);
        if (slice) setColormap(slice, cm);
      };

      /** Box, slice through the soma layer, and the neurons from the first frame. */
      const build = (frame: FrameMessage, shape: Vec3) => {
        const st = frame.stats ?? {};
        size = [st.sizeX!, st.sizeY!, st.sizeZ!];
        const extent = new THREE.Vector3(...size).multiplyScalar(WORLD_PER_UM);
        // The somata fill the lower half of the slab: slice through its middle.
        slice = new SlicePlane({ shape, extent, axis: 'z', colormap: viridis, opacity: 0.9, smooth: view.floatLinear });
        slice.index = Math.floor(shape[2] / 4);
        view.scene.add(slice.mesh);
        view.scene.add(
          new THREE.LineSegments(
            new THREE.EdgesGeometry(new THREE.BoxGeometry(extent.x, extent.y, extent.z)),
            new THREE.LineBasicMaterial({ color: 0x8890a8 }),
          ),
        );
        view.frameBox(extent);
      };

      const placeNeurons = (somata: Float32Array) => {
        const n = somata.length / 3;
        if (!neurons || neurons.capacity < n) {
          neurons?.dispose();
          neurons = new PointCloud({ capacity: n, size: NEURON_UM * WORLD_PER_UM, colormap: coolwarm, range: V_RANGE });
          neurons.setGlow(glow(), NEURON_GLOW);
          view.scene.add(neurons.points);
        }
        const xyz = new Float32Array(somata.length);
        for (let i = 0; i < n; i++) xyz.set(gridToWorld(somata.subarray(3 * i, 3 * i + 3), size!), 3 * i);
        neuronXyz = xyz;
        neurons.update(xyz, n, new Float32Array(n).fill(-65));
      };

      /** Yellow markers on the driven neurons (indices into the soma list). */
      const markDriven = (drivenIdx: Float32Array) => {
        if (driven) {
          view.scene.remove(driven);
          driven.geometry.dispose();
          (driven.material as THREE.Material).dispose();
          driven = null;
        }
        if (drivenIdx.length) {
          const pos = new Float32Array(3 * drivenIdx.length);
          drivenIdx.forEach((i, j) => pos.set(neuronXyz.subarray(3 * i, 3 * i + 3), 3 * j));
          const g = new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(pos, 3));
          driven = new THREE.Points(g, new THREE.PointsMaterial({ color: 0xffe14d, size: 8, sizeAttenuation: false }));
          view.scene.add(driven);
        }
      };

      const paintSlice = () => {
        if (!slice || !lastField || !colorbar) return;
        slice.update(lastField);
        setRange(slice, colorbar.update(slice.values));
      };

      const draw = (frame: FrameMessage) => {
        const f = Object.fromEntries(frame.fields.map((x) => [x.name, x.data]));
        const field = frame.fields[0]!;
        const st = frame.stats ?? {};
        if (!size) build(frame, field.shape as Vec3);
        if (field.name !== shownField) {
          shownField = field.name;
          rebuildColorbar(shownField);
          kPlot.clear();
        }
        if (f.somata) placeNeurons(f.somata);
        if (f.driven) markDriven(f.driven);
        if (st.generation !== generation || frame.timeMs < lastTime) {
          generation = st.generation ?? 0;
          for (const p of plots) p.clear();
        }
        lastTime = frame.timeMs;
        lastField = field.data;
        paintSlice();
        if (neurons && f.v) neurons.update(neuronXyz, f.v.length, f.v);

        const values = slice?.values ?? field.data;
        let max = -Infinity;
        let sum = 0;
        for (const v of values) {
          if (v > max) max = v;
          sum += v;
        }
        const depol = 100 * (st.depolarised ?? 0);
        depolPlot.push(0, frame.timeMs, depol);
        if (shownField === 'k') {
          kPlot.push(0, frame.timeMs, max);
          kPlot.push(1, frame.timeMs, sum / values.length);
        }
        for (const p of plots) p.draw();
        out.depol.textContent = `${depol.toFixed(1)} %`;
        out.peak.textContent = shownField === 'k' ? `${max.toFixed(1)} mM` : '–';
        out.cost.textContent = `${(st.substeps ?? 0).toFixed(1)} per cell per step`;
      };

      // The [K⁺]e slice is the point of this view, so it stays in both styles.
      const applyStyle = (next: RenderStyle) => {
        style = next;
        view.setBackground(glow() ? 0x05060a : null);
        view.setBloom(glow() ? { strength: 0.7, radius: 0.5, threshold: 0.6 } : null);
        neurons?.setGlow(glow(), NEURON_GLOW);
      };
      stopStyle = watchRenderStyle(applyStyle);

      view.start();
      try {
        await client.ready();
      } catch (err) {
        out.depol.textContent = `Failed to load solver: ${String(err)}`;
        return;
      }
      startRenderLoop(client, { draw, stats: panel });
      client.create('knp-sd', panel.params());
      panel.setRunning(true);
    },
    unmount() {
      stopStyle?.();
      stopStyle = null;
      for (const p of plots) p.dispose();
      plots = [];
      scene?.dispose();
      scene = null;
    },
  };
}
