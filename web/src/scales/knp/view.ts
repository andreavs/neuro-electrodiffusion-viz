import * as THREE from 'three';
import {
  Colorbar,
  FlowTracers,
  SlicePlane,
  SurfaceField,
  TimeSeriesPlot,
  VolumeField,
  coolwarm,
  createScene,
  disposeObject,
  icefire,
  restRange,
  setColormap,
  setRange,
  viridis,
  watchRenderStyle,
  type Axis,
  type RenderStyle,
  type SceneHandle,
} from '../../render';
import type { SimClient } from '../../sim/client';
import type { FrameMessage } from '../../sim/protocol';
import { createAboutPanel } from '../../ui/about';
import { ControlPanel } from '../../ui/controls';
import { startRenderLoop } from '../../ui/renderLoop';
import { getScale, type ScaleView } from '../registry';
import {
  RateMeter,
  WORLD_PER_UM,
  clipRange,
  buildNeuron,
  gridToWorld,
  spreadValues,
  worldToGrid,
  type Vec3,
} from './neuron';
import { KNP_FIELDS, KNP_PARAMS, PROBE_ORDER, encodeProbes } from './params';
import './view.css';

const AXES: readonly Axis[] = ['x', 'y', 'z'];
const PROBE_COLORS = ['#ff9f1c', '#4cc9f0', '#f72585', '#80ed99'];
/** Fraction of slice values inside the automatic colour range. */
const CLIP_QUANTILE = 0.99;
/** The glow colours centre on the median of every n-th field value (the resting value)... */
const REST_STRIDE = 3;
/** ...and span this fraction of the deviations from it, refreshed every few frames. */
const REST_QUANTILE = 0.999;
const REST_EVERY = 10;
/** Membrane potential colour range, mV. */
const V_RANGE = { min: -80, max: 40 };
/** Starting probes relative to the soma (µm): at the soma, beside it, and up the apical trunk. */
const DEFAULT_PROBES: Vec3[] = [
  [0, 0, 0],
  [40, 0, 0],
  [0, 0, 300],
];

export function createKnpView(): ScaleView {
  let scene: SceneHandle | null = null;
  let plots: TimeSeriesPlot[] = [];
  let disposed = false;
  let stopStyle: (() => void) | null = null;

  return {
    async mount(root: HTMLElement, client: SimClient) {
      const info = getScale('knp')!;
      root.innerHTML = `
        <section class="kv">
          <header class="kv-head">
            <h2>${info.title}</h2>
            <p class="muted">A pyramidal neuron, driven by random synaptic input, sits in a column of tissue. Every
              compartment releases and takes up Na⁺, K⁺, Cl⁻ (X⁻) and, in the Hay et al. cell, Ca²⁺ at a point, and the ions
              spread through the extracellular space by diffusion and electric drift (the KNP framework of Publication 1). Watch K⁺ build
              up around the soma as the cell fires. The cell is coloured by its membrane potential; click a slice to
              place a probe.</p>
          </header>
          <div class="kv-main">
            <div class="kv-stage">
              <canvas class="kv-canvas" aria-label="3-D view of the neuron in the tissue"></canvas>
              <div class="kv-colorbar"></div>
              <div class="kv-legend">Neuron: <i class="kv-vbar"></i> ${V_RANGE.min} to +${V_RANGE.max} mV</div>
            </div>
            <aside class="kv-side">
              <div class="kv-viewctl">
                <label>Slices <select data-view="slices">
                  <option value="one">One (x–z plane)</option><option value="three">Three orthogonal</option>
                </select></label>
                ${AXES.map((a) => `<label>${a} position <input type="range" data-slice="${a}" min="0" max="1" step="1" /></label>`).join('')}
                <label><input type="checkbox" data-view="neuron" checked /> show neuron</label>
                <button type="button" class="ctl ctl-button" data-view="clear">Clear probes</button>
              </div>
            </aside>
          </div>
          <div class="kv-plots">
            <figure><figcaption>Somatic membrane potential</figcaption><canvas data-plot="soma"></canvas></figure>
            <figure><figcaption data-out="probe-title">Probes</figcaption><canvas data-plot="probes"></canvas></figure>
            <dl class="kv-readout">
              <dt>Spikes</dt><dd data-out="spikes">–</dd>
              <dt>Firing rate</dt><dd data-out="rate">–</dd>
              <dt>Field maximum</dt><dd data-out="peak">–</dd>
            </dl>
          </div>
        </section>`;
      const q = <T extends HTMLElement>(sel: string) => root.querySelector<T>(sel)!;
      const out = {
        spikes: q('[data-out="spikes"]'),
        rate: q('[data-out="rate"]'),
        peak: q('[data-out="peak"]'),
      };
      q('.kv-head').append(createAboutPanel('knp'));
      const panel = new ControlPanel(q('.kv-side'), client, KNP_PARAMS);
      q('.kv-side').append(q('.kv-viewctl'));
      const sliceMode = q<HTMLSelectElement>('[data-view="slices"]');
      const showNeuron = q<HTMLInputElement>('[data-view="neuron"]');
      const sliders = Object.fromEntries(AXES.map((a) => [a, q<HTMLInputElement>(`[data-slice="${a}"]`)])) as Record<
        Axis,
        HTMLInputElement
      >;

      try {
        scene = createScene(q<HTMLCanvasElement>('.kv-canvas'));
      } catch (err) {
        out.spikes.textContent = `3-D view unavailable (WebGL2 needed): ${String(err)}`;
        return;
      }
      const view = scene;
      const somaPlot = new TimeSeriesPlot(q('[data-plot="soma"]'), {
        traces: [{ name: 'soma', color: '#e4572e' }],
        windowMs: 300,
        capacity: 8192,
        xLabel: 'time (ms)',
        yLabel: 'V (mV)',
        yRange: [-90, 50],
      });
      plots = [somaPlot];
      let probePlot: TimeSeriesPlot | null = null;

      // Built from the first frame, once the grid is known, and rebuilt when the cell changes.
      let size: Vec3 | null = null;
      let built: THREE.Object3D[] = [];
      let builtComps = 0;
      let shape: Vec3 = [1, 1, 1];
      let slices: Partial<Record<Axis, SlicePlane>> = {};
      let volume: VolumeField | null = null;
      let flow: FlowTracers | null = null;
      let box: THREE.LineSegments<THREE.EdgesGeometry, THREE.LineBasicMaterial> | null = null;
      let style: RenderStyle = 'classic';
      let restSample = new Float32Array(0);
      let restFrames = 0;
      /** 1 for the grid cells holding a compartment, which the glow range ignores. */
      let sourceMask = new Uint8Array(0);
      let neuron: SurfaceField | null = null;
      let vertexComp: Uint16Array = new Uint16Array(0);
      let vertexValues = new Float32Array(0);
      let probes: Vec3[] = [];
      const markers = new THREE.Group();
      view.scene.add(markers);
      let colorbar: Colorbar | null = null;
      let shownField = '';
      const rate = new RateMeter();
      const clipScratch: number[] = [];
      let lastField: Float32Array | null = null;
      let lastTime = -1;

      const fieldInfo = (f: string) => KNP_FIELDS.find((x) => x.value === f) ?? KNP_FIELDS[0];

      const glow = () => style === 'glow';
      /** Glow shows changes from the resting value, of either sign, in a map that is black at rest. */
      const colormapFor = (field: string) => (glow() ? icefire : field === 'phi' ? coolwarm : viridis);

      const rebuildColorbar = (field: string) => {
        colorbar?.dispose();
        const fi = fieldInfo(field);
        const cm = colormapFor(field);
        // Glow centres concentrations on their resting value (below), not on zero.
        colorbar = new Colorbar({ colormap: cm, label: fi.label, units: fi.units, symmetric: field === 'phi' });
        q('.kv-colorbar').append(colorbar.el);
        restFrames = 0; // a fresh bar holds no range yet: fit the glow range on the next paint
        for (const s of Object.values(slices)) setColormap(s, cm);
        if (volume) {
          setColormap(volume, cm);
          volume.setMode(cm.diverging ? 'diverging' : 'sequential');
        }
        if (flow) {
          setColormap(flow, cm);
          flow.reset();
        }
      };

      const rebuildProbePlot = () => {
        probePlot?.dispose();
        const fi = fieldInfo(shownField);
        probePlot = new TimeSeriesPlot(q('[data-plot="probes"]'), {
          traces: probes.map((_, i) => ({
            name: `probe ${i + 1}`,
            color: PROBE_COLORS[i]!,
          })),
          windowMs: 1000,
          xLabel: 'time (ms)',
          yLabel: `${fi.label} (${fi.units})`,
        });
        plots = [somaPlot, probePlot];
        q('[data-out="probe-title"]').textContent = probes.length
          ? `${fi.label} at the probes`
          : 'Click a slice to place a probe';
      };

      const setProbes = (next: Vec3[]) => {
        probes = next.slice(-PROBE_COLORS.length);
        client.setParams({ probes: encodeProbes(probes) });
        disposeObject(markers);
        markers.clear();
        if (size) {
          probes.forEach((p, i) => {
            const m = new THREE.Mesh(
              new THREE.SphereGeometry(12 * WORLD_PER_UM, 16, 12),
              new THREE.MeshBasicMaterial({ color: PROBE_COLORS[i]! }),
            );
            m.position.set(...gridToWorld(p, size!));
            markers.add(m);
          });
        }
        rebuildProbePlot();
      };

      /** Copies the visible layers of the last field into the slices and sets their range. */
      const paintSlices = () => {
        if (!lastField || !colorbar) return;
        if (glow() && volume) {
          volume.update(lastField);
          let data: [number, number] | [] = [];
          if (colorbar.isAuto && restFrames++ % REST_EVERY === 0) {
            const n = Math.ceil(lastField.length / REST_STRIDE);
            if (restSample.length !== n) restSample = new Float32Array(n);
            for (let i = 0; i < n; i++) restSample[i] = lastField[i * REST_STRIDE]!;
            // Median of the finite values (sort puts NaN last).
            let finite = 0;
            for (let i = 0; i < n; i++) if (Number.isFinite(restSample[i]!)) finite++;
            const rest = shownField === 'phi' || finite === 0 ? 0 : restSample.sort()[finite >> 1]!;
            data = restRange(lastField, sourceMask, rest, REST_QUANTILE, REST_STRIDE, restSample);
          }
          const range = colorbar.update(data);
          setRange(volume, range);
          if (flow) {
            flow.setField(lastField);
            setRange(flow, range);
          }
          return;
        }
        const shown = Object.values(slices).filter((s) => s.mesh.visible);
        shown.forEach((s) => s.update(lastField!));
        // The cells holding a compartment are far off the rest; clip them so the halo shows.
        const data = colorbar.isAuto
          ? clipRange(
              shown.map((s) => s.values),
              CLIP_QUANTILE,
              clipScratch,
            )
          : [];
        const range = colorbar.update(data);
        for (const s of shown) setRange(s, range);
      };

      const layoutSlices = () => {
        if (!size) return;
        colorbar?.resetAuto(); // a different plane: refit the colours to it
        restFrames = 0; // ...and recompute the glow range on the next paint
        const three = sliceMode.value === 'three';
        for (const a of AXES) {
          const visible = !glow() && (three || a === 'y');
          slices[a]!.mesh.visible = visible;
          sliders[a].parentElement!.hidden = !visible;
          slices[a]!.index = Number(sliders[a].value);
        }
        paintSlices(); // also while paused, when no frame will arrive
      };

      /** Per animation frame: stream the glow tracers (they keep moving while paused). */
      const onFrame = (dtMs: number) => {
        if (glow()) flow?.step(dtMs / 1000);
      };

      const applyStyle = (next: RenderStyle) => {
        const changed = next !== style;
        style = next;
        view.setBackground(glow() ? 0x05060a : null);
        view.setBloom(glow() ? { strength: 0.9, radius: 0.6, threshold: 0.45 } : null);
        q<HTMLElement>('[data-view="slices"]').parentElement!.hidden = glow();
        neuron?.setGlow(glow());
        if (volume) volume.mesh.visible = glow();
        if (flow) flow.lines.visible = glow();
        if (box) box.material.opacity = glow() ? 0.25 : 1;
        if (changed && shownField) rebuildColorbar(shownField);
        if (size) layoutSlices();
      };

      const build = (frame: FrameMessage, f: Record<string, Float32Array>) => {
        const st = frame.stats ?? {};
        const dims: Vec3 = [st.sizeX!, st.sizeY!, st.sizeZ!];
        shape = frame.fields[0]!.shape as Vec3;
        const extent = new THREE.Vector3(...dims).multiplyScalar(WORLD_PER_UM);
        const comps = f.compartments!;
        const soma = [0, 1, 2].map((a) => (comps[a]! + comps[3 + a]!) / 2) as Vec3;
        AXES.forEach((a, i) => {
          const n = shape[i]!;
          const idx = Math.floor((soma[i]! / dims[i]!) * n);
          Object.assign(sliders[a], { max: String(n - 1), value: String(idx) });
          slices[a] = new SlicePlane({
            shape,
            extent,
            axis: a,
            colormap: viridis,
            opacity: 0.85,
            smooth: view.floatLinear,
          });
          view.scene.add(slices[a]!.mesh);
          built.push(slices[a]!.mesh);
        });
        box = new THREE.LineSegments(
          new THREE.EdgesGeometry(new THREE.BoxGeometry(extent.x, extent.y, extent.z)),
          new THREE.LineBasicMaterial({ color: 0x8890a8, transparent: true }),
        );
        view.scene.add(box);
        built.push(box);
        volume = new VolumeField({
          shape,
          extent,
          colormap: colormapFor(shownField),
          smooth: view.floatLinear,
          steps: view.software ? 48 : 160,
        });
        view.scene.add(volume.mesh);
        built.push(volume.mesh);
        flow = new FlowTracers({ shape, extent, colormap: colormapFor(shownField) });
        view.scene.add(flow.lines);
        built.push(flow.lines);
        sourceMask = new Uint8Array(shape[0] * shape[1] * shape[2]);
        for (let c = 0; 7 * c + 6 < comps.length; c++) {
          const cell = [0, 1, 2].map((a) =>
            Math.floor(((comps[7 * c + a]! + comps[7 * c + 3 + a]!) / 2 / dims[a]!) * shape[a]!),
          );
          const [x, y, z] = [0, 1, 2].map((a) => Math.min(shape[a]! - 1, Math.max(0, cell[a]!)));
          sourceMask[x! + shape[0] * (y! + shape[1] * z!)] = 1;
        }
        const mesh = buildNeuron(comps, dims);
        vertexComp = mesh.vertexComp;
        vertexValues = new Float32Array(vertexComp.length);
        neuron = new SurfaceField(mesh.geometry, coolwarm, V_RANGE);
        view.scene.add(neuron.mesh);
        built.push(neuron.mesh);
        neuron.mesh.visible = showNeuron.checked;
        builtComps = comps.length;
        view.frameBox(extent);
        size = dims; // last, so a failed build is retried on the next frame
        applyStyle(style);
        setProbes(DEFAULT_PROBES.map((p) => [0, 1, 2].map((a) => p[a]! + soma[a]!) as Vec3));
      };

      /** Removes what `build` added, so the next frame builds the scene for a new cell. */
      const teardown = () => {
        for (const o of built) {
          view.scene.remove(o);
          disposeObject(o);
        }
        built = [];
        slices = {};
        neuron = null;
        volume = null;
        box = null;
        flow = null;
        lastField = null; // sized for the old grid
        size = null;
      };

      const draw = (frame: FrameMessage) => {
        const f = Object.fromEntries(frame.fields.map((x) => [x.name, x.data]));
        const field = frame.fields[0]!;
        if (size && f.compartments!.length !== builtComps) {
          teardown();
          lastTime = Infinity; // a different cell: clear the plots below
        }
        if (!size) build(frame, f);
        if (field.name !== shownField) {
          shownField = field.name;
          rebuildColorbar(shownField);
          rebuildProbePlot();
        }
        if (frame.timeMs < lastTime) {
          somaPlot.clear();
          probePlot?.clear();
          colorbar?.resetAuto();
          restFrames = 0;
          flow?.reset();
        }
        lastTime = frame.timeMs;

        lastField = field.data;
        paintSlices();
        if (neuron) neuron.update(spreadValues(f.v!, vertexComp, vertexValues));
        const trace = f.trace!;
        for (let i = 0; i < trace.length; i += 2) somaPlot.push(0, trace[i]!, trace[i + 1]!);
        const pv = f.probes!;
        const k = PROBE_ORDER.indexOf(field.name as (typeof PROBE_ORDER)[number]);
        const nProbe = Math.min(probes.length, pv.length / PROBE_ORDER.length);
        for (let p = 0; p < nProbe; p++) probePlot!.push(p, frame.timeMs, pv[p * PROBE_ORDER.length + k]!);
        for (const p of plots) p.draw();

        const st = frame.stats ?? {};
        const spikes = st.spikes ?? 0;
        let peak = -Infinity;
        for (let i = 0; i < field.data.length; i++) if (field.data[i]! > peak) peak = field.data[i]!;
        out.spikes.textContent = spikes.toLocaleString();
        out.rate.textContent = `${rate.push(frame.timeMs, spikes).toFixed(1)} Hz`;
        out.peak.textContent = `${peak.toFixed(3)} ${fieldInfo(field.name).units}`;
      };

      // Probes: a click (not a drag) on a visible slice.
      const raycaster = new THREE.Raycaster();
      let down: { x: number; y: number } | null = null;
      const canvas = view.renderer.domElement;
      canvas.addEventListener('pointerdown', (e) => (down = e.button === 0 ? { x: e.clientX, y: e.clientY } : null));
      canvas.addEventListener('pointerup', (e) => {
        const start = down;
        down = null;
        if (!start || !size || Math.hypot(e.clientX - start.x, e.clientY - start.y) > 4) return;
        const r = canvas.getBoundingClientRect();
        const ndc = new THREE.Vector2(
          ((e.clientX - r.left) / r.width) * 2 - 1,
          -((e.clientY - r.top) / r.height) * 2 + 1,
        );
        raycaster.setFromCamera(ndc, view.camera);
        // In the glow style the slices are hidden; clicks still land on the (hidden) x–z slice.
        const targets = Object.values(slices)
          .filter((s) => (glow() ? s.axis === 'y' : s.mesh.visible))
          .map((s) => s.mesh);
        const hit = raycaster.intersectObjects(targets, false)[0];
        if (hit) setProbes([...probes, worldToGrid(hit.point.toArray(), size)]);
      });
      q('[data-view="clear"]').addEventListener('click', () => setProbes([]));
      sliceMode.addEventListener('change', layoutSlices);
      for (const a of AXES) sliders[a].addEventListener('input', layoutSlices);
      showNeuron.addEventListener('change', () => neuron && (neuron.mesh.visible = showNeuron.checked));

      stopStyle = watchRenderStyle(applyStyle);
      view.start(onFrame);
      try {
        await client.ready();
      } catch (err) {
        out.spikes.textContent = `Failed to load solver: ${String(err)}`;
        return;
      }
      if (disposed) return;
      startRenderLoop(client, { draw, stats: panel });
      view.start(onFrame);
      client.create('knp', panel.params());
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
