import * as THREE from 'three';
import {
  Colorbar,
  SurfaceField,
  TimeSeriesPlot,
  coolwarm,
  createScene,
  disposeObject,
  setRange,
  viridis,
  type Colormap,
  type SceneHandle,
} from '../../render';
import type { SimClient } from '../../sim/client';
import type { FrameMessage } from '../../sim/protocol';
import { createAboutPanel } from '../../ui/about';
import { ControlPanel } from '../../ui/controls';
import { startRenderLoop } from '../../ui/renderLoop';
import { getScale, type ScaleView } from '../registry';
import { gather, membraneMesh, sliceAxisOf, sliceMesh, slicePlane, stimulatedCells, type QuadMesh } from './geometry';
import { AXON_START, EMI_PARAMS } from './params';
import './view.css';

/** Membrane potential colour range, mV: rest to the peak of an action potential. */
const PHI_M_RANGE = { min: -80, max: 40 };
/** Simulated time shown in the trace plots, ms. */
const TRACE_WINDOW_MS = 30;
const STIM_COLORS = ['#e4572e', '#f3a712', '#d1495b', '#ff7f51'];
const OTHER_COLORS = ['#4cc9f0', '#7bd389', '#b388eb', '#90a4ae', '#3a86ff', '#ffd166', '#06d6a0', '#ef476f'];

const FIELDS: Record<string, { label: string; units: string; colormap: Colormap }> = {
  phi: { label: 'Potential φ', units: 'mV', colormap: coolwarm },
  k: { label: '[K⁺]', units: 'mM', colormap: viridis },
  na: { label: '[Na⁺]', units: 'mM', colormap: viridis },
  cl: { label: '[Cl⁻]', units: 'mM', colormap: viridis },
};

/** A colormapped quad mesh plus the scratch array its per-vertex values are gathered into. */
interface Layer {
  field: SurfaceField;
  mesh: QuadMesh;
  values: Float32Array;
}

function makeLayer(mesh: QuadMesh, colormap: Colormap): Layer {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(mesh.positions, 3));
  g.setIndex(new THREE.BufferAttribute(mesh.index, 1));
  const field = new SurfaceField(g, colormap);
  return { field, mesh, values: new Float32Array(mesh.source.length) };
}

export function createEmiView(): ScaleView {
  let scene: SceneHandle | null = null;
  let plots: TimeSeriesPlot[] = [];
  let disposed = false;

  return {
    async mount(root: HTMLElement, client: SimClient) {
      const info = getScale('emi')!;
      root.innerHTML = `
        <section class="ev">
          <header class="ev-head">
            <h2>${info.title}</h2>
            <p class="muted">A bundle of unmyelinated axons, 200 µm long and 0.2 µm wide, with the extracellular space
              between them resolved by finite elements (Publication 2 of the thesis). A synapse at the left end of the
              stimulated axon starts an action potential. The axon surfaces show the membrane potential; the slice shows
              the potential or an ion concentration around them, so you can see the extracellular signal and K⁺ release
              that reach the neighbouring axons (ephaptic coupling).</p>
          </header>
          <div class="ev-main">
            <div class="ev-stage">
              <canvas class="ev-canvas" aria-label="3-D view of the axon bundle"></canvas>
              <div class="ev-bars"></div>
              <p class="ev-note" data-out="note"></p>
            </div>
            <aside class="ev-side"></aside>
          </div>
          <div class="ev-plots">
            <figure><figcaption>Membrane potential at the probe, stimulated axons (by number)</figcaption><canvas data-plot="stim"></canvas></figure>
            <figure><figcaption>Membrane potential at the probe, other axons by number (note the small scale)</figcaption><canvas data-plot="other"></canvas></figure>
          </div>
        </section>`;
      const q = <T extends HTMLElement>(sel: string) => root.querySelector<T>(sel)!;
      const note = q('[data-out="note"]');
      q('.ev-head').append(createAboutPanel('emi'));
      const panel = new ControlPanel(q('.ev-side'), client, EMI_PARAMS);

      try {
        scene = createScene(q<HTMLCanvasElement>('.ev-canvas'));
      } catch (err) {
        note.textContent = `3-D view unavailable (WebGL2 needed): ${String(err)}`;
        return;
      }
      const view = scene;
      // The bundle is ~1000× longer than wide: draw it in a group compressed along x.
      const bundle = new THREE.Group();
      view.scene.add(bundle);

      const bars = q('.ev-bars');
      let membrane: Layer | null = null;
      const membraneBar = new Colorbar({
        colormap: viridis,
        label: 'φ_M',
        units: 'mV',
        auto: false,
        range: PHI_M_RANGE,
        onChange: (r) => membrane && setRange(membrane.field, r),
      });
      bars.append(membraneBar.el);
      let sliceBar: Colorbar | null = null;
      let sliceField = '';

      let slice: Layer | null = null;
      let probe: THREE.LineLoop | null = null;
      // Cached static data, by the solver's version counters.
      let geometryId = -1;
      let sliceId = -1;
      let inside = false;
      let nodes: Float32Array = new Float32Array();
      let quads: Float32Array = new Float32Array();
      let corners: Float32Array = new Float32Array();
      let tags: Float32Array = new Float32Array();
      let extent = new THREE.Vector3(1, 1, 1);
      let probeX = NaN;
      // Trace routing: plot and trace index per cell (1-based), rebuilt when the stimulus set changes.
      let traceKey = '';
      let route: [number, number][] = [];
      let lastTraceT = -Infinity;

      const replace = (old: Layer | null, mesh: QuadMesh, colormap: Colormap): Layer => {
        old?.field.dispose();
        const layer = makeLayer(mesh, colormap);
        bundle.add(layer.field.mesh);
        return layer;
      };

      const rebuildMeshes = () => {
        const axis = sliceAxisOf(corners);
        membrane = replace(membrane, membraneMesh(nodes, quads, axis, slicePlane(corners, axis)), viridis);
        setRange(membrane.field, membraneBar.range);
        slice = replace(slice, sliceMesh(corners, tags, inside), FIELDS[sliceField]?.colormap ?? coolwarm);
      };

      let framed = '';
      const placeBundle = () => {
        const sx = (3 * extent.y) / extent.x;
        bundle.scale.set(sx, 1, 1);
        bundle.position.set((-extent.x * sx) / 2, -extent.y / 2, -extent.z / 2);
        // Keep the user's camera across resets; reframe only when the bundle's size changes.
        const key = `${extent.x}:${extent.y}:${extent.z}`;
        if (key !== framed) view.frameBox(new THREE.Vector3(extent.x * sx, extent.y, extent.z));
        framed = key;
        note.textContent = `Lengths along the axons are drawn ${Math.round(1 / sx)}× shorter than across them.`;
      };

      const placeProbe = (x: number) => {
        if (probe) {
          disposeObject(probe);
          bundle.remove(probe);
        }
        const [y, z] = [extent.y, extent.z];
        const pts = [
          [0, 0],
          [y, 0],
          [y, z],
          [0, z],
        ].map(([a, b]) => new THREE.Vector3(AXON_START + x, a, b));
        probe = new THREE.LineLoop(
          new THREE.BufferGeometry().setFromPoints(pts),
          new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.6 }),
        );
        bundle.add(probe);
      };

      const rebuildPlots = (nCells: number, stimulated: Set<number>) => {
        for (const p of plots) p.dispose();
        const stim: number[] = [];
        const other: number[] = [];
        for (let c = 1; c <= nCells; c++) (stimulated.has(c) ? stim : other).push(c);
        const make = (sel: string, cells: number[], colors: string[]) =>
          new TimeSeriesPlot(q(`[data-plot="${sel}"]`), {
            traces: cells.map((c, i) => ({ name: `${c}`, color: colors[i % colors.length]! })),
            windowMs: TRACE_WINDOW_MS,
            xLabel: 'time (ms)',
            yLabel: 'φ_M (mV)',
          });
        plots = [make('stim', stim, STIM_COLORS), make('other', other, OTHER_COLORS)];
        route = [];
        stim.forEach((c, i) => (route[c] = [0, i]));
        other.forEach((c, i) => (route[c] = [1, i]));
      };

      const draw = (frame: FrameMessage) => {
        const f = Object.fromEntries(frame.fields.map((x) => [x.name, x.data]));
        const s = frame.stats ?? {};
        const params = panel.params();
        let meshesStale = false;

        if (s.geometry !== geometryId && f.nodes && f.quads) {
          geometryId = s.geometry!;
          nodes = f.nodes;
          quads = f.quads;
          extent = new THREE.Vector3(s.extentX, s.extentY, s.extentZ);
          placeBundle();
          probeX = NaN;
          traceKey = '';
          meshesStale = true;
        }
        if (s.slice !== sliceId && f.slicePositions && f.sliceTags) {
          sliceId = s.slice!;
          corners = f.slicePositions;
          tags = f.sliceTags;
          meshesStale = true;
        }
        if (Boolean(params.inside) !== inside) {
          inside = Boolean(params.inside);
          meshesStale = true;
        }
        const fieldKey = String(params.field);
        if (fieldKey !== sliceField) {
          sliceField = fieldKey;
          const spec = FIELDS[fieldKey] ?? FIELDS.phi!;
          sliceBar?.dispose();
          sliceBar = new Colorbar({ colormap: spec.colormap, label: spec.label, units: spec.units });
          bars.append(sliceBar.el);
          meshesStale = true;
        }
        if (geometryId < 0 || sliceId < 0 || corners.length < 12) return;
        if (meshesStale) rebuildMeshes();

        if (s.probeX !== probeX) {
          probeX = s.probeX!;
          placeProbe(probeX);
          for (const p of plots) p.clear();
        }
        const nCells = s.nCells ?? 0;
        const key = `${geometryId}:${s.stimulated}`;
        if (key !== traceKey) {
          traceKey = key;
          rebuildPlots(nCells, stimulatedCells(s.stimulated ?? 0, nCells));
        }

        membrane!.field.update(gather(f.phiM!, membrane!.mesh.source, membrane!.values));
        const sv = gather(f.slice!, slice!.mesh.source, slice!.values);
        slice!.field.update(sv);

        const tr = f.traces!;
        const width = 1 + nCells;
        for (let r = 0; r + width <= tr.length; r += width) {
          const t = tr[r]!;
          // A reset or rebuild restarts the clock.
          if (t < lastTraceT) for (const p of plots) p.clear();
          lastTraceT = t;
          for (let c = 1; c <= nCells; c++) {
            const [p, i] = route[c]!;
            plots[p]!.push(i, t, tr[r + c]!);
          }
        }
        for (const p of plots) p.draw();
        // DOM writes last, after the canvases are drawn (they read layout).
        setRange(slice!.field, sliceBar!.update(sv));
      };

      view.start();
      try {
        await client.ready();
      } catch (err) {
        note.textContent = `Failed to load solver: ${String(err)}`;
        return;
      }
      if (disposed) return;
      startRenderLoop(client, { draw, stats: panel });
      // Re-register the scene's rAF after the frame consumer's, so each animation frame
      // renders the values it just received.
      view.start();
      client.create('emi', panel.params());
      panel.setRunning(true);
    },
    unmount() {
      disposed = true;
      for (const p of plots) p.dispose();
      plots = [];
      scene?.dispose();
      scene = null;
    },
  };
}
