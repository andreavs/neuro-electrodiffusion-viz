import { MESH_GEOMETRIES, fetchMesh, siteKey, type MeshGeometry, type MeshLoader } from '../../scales/emi/meshes';
import { EMI_PARAMS, SLICE_FIELDS, sliceAxis } from '../../scales/emi/params';
import { defaultParams } from '../../ui/controls';
import { EmiSim } from '../../wasm/pkg/ed_wasm.js';
import type { FrameField, Params } from '../protocol';
import type { SolverFactory, SolverFrame, WorkerSolver } from '../solver';

/** The parts of the wasm-bindgen `EmiSim` the adapter uses, so tests can pass a fake. */
export interface EmiWasm {
  stimulate(which: string): void;
  stimulate_sphere(cell: number, x: number, y: number, z: number, r: number, g: number): number;
  add_probe(x: number, y: number, z: number, r: number): number;
  n_probes(): number;
  set_dt(dt: number): void;
  set_probe(x: number): void;
  step(n: number): number;
  time_ms(): number;
  n_cells(): number;
  probe_x(): number;
  stimulated(): number;
  bounds(): Float32Array;
  membrane_positions(): Float32Array;
  membrane_triangles(): Uint32Array;
  phi_m(): Float32Array;
  slice_positions(axis: number, frac: number): Float32Array;
  slice_tags(axis: number, frac: number): Float32Array;
  slice_values(axis: number, frac: number, field: string): Float32Array;
  take_traces(): Float32Array;
  free(): void;
}

/** What to simulate on: the axon bundle, or a loaded mesh file. */
export type EmiGeometry = { kind: 'bundle'; perSide: number; gap: number } | ({ kind: 'mesh' } & MeshGeometry);

export type EmiWasmFactory = (geometry: EmiGeometry, dt: number) => EmiWasm;

/**
 * Mesh and slice versions, shared by every solver in this worker so a new solver (after a
 * Reset) never repeats a version the view has already cached.
 */
let version = 0;

/** Keys that need a new mesh. */
const GEOMETRY_KEYS = ['geometry'];
/** Keys that need a new bundle (they mean nothing to a mesh geometry). */
const BUNDLE_KEYS = ['axons', 'gap'];
const SLICE_KEYS = ['slice', 'slicePos'];
/** Most wall time one tick may catch up on, so a stall does not turn into a burst, ms. */
const MAX_CATCH_UP_MS = 100;

/**
 * Frame state the view must receive once: the mesh after every rebuild and the slice triangles
 * after every move. `geometry` and `slice` count the versions so the view can cache by them.
 */
export interface PackState {
  geometry: number;
  slice: number;
  geometrySent: boolean;
  sliceSent: boolean;
}

/**
 * Packs the current state of `sim` into a frame. Every array is a fresh copy (transferable).
 * Static geometry is included only while `state` says it has not been sent.
 */
export function packFrame(sim: EmiWasm, params: Params, state: PackState, steps: number): SolverFrame {
  const geometry = String(params.geometry ?? 'bundle');
  const axis = sliceAxis(String(params.slice), geometry);
  const frac = Number(params.slicePos);
  const nCells = sim.n_cells();
  const nProbes = sim.n_probes();
  const fields: FrameField[] = [];
  if (!state.geometrySent) {
    const nodes = sim.membrane_positions();
    const tris = sim.membrane_triangles();
    fields.push(
      { name: 'nodes', shape: [nodes.length / 3, 3], data: nodes },
      // Node indices stay exact in float32 (meshes have far fewer than 2²⁴ nodes).
      { name: 'tris', shape: [tris.length / 3, 3], data: Float32Array.from(tris) },
    );
  }
  const sendSlice = !state.sliceSent;
  if (sendSlice) {
    const tags = sim.slice_tags(axis, frac);
    fields.push(
      { name: 'slicePositions', shape: [tags.length, 3, 3], data: sim.slice_positions(axis, frac) },
      { name: 'sliceTags', shape: [tags.length], data: tags },
    );
  }
  const phiM = sim.phi_m();
  const slice = sim.slice_values(axis, frac, String(params.field));
  const traces = sim.take_traces();
  // Only now that every wasm call has succeeded: a throw above resends the static parts.
  state.geometrySent = true;
  if (sendSlice) state.sliceSent = true;
  fields.push(
    { name: 'phiM', shape: [phiM.length], data: phiM },
    { name: 'slice', shape: [slice.length / 3, 3], data: slice },
    { name: 'traces', shape: [traces.length / (1 + nProbes), 1 + nProbes], data: traces },
  );
  const [lx, ly, lz, hx, hy, hz] = sim.bounds();
  return {
    timeMs: sim.time_ms(),
    fields,
    steps,
    stats: {
      geometry: state.geometry,
      slice: state.slice,
      nCells,
      nProbes,
      // Which geometry this frame shows: an index into MESH_GEOMETRIES, or -1 for the bundle.
      geometryIndex: MESH_GEOMETRIES.findIndex((g) => g.value === geometry),
      probeX: sim.probe_x(),
      // Which field `slice` holds (index into SLICE_FIELDS), so the view never colours it with another field's bar.
      sliceField: SLICE_FIELDS.indexOf(String(params.field)),
      stimulated: sim.stimulated(),
      loX: lx!,
      loY: ly!,
      loZ: lz!,
      hiX: hx!,
      hiY: hy!,
      hiZ: hz!,
    },
  };
}

/** Worker-side adapter around the wasm KNP-EMI simulation. */
export function makeEmiSolver(
  initial: Params,
  create: EmiWasmFactory,
  now: () => number = () => performance.now(),
  load: MeshLoader = fetchMesh,
): WorkerSolver {
  const params: Params = { ...defaultParams(EMI_PARAMS), ...initial };
  const state: PackState = { geometry: version, slice: version, geometrySent: false, sliceSent: false };
  let steps = 0;
  let msPerStep = 0;
  let nextStim = Infinity;
  /** Simulated time owed by the playback rate, ms, and when it was last topped up. */
  let owed = 0;
  let lastWall = NaN;
  /** Loaded mesh geometries by name, and the one being fetched. */
  const meshes = new Map<string, MeshGeometry>();
  let loading: { name: string; error?: unknown } | null = null;
  let wake = () => {};

  const geometryName = () => String(params.geometry ?? 'bundle');
  const stimulate = (s: EmiWasm) => {
    const mesh = meshes.get(geometryName());
    if (geometryName() === 'bundle' || !mesh) {
      s.stimulate(String(params.stimulus));
      return;
    }
    const { stimuli } = mesh.meta;
    const site = stimuli.find((x) => x.key === params[siteKey(geometryName())]) ?? stimuli[0];
    if (site) s.stimulate_sphere(site.cell, ...site.centre, site.radius, site.g);
  };

  const schedule = (s: EmiWasm) => {
    const period = Number(params.period);
    nextStim = period > 0 ? s.time_ms() + period : Infinity;
  };

  /**
   * A fresh simulation with the current parameters, stimulated at t = 0, or null while its
   * mesh is being fetched (the loop is woken when it arrives).
   */
  const build = (): EmiWasm | null => {
    const name = geometryName();
    let geometry: EmiGeometry;
    if (name === 'bundle') {
      geometry = { kind: 'bundle', perSide: Number(params.axons), gap: Number(params.gap) };
    } else {
      const mesh = meshes.get(name);
      if (!mesh) {
        fetchGeometry(name);
        return null;
      }
      geometry = { kind: 'mesh', ...mesh };
    }
    const s = create(geometry, Number(params.dt));
    loading = null; // a fetch still running for another geometry no longer matters
    try {
      if (geometry.kind === 'bundle') s.set_probe(Number(params.probe));
      else for (const p of geometry.meta.probes) s.add_probe(...p.centre, p.radius);
      stimulate(s);
    } catch (err) {
      s.free();
      throw err;
    }
    schedule(s);
    steps = 0;
    msPerStep = 0;
    owed = 0;
    lastWall = NaN;
    state.geometry = ++version;
    state.slice = ++version;
    state.geometrySent = false;
    state.sliceSent = false;
    return s;
  };

  const fetchGeometry = (name: string) => {
    if (loading?.name === name && !loading.error) return;
    const pending: { name: string; error?: unknown } = { name };
    loading = pending;
    load(name).then(
      (mesh) => {
        meshes.set(name, mesh);
        if (loading !== pending) return;
        loading = null;
        if (geometryName() !== name || current) return;
        // Build now, not on the next tick, so a paused view shows the cell too.
        try {
          current = build();
        } catch (err) {
          pending.error = err;
          loading = pending;
        }
        wake();
      },
      (err: unknown) => {
        if (loading !== pending) return;
        pending.error = err;
        wake();
      },
    );
  };

  /** Rethrows a failed mesh load once, so the loop reports it. */
  const throwLoadError = () => {
    if (!loading?.error) return;
    const err = loading.error;
    loading = null;
    throw err;
  };

  let current = build();

  return {
    setWake(w) {
      wake = w;
    },
    advance(budgetMs) {
      throwLoadError();
      if (!current) {
        current = build();
        if (!current) return 'wait';
      }
      const s = current;
      const t0 = now();
      const dt = Number(params.dt);
      // Top up the simulated time owed by the playback rate, catching up on at most
      // MAX_CATCH_UP_MS of wall time after a pause or a slow tick.
      const cap = Math.max(dt, (MAX_CATCH_UP_MS * Number(params.rate)) / 1000);
      const wall = Number.isFinite(lastWall) ? t0 - lastWall : 0;
      if (!Number.isFinite(owed)) owed = 0;
      owed = Math.min(owed + (Math.min(wall, MAX_CATCH_UP_MS) * Number(params.rate)) / 1000, cap);
      lastWall = t0;
      let taken = 0;
      // Whole steps only; stop before one that would overrun the budget.
      while (owed >= dt * (1 - 1e-9)) {
        const elapsed = now() - t0;
        if (taken > 0 && elapsed + msPerStep > budgetMs) break;
        s.step(1);
        taken++;
        owed -= dt;
        if (s.time_ms() >= nextStim - 1e-9) {
          stimulate(s);
          nextStim += Number(params.period);
        }
        const per = Math.max(now() - t0, 1e-3) / taken;
        msPerStep = msPerStep === 0 ? per : 0.8 * msPerStep + 0.2 * per;
      }
      steps += taken;
      return true;
    },
    frame() {
      throwLoadError();
      return current ? packFrame(current, params, state, steps) : { timeMs: 0, fields: [], steps: 0, stats: { loading: 1 } };
    },
    setParams(changed) {
      Object.assign(params, changed);
      const bundle = geometryName() === 'bundle';
      if (GEOMETRY_KEYS.some((k) => k in changed) || (bundle && BUNDLE_KEYS.some((k) => k in changed))) {
        // Build first: if that throws, the old simulation is still intact.
        const next = build();
        current?.free();
        current = next;
        return;
      }
      if (!current) return; // the mesh is still loading; build() reads params when it arrives
      if ('dt' in changed) current.set_dt(Number(params.dt));
      if ('period' in changed) schedule(current);
      if ('probe' in changed) current.set_probe(Number(params.probe));
      if (SLICE_KEYS.some((k) => k in changed)) {
        state.slice = ++version;
        state.sliceSent = false;
      }
    },
    action(name) {
      if (name === 'stimulate' && current) {
        stimulate(current);
        schedule(current);
      }
    },
    dispose() {
      current?.free();
      current = null;
      loading = null;
    },
  };
}

export const createEmiSolver: SolverFactory = (params) =>
  makeEmiSolver(params, (g, dt) =>
    g.kind === 'bundle' ? new EmiSim(g.perSide, g.gap, dt) : EmiSim.from_mesh(g.bytes, dt),
  );
