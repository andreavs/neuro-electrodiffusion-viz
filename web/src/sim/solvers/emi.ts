import { EMI_PARAMS, SLICE_FIELDS, sliceAxis } from '../../scales/emi/params';
import { defaultParams } from '../../ui/controls';
import { EmiSim } from '../../wasm/pkg/ed_wasm.js';
import type { FrameField, Params } from '../protocol';
import type { SolverFactory, SolverFrame, WorkerSolver } from '../solver';

/** The parts of the wasm-bindgen `EmiSim` the adapter uses, so tests can pass a fake. */
export interface EmiWasm {
  stimulate(which: string): void;
  set_dt(dt: number): void;
  set_probe(x: number): void;
  step(n: number): number;
  time_ms(): number;
  n_cells(): number;
  probe_x(): number;
  stimulated(): number;
  extent(): Float32Array;
  membrane_positions(): Float32Array;
  membrane_quads(): Uint32Array;
  phi_m(): Float32Array;
  slice_positions(axis: number, frac: number): Float32Array;
  slice_tags(axis: number, frac: number): Float32Array;
  slice_values(axis: number, frac: number, field: string): Float32Array;
  take_traces(): Float32Array;
  free(): void;
}

export type EmiWasmFactory = (perSide: number, gap: number, dt: number) => EmiWasm;

/**
 * Mesh and slice versions, shared by every solver in this worker so a new solver (after a
 * Reset) never repeats a version the view has already cached.
 */
let version = 0;

/** Keys that need a new mesh. */
const GEOMETRY_KEYS = ['axons', 'gap'];
const SLICE_KEYS = ['slice', 'slicePos'];
/** Most wall time one tick may catch up on, so a stall does not turn into a burst, ms. */
const MAX_CATCH_UP_MS = 100;

/**
 * Frame state the view must receive once: the mesh after every rebuild and the slice quads
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
  const axis = sliceAxis(String(params.slice));
  const frac = Number(params.slicePos);
  const nCells = sim.n_cells();
  const fields: FrameField[] = [];
  if (!state.geometrySent) {
    const nodes = sim.membrane_positions();
    const quads = sim.membrane_quads();
    fields.push(
      { name: 'nodes', shape: [nodes.length / 3, 3], data: nodes },
      // Node indices stay exact in float32 (meshes have far fewer than 2²⁴ nodes).
      { name: 'quads', shape: [quads.length / 4, 4], data: Float32Array.from(quads) },
    );
  }
  const sendSlice = !state.sliceSent;
  if (sendSlice) {
    const tags = sim.slice_tags(axis, frac);
    fields.push(
      { name: 'slicePositions', shape: [tags.length, 4, 3], data: sim.slice_positions(axis, frac) },
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
    { name: 'slice', shape: [slice.length / 4, 4], data: slice },
    { name: 'traces', shape: [traces.length / (1 + nCells), 1 + nCells], data: traces },
  );
  const [ex, ey, ez] = sim.extent();
  return {
    timeMs: sim.time_ms(),
    fields,
    steps,
    stats: {
      geometry: state.geometry,
      slice: state.slice,
      nCells,
      probeX: sim.probe_x(),
      // Which field `slice` holds (index into SLICE_FIELDS), so the view never colours it with another field's bar.
      sliceField: SLICE_FIELDS.indexOf(String(params.field)),
      stimulated: sim.stimulated(),
      extentX: ex!,
      extentY: ey!,
      extentZ: ez!,
    },
  };
}

/** Worker-side adapter around the wasm KNP-EMI simulation. */
export function makeEmiSolver(
  initial: Params,
  create: EmiWasmFactory,
  now: () => number = () => performance.now(),
): WorkerSolver {
  const params: Params = { ...defaultParams(EMI_PARAMS), ...initial };
  const state: PackState = { geometry: version, slice: version, geometrySent: false, sliceSent: false };
  let steps = 0;
  let msPerStep = 0;
  let nextStim = Infinity;
  /** Simulated time owed by the playback rate, ms, and when it was last topped up. */
  let owed = 0;
  let lastWall = NaN;

  const schedule = (s: EmiWasm) => {
    const period = Number(params.period);
    nextStim = period > 0 ? s.time_ms() + period : Infinity;
  };

  /** A fresh simulation with the current parameters, stimulated at t = 0. */
  const build = (): EmiWasm => {
    const s = create(Number(params.axons), Number(params.gap), Number(params.dt));
    try {
      s.set_probe(Number(params.probe));
      s.stimulate(String(params.stimulus));
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
  let current = build();

  return {
    advance(budgetMs) {
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
        current.step(1);
        taken++;
        owed -= dt;
        if (current.time_ms() >= nextStim - 1e-9) {
          current.stimulate(String(params.stimulus));
          nextStim += Number(params.period);
        }
        const per = Math.max(now() - t0, 1e-3) / taken;
        msPerStep = msPerStep === 0 ? per : 0.8 * msPerStep + 0.2 * per;
      }
      steps += taken;
      return true;
    },
    frame: () => packFrame(current, params, state, steps),
    setParams(changed) {
      Object.assign(params, changed);
      if (GEOMETRY_KEYS.some((k) => k in changed)) {
        // Build first: if that throws, the old simulation is still intact.
        const next = build();
        current.free();
        current = next;
        return;
      }
      if ('dt' in changed) current.set_dt(Number(params.dt));
      if ('period' in changed) schedule(current);
      if ('probe' in changed) current.set_probe(Number(params.probe));
      if (SLICE_KEYS.some((k) => k in changed)) {
        state.slice = ++version;
        state.sliceSent = false;
      }
    },
    action(name) {
      if (name === 'stimulate') {
        current.stimulate(String(params.stimulus));
        schedule(current);
      }
    },
    dispose() {
      current.free();
    },
  };
}

export const createEmiSolver: SolverFactory = (params) =>
  makeEmiSolver(params, (perSide, gap, dt) => new EmiSim(perSide, gap, dt));
