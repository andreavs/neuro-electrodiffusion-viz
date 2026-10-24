import { SD_FIELDS, SD_GLIA_FIELDS, SD_PARAMS, SD_REBUILD_KEYS } from '../../scales/knp/spreading-params';
import { defaultParams } from '../../ui/controls';
import { SdSim } from '../../wasm/pkg/ed_wasm.js';
import type { Params } from '../protocol';
import type { SolverFactory, SolverFrame, WorkerSolver } from '../solver';

/** The parts of the wasm-bindgen `SdSim` the adapter uses, so tests can pass a fake. */
export interface SdWasm {
  set_drive(count: number, g_ns: number): void;
  set_pump(scale: number): void;
  set_buffering(rate: number): void;
  step(n: number): void;
  dt(): number;
  time_ms(): number;
  substeps_per_neuron(): number;
  depolarised(): number;
  shape(): Uint32Array;
  size(): Float64Array;
  field(name: string): Float32Array;
  somata(): Float32Array;
  v_soma(): Float32Array;
  driven(): Uint32Array;
  free(): void;
}

export type SdCreate = (seed: number, neurons: number, gNap: number, glia: number, coupling: number) => SdWasm;

/**
 * Counts builds across all adapters, so a view sees a new generation after a reset or a new
 * `create` as well as after a rebuild.
 */
let builds = 0;

/** wasm-bindgen wraps negative or fractional numbers for `usize`; reject them here. */
function count(name: string, v: unknown): number {
  const n = Number(v);
  if (!Number.isInteger(n) || n < 0) throw new Error(`${name} must be a whole number ≥ 0, got ${String(v)}`);
  return n;
}

/** Worker-side adapter around the wasm spreading-depression tissue. */
export function makeSpreadingSolver(
  initial: Params,
  create: SdCreate,
  now: () => number = () => performance.now(),
): WorkerSolver {
  const params: Params = { ...defaultParams(SD_PARAMS), ...initial };
  let sim: SdWasm | undefined;
  /** Changes on every build, so the view knows when the neurons moved. */
  let generation = 0;
  let somataSent = false;
  let drivenSent = false;
  let steps = 0;
  let msPerStep = 0;

  /** Applies the run-time parameters among the keys of `p` (values from `params`) to `s`. */
  const applyRuntime = (s: SdWasm, p: Params) => {
    if ('driven' in p || 'drive' in p) {
      s.set_drive(count('driven neurons', params.driven), Number(params.drive));
      drivenSent = false;
    }
    if ('pump' in p) s.set_pump(Number(params.pump));
    if ('buffering' in p) s.set_buffering(Number(params.buffering) / 1000); // 1/s → 1/ms
  };
  /** Builds a new tissue; the old one is kept if that fails. */
  const build = () => {
    const next = create(
      1,
      count('neurons', params.neurons),
      Number(params.gNap),
      Number(params.glia),
      Number(params.coupling),
    );
    try {
      applyRuntime(next, params);
    } catch (err) {
      next.free();
      throw err;
    }
    sim?.free();
    sim = next;
    generation = ++builds;
    somataSent = drivenSent = false;
    steps = 0;
    msPerStep = 0;
  };

  /** Applies `p`; a rejected value throws and is not kept. */
  const apply = (p: Params) => {
    if ('field' in p && !SD_FIELDS.some((f) => f.value === p.field)) throw new Error(`unknown field ${p.field}`);
    const old = { ...params };
    Object.assign(params, p);
    try {
      if (SD_REBUILD_KEYS.some((k) => k in p && p[k] !== old[k])) build();
      else applyRuntime(sim!, p);
    } catch (err) {
      Object.assign(params, old);
      throw err;
    }
  };
  build();
  const live = () => sim!;

  return {
    advance(budgetMs) {
      const sim = live();
      const t0 = now();
      let taken = 0;
      do {
        sim.step(1);
        taken++;
        const per = Math.max(now() - t0, 1e-3) / taken;
        msPerStep = msPerStep === 0 ? per : 0.8 * msPerStep + 0.2 * per;
      } while (budgetMs - (now() - t0) >= msPerStep);
      steps += taken;
      return true;
    },
    frame(): SolverFrame {
      const sim = live();
      const [nx, ny, nz] = sim.shape();
      const [sx, sy, sz] = sim.size();
      const v = sim.v_soma();
      const glial = SD_GLIA_FIELDS.includes(String(params.field)) && !(Number(params.glia) > 0);
      const shown = glial ? 'k' : String(params.field);
      const fields = [
        { name: shown, shape: [nx!, ny!, nz!], data: sim.field(shown) },
        { name: 'v', shape: [v.length], data: v },
      ];
      // Positions only change with a build, the driven set with the drive.
      if (!somataSent) {
        const somata = sim.somata();
        fields.push({ name: 'somata', shape: [somata.length / 3, 3], data: somata });
        somataSent = true;
      }
      if (!drivenSent) {
        const driven = Float32Array.from(sim.driven());
        fields.push({ name: 'driven', shape: [driven.length], data: driven });
        drivenSent = true;
      }
      return {
        timeMs: sim.time_ms(),
        steps,
        fields,
        stats: {
          generation,
          depolarised: sim.depolarised(),
          substeps: sim.substeps_per_neuron(),
          dt: sim.dt(),
          sizeX: sx!,
          sizeY: sy!,
          sizeZ: sz!,
        },
      };
    },
    setParams: apply,
    dispose() {
      live().free();
    },
  };
}

export const createSpreadingSolver: SolverFactory = (params) =>
  makeSpreadingSolver(
    params,
    (seed, neurons, gNap, glia, coupling) => new SdSim(seed, neurons, gNap, glia, coupling),
  );
