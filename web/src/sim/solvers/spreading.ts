import { SD_FIELDS, SD_PARAMS, SD_REBUILD_KEYS } from '../../scales/knp/spreading-params';
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

export type SdCreate = (seed: number, neurons: number, gNap: number) => SdWasm;

/** Worker-side adapter around the wasm spreading-depression tissue. */
export function makeSpreadingSolver(
  initial: Params,
  create: SdCreate,
  now: () => number = () => performance.now(),
): WorkerSolver {
  const params: Params = { ...defaultParams(SD_PARAMS), ...initial };
  let sim: SdWasm;
  /** Bumped on every rebuild, so the view knows when the neurons moved. */
  let generation = 0;
  let geometrySent = false;
  let steps = 0;
  let msPerStep = 0;

  const applyRuntime = (p: Params) => {
    if ('driven' in p || 'drive' in p) sim.set_drive(Number(params.driven), Number(params.drive));
    if ('pump' in p) sim.set_pump(Number(p.pump));
    if ('buffering' in p) sim.set_buffering(Number(p.buffering) / 1000); // 1/s → 1/ms
  };
  const build = () => {
    const next = create(1, Number(params.neurons), Number(params.gNap));
    sim?.free();
    sim = next;
    generation++;
    geometrySent = false;
    msPerStep = 0;
    applyRuntime(params);
  };

  /** Applies `p`; a rejected value throws and is not kept. */
  const apply = (p: Params) => {
    if ('field' in p && !SD_FIELDS.some((f) => f.value === p.field)) throw new Error(`unknown field ${p.field}`);
    const old = { ...params };
    Object.assign(params, p);
    try {
      if (SD_REBUILD_KEYS.some((k) => k in p && p[k] !== old[k])) build();
      else applyRuntime(p);
    } catch (err) {
      Object.assign(params, old);
      throw err;
    }
  };
  build();

  return {
    advance(budgetMs) {
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
      const [nx, ny, nz] = sim.shape();
      const [sx, sy, sz] = sim.size();
      const v = sim.v_soma();
      const fields = [
        { name: String(params.field), shape: [nx!, ny!, nz!], data: sim.field(String(params.field)) },
        { name: 'v', shape: [v.length], data: v },
      ];
      if (!geometrySent) {
        const somata = sim.somata();
        const driven = Float32Array.from(sim.driven());
        fields.push({ name: 'somata', shape: [somata.length / 3, 3], data: somata });
        fields.push({ name: 'driven', shape: [driven.length], data: driven });
        geometrySent = true;
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
    setParams(p) {
      apply(p);
      // A new drive selects other neurons; resend them.
      if ('driven' in p) geometrySent = false;
    },
    dispose() {
      sim.free();
    },
  };
}

export const createSpreadingSolver: SolverFactory = (params) =>
  makeSpreadingSolver(params, (seed, neurons, gNap) => new SdSim(seed, neurons, gNap));
