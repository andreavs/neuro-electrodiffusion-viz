import { KNP_PARAMS, decodeProbes } from '../../scales/knp/params';
import { defaultParams } from '../../ui/controls';
import { KnpSim } from '../../wasm/pkg/ed_wasm.js';
import type { Params } from '../protocol';
import type { SolverFactory, SolverFrame, WorkerSolver } from '../solver';

/** The parts of the wasm-bindgen `KnpSim` the adapter uses, so tests can pass a fake. */
export interface KnpWasm {
  set_syn_rate(hz: number): void;
  set_stim(nA: number): void;
  set_dt(dt: number): void;
  dt(): number;
  step(n: number): void;
  time_ms(): number;
  spikes(): number;
  shape(): Uint32Array;
  size(): Float64Array;
  field(name: string): Float32Array;
  v(): Float32Array;
  compartments(): Float32Array;
  take_trace(): Float64Array;
  probe(xyz: Float64Array): Float32Array;
  free(): void;
}

/** Somatic trace samples kept between frames (oldest dropped first). */
export const MAX_TRACE_SAMPLES = 8192;

/** Collects somatic (t, V) pairs between frames, keeping the newest `capacity`. */
export class TraceBuffer {
  private chunks: Float64Array[] = [];
  private pairs = 0;

  constructor(private readonly capacity = MAX_TRACE_SAMPLES) {}

  push(pairs: Float64Array): void {
    if (pairs.length === 0) return;
    this.chunks.push(pairs);
    this.pairs += pairs.length / 2;
    while (this.pairs - this.chunks[0]!.length / 2 >= this.capacity) {
      this.pairs -= this.chunks.shift()!.length / 2;
    }
  }

  /** The newest samples as a fresh Float32Array of (t, V) pairs; empties the buffer. */
  take(): Float32Array {
    const n = Math.min(this.pairs, this.capacity);
    const out = new Float32Array(2 * n);
    let o = out.length;
    for (let c = this.chunks.length - 1; c >= 0 && o > 0; c--) {
      const chunk = this.chunks[c]!;
      const k = Math.min(chunk.length, o);
      out.set(chunk.subarray(chunk.length - k), o - k);
      o -= k;
    }
    this.chunks = [];
    this.pairs = 0;
    return out;
  }
}

/** Packs the current state into a frame. Every array is a fresh copy (transferable). */
export function packFrame(
  sim: KnpWasm,
  field: string,
  probes: Float64Array,
  trace: Float32Array,
  steps: number,
): SolverFrame {
  const [nx, ny, nz] = sim.shape();
  const [sx, sy, sz] = sim.size();
  const v = sim.v();
  const geometry = sim.compartments();
  const probeValues = sim.probe(probes);
  return {
    timeMs: sim.time_ms(),
    steps,
    fields: [
      { name: field, shape: [nx!, ny!, nz!], data: sim.field(field) },
      { name: 'v', shape: [v.length], data: v },
      { name: 'compartments', shape: [geometry.length / 7, 7], data: geometry },
      { name: 'trace', shape: [trace.length / 2, 2], data: trace },
      { name: 'probes', shape: [probes.length / 3, probeValues.length / Math.max(1, probes.length / 3)], data: probeValues },
    ],
    stats: { spikes: sim.spikes(), dt: sim.dt(), sizeX: sx!, sizeY: sy!, sizeZ: sz! },
  };
}

/** Worker-side adapter around the wasm tissue simulation. */
export function makeKnpSolver(
  initial: Params,
  create: (seed: number) => KnpWasm,
  now: () => number = () => performance.now(),
): WorkerSolver {
  const params: Params = { field: 'k', probes: '', ...defaultParams(KNP_PARAMS), ...initial };
  const sim = create(1);
  const trace = new TraceBuffer();
  let probes = new Float64Array(0);
  let steps = 0;
  let msPerStep = 0;

  const apply = (p: Params) => {
    if ('synRate' in p) sim.set_syn_rate(Number(p.synRate));
    if ('stim' in p) sim.set_stim(Number(p.stim));
    if ('dt' in p) sim.set_dt(Number(p.dt));
    if ('probes' in p) probes = Float64Array.from(decodeProbes(String(p.probes)));
  };
  apply(params);

  return {
    advance(budgetMs) {
      const t0 = now();
      let taken = 0;
      // Always one step; more only while the measured cost still fits the budget.
      do {
        sim.step(1);
        taken++;
        trace.push(sim.take_trace());
        const per = Math.max(now() - t0, 1e-3) / taken;
        msPerStep = msPerStep === 0 ? per : 0.8 * msPerStep + 0.2 * per;
      } while (budgetMs - (now() - t0) >= msPerStep);
      steps += taken;
      return true;
    },
    frame: () => packFrame(sim, String(params.field), probes, trace.take(), steps),
    setParams(changed) {
      Object.assign(params, changed);
      apply(changed);
    },
    dispose() {
      sim.free();
    },
  };
}

export const createKnpSolver: SolverFactory = (params) => makeKnpSolver(params, (seed) => new KnpSim(seed));
