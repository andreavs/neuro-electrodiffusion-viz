import { PARTICLE_PARAMS, REFERENCE_PARTICLES, signedDrop } from '../../scales/particles/params';
import { defaultParams } from '../../ui/controls';
import { ParticlesSim } from '../../wasm/pkg/ed_wasm.js';
import type { FrameField, Params } from '../protocol';
import type { SolverFactory, SolverFrame, WorkerSolver } from '../solver';
import type { FromShard } from './particleShard';
import { ShardedParticlesSolver, type ShardPort } from './shardedParticles';

/** The parts of the wasm-bindgen `ParticlesSim` the adapter uses, so tests can pass a fake. */
export interface ParticlesWasm {
  set_field(kind: string, deltaVMv: number): void;
  set_dt(dt: number): void;
  release(n: number, nReference: number): void;
  step(n: number): number;
  time_ms(): number;
  n_alive(): number;
  n_released(): number;
  has_reference(): boolean;
  positions(): Float32Array;
  regions(): Float32Array;
  survival(reference: boolean): Float32Array;
  histogram(reference: boolean, bins: number, tMax: number): Float32Array;
  mfpt(reference: boolean): Float64Array;
  profile(segments: number): Float32Array;
  neck_top(): number;
  neck_radius(): number;
  free(): void;
}

export type ParticlesWasmFactory = (geometry: string, dt: number, seed: number) => ParticlesWasm;

export const HISTOGRAM_BINS = 40;
/** Segments of the head arc in the wall profile sent to the view. */
const PROFILE_SEGMENTS = 48;
/** Steps per wasm call are capped so a slow step cannot overrun the budget by much. */
const MAX_BATCH = 256;

/** Smallest 1, 2 or 5 × 10ᵏ that is ≥ x (x > 0), so the histogram axis changes in steps. */
export function niceCeil(x: number): number {
  if (!(x > 0)) return 1;
  const p = 10 ** Math.floor(Math.log10(x));
  for (const m of [1, 2, 5, 10]) if (m * p >= x * (1 - 1e-12)) return m * p;
  return 10 * p;
}

/** Steps to request next: what fits in `remainingMs` at the measured cost, at least one. */
export function batchSize(remainingMs: number, msPerStep: number): number {
  if (!(msPerStep > 0)) return 1;
  return Math.max(1, Math.min(MAX_BATCH, Math.floor(remainingMs / msPerStep)));
}

/** Flattens `[mean, se, n, censored]` into frame stats under `prefix`. */
function mfptStats(prefix: string, m: Float64Array): Record<string, number> {
  const key = (name: string) => (prefix ? prefix + name : name[0]!.toLowerCase() + name.slice(1));
  return {
    [key('Mfpt')]: m[0]!,
    [key('MfptSe')]: m[1]!,
    [key('Escaped')]: m[2]!,
    [key('MfptCensored')]: m[3]!,
  };
}

/** Packs the current state of `sim` into a frame. Every array is a fresh copy (transferable). */
export function packFrame(sim: ParticlesWasm, steps: number, nReference: number): SolverFrame {
  const time = sim.time_ms();
  const tMax = niceCeil(time);
  const positions = sim.positions();
  const n = positions.length / 3;
  const fields: FrameField[] = [
    { name: 'positions', shape: [n, 3], data: positions },
    { name: 'regions', shape: [n], data: sim.regions() },
  ];
  const profile = sim.profile(PROFILE_SEGMENTS);
  fields.push({ name: 'profile', shape: [profile.length / 2, 2], data: profile });
  const curves: [string, boolean][] = [['', false]];
  if (sim.has_reference()) curves.push(['Ref', true]);
  let stats: Record<string, number> = {
    alive: sim.n_alive(),
    released: sim.n_released(),
    histTMax: tMax,
    hasReference: sim.has_reference() ? 1 : 0,
    refReleased: sim.has_reference() ? nReference : 0,
    neckTop: sim.neck_top(),
    neckRadius: sim.neck_radius(),
  };
  for (const [suffix, ref] of curves) {
    const s = sim.survival(ref);
    fields.push(
      { name: `survival${suffix}`, shape: [s.length / 2, 2], data: s },
      { name: `histogram${suffix}`, shape: [HISTOGRAM_BINS], data: sim.histogram(ref, HISTOGRAM_BINS, tMax) },
    );
    stats = { ...stats, ...mfptStats(ref ? 'ref' : '', sim.mfpt(ref)) };
  }
  return { timeMs: time, fields, steps, stats };
}

/** Worker-side adapter around the wasm particle simulation. */
export function makeParticlesSolver(
  initial: Params,
  create: ParticlesWasmFactory,
  now: () => number = () => performance.now(),
): WorkerSolver {
  const params: Params = { ...defaultParams(PARTICLE_PARAMS), ...initial };
  let seed = 1;
  const build = () => create(String(params.geometry), Number(params.dt), seed++);
  let sim = build();
  let steps = 0;
  let msPerStep = 0;
  let nReference = 0;

  /** Applies the parameters and releases a fresh population. */
  const restart = () => {
    sim.set_dt(Number(params.dt));
    sim.set_field(String(params.field), signedDrop(Number(params.dv), String(params.direction)));
    // One wasm population holds at most a million ions; more need the sharded solver.
    const n = Math.min(Number(params.particles), 1_000_000);
    nReference = Math.min(n, REFERENCE_PARTICLES);
    sim.release(n, nReference);
    steps = 0;
    msPerStep = 0;
  };
  restart();

  return {
    advance(budgetMs) {
      const t0 = now();
      let taken = 0;
      // Cost estimate for this tick: the running average, or this tick's own once it has one.
      let est = msPerStep;
      for (;;) {
        const remaining = budgetMs - (now() - t0);
        // Stop before a step that would overrun the budget (but always take at least one).
        if (taken > 0 && (remaining <= 0 || remaining < est)) break;
        const n = sim.step(batchSize(remaining, est));
        if (n === 0) break; // every ion has escaped
        taken += n;
        if (msPerStep === 0) est = Math.max(now() - t0, 1e-3) / taken;
      }
      if (taken === 0) return false;
      steps += taken;
      // Averaged over the whole tick: coarse timers can read 0 for a single fast batch.
      const per = Math.max(now() - t0, 1e-3) / taken;
      msPerStep = msPerStep === 0 ? per : 0.7 * msPerStep + 0.3 * per;
      return true;
    },
    frame: () => packFrame(sim, steps, nReference),
    setParams(changed) {
      Object.assign(params, changed);
      if ('geometry' in changed) {
        sim.free();
        sim = build();
      }
      // Any physical change starts a new release, so the statistics stay meaningful.
      restart();
    },
    action(name) {
      if (name === 'release') restart();
    },
    dispose() {
      sim.free();
    },
  };
}

/** Shard workers for the particle solver, one per spare core; inside the solver's own worker. */
function spawnShard(index: number): ShardPort {
  const w = new Worker(new URL('../particleShard.worker.ts', import.meta.url), {
    type: 'module',
    name: String(7919 * (index + 1)),
  });
  return {
    post: (msg) => w.postMessage(msg),
    onMessage: (cb) => {
      w.onmessage = (ev: MessageEvent<FromShard>) => cb(ev.data);
      w.onerror = (ev) => cb({ type: 'error', message: ev.message || 'shard worker failed to load' });
    },
    terminate: () => w.terminate(),
  };
}

/** Spreads the ions over shard workers where nested workers exist; otherwise runs them here. */
export const createParticlesSolver: SolverFactory = (params) =>
  typeof Worker === 'function'
    ? new ShardedParticlesSolver(params, spawnShard, navigator.hardwareConcurrency || 4)
    : makeParticlesSolver(params, (geometry, dt, seed) => new ParticlesSim(geometry, dt, seed));
