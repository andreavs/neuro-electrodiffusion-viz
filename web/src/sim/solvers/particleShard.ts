import type { ParticlesWasm, ParticlesWasmFactory } from './particles';

/**
 * Protocol between the particle solver and its shard workers. Ions move independently, so
 * a population of n ions can be split into shards of n/K ions, each stepped by its own
 * `ParticlesSim` in its own worker, and the statistics combined exactly afterwards.
 */
export interface ShardStart {
  type: 'start';
  /** Release generation; replies to older generations are dropped. */
  gen: number;
  geometry: string;
  dt: number;
  field: string;
  /** Signed potential drop, mV. */
  dv: number;
  n: number;
  /** Zero-field reference ions (only shard 0 gets any). */
  nRef: number;
}

export interface ShardStep {
  type: 'step';
  gen: number;
  steps: number;
  /** Histogram range, ms, shared by all shards so their bins line up. */
  tMax: number;
  bins: number;
  /** Most positions to send back for drawing. */
  display: number;
}

export type ToShard = ShardStart | ShardStep;

export interface ShardRefState {
  survival: Float32Array;
  histogram: Float32Array;
  mfpt: Float64Array;
}

export interface ShardState {
  type: 'state';
  gen: number;
  /** Steps taken since the release. */
  steps: number;
  timeMs: number;
  alive: number;
  released: number;
  /** Fraction of this shard's released ions per bin. */
  histogram: Float32Array;
  /** `[mean, se, n_escaped, censored_mean]`, as `ParticlesSim.mfpt`. */
  mfpt: Float64Array;
  /** The first `display` live ions only. */
  positions: Float32Array;
  regions: Float32Array;
  /** Histogram range used, ms. */
  tMax: number;
  /** Measured wall-clock cost of one step, ms (0 before the first step). */
  msPerStep: number;
  ref?: ShardRefState;
  /** Wall profile and neck size, sent with the reply to `start`. */
  profile?: Float32Array;
  neckTop?: number;
  neckRadius?: number;
}

export interface ShardError {
  type: 'error';
  message: string;
}

export type FromShard = ShardState | ShardError;

/** Arrays in a shard reply, for the transfer list. */
export function shardTransferables(s: ShardState): ArrayBuffer[] {
  const arrays = [s.histogram, s.mfpt, s.positions, s.regions, s.profile];
  if (s.ref) arrays.push(s.ref.survival, s.ref.histogram, s.ref.mfpt);
  return arrays.filter((a): a is NonNullable<typeof a> => !!a).map((a) => a.buffer as ArrayBuffer);
}

const PROFILE_SEGMENTS = 48;
/** Steps per wasm call; the loop runs all requested steps (the coordinator sizes requests). */
const BATCH = 64;

/** Shard-side state machine, free of Worker APIs; the shard worker is a thin adapter. */
export class ShardRunner {
  private sim: ParticlesWasm | null = null;
  private geometry = '';
  private gen = -1;
  private steps = 0;
  private msPerStep = 0;
  private bins = 1;
  private tMax = 1;

  constructor(
    private readonly create: ParticlesWasmFactory,
    private readonly seed: number,
    private readonly now: () => number = () => performance.now(),
  ) {}

  handle(msg: ToShard): ShardState | null {
    if (msg.type === 'start') return this.start(msg);
    if (msg.gen !== this.gen || !this.sim) return null;
    const t0 = this.now();
    let taken = 0;
    while (taken < msg.steps) {
      const k = this.sim.step(Math.min(BATCH, msg.steps - taken));
      if (k === 0) break;
      taken += k;
    }
    if (taken > 0) {
      const per = Math.max(this.now() - t0, 1e-3) / taken;
      this.msPerStep = this.msPerStep === 0 ? per : 0.7 * this.msPerStep + 0.3 * per;
    }
    this.steps += taken;
    this.bins = msg.bins;
    this.tMax = msg.tMax;
    return this.state(msg.display, false);
  }

  dispose(): void {
    this.sim?.free();
    this.sim = null;
  }

  private start(msg: ShardStart): ShardState {
    if (!this.sim || msg.geometry !== this.geometry) {
      this.sim?.free();
      this.sim = null; // stays null if `create` throws
      this.sim = this.create(msg.geometry, msg.dt, this.seed);
      this.geometry = msg.geometry;
    }
    this.sim.set_dt(msg.dt);
    this.sim.set_field(msg.field, msg.dv);
    this.sim.release(msg.n, msg.nRef);
    this.gen = msg.gen;
    this.steps = 0;
    this.bins = 1;
    this.tMax = 1;
    this.msPerStep = 0;
    return this.state(0, true);
  }

  private state(display: number, withProfile: boolean): ShardState {
    const sim = this.sim!;
    const all = sim.positions();
    const m = Math.min(display, all.length / 3);
    const s: ShardState = {
      type: 'state',
      gen: this.gen,
      steps: this.steps,
      timeMs: sim.time_ms(),
      alive: sim.n_alive(),
      released: sim.n_released(),
      histogram: sim.histogram(false, this.bins, this.tMax),
      mfpt: sim.mfpt(false),
      positions: all.slice(0, 3 * m),
      regions: sim.regions().slice(0, m),
      tMax: this.tMax,
      msPerStep: this.msPerStep,
    };
    if (sim.has_reference()) {
      s.ref = {
        survival: sim.survival(true),
        histogram: sim.histogram(true, this.bins, this.tMax),
        mfpt: sim.mfpt(true),
      };
    }
    if (withProfile) {
      s.profile = sim.profile(PROFILE_SEGMENTS);
      s.neckTop = sim.neck_top();
      s.neckRadius = sim.neck_radius();
    }
    return s;
  }
}
