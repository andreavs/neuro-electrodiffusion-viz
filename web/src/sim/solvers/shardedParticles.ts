import { PARTICLE_PARAMS, REFERENCE_PARTICLES, signedDrop } from '../../scales/particles/params';
import { defaultParams } from '../../ui/controls';
import type { FrameField, Params } from '../protocol';
import type { SolverFrame, WorkerSolver } from '../solver';
import type { FromShard, ShardState, ToShard } from './particleShard';
import { HISTOGRAM_BINS, niceCeil } from './particles';
import { mergeHistograms, mergeMfpt, shardCount, splitEvenly } from './shardMerge';

/** One shard worker as the coordinator sees it. */
export interface ShardPort {
  post(msg: ToShard): void;
  onMessage(cb: (msg: FromShard) => void): void;
  terminate(): void;
}

/** Ions drawn at most, over all shards; the rest are simulated but not sent to the view. */
export const DISPLAY_MAX = 100_000;
const MAX_STEPS_PER_ROUND = 4096;
const SURVIVAL_CAPACITY = 2048;
/** True once every ion of a round, reference included, has escaped (and left the dendrite). */
function finished(round: ShardState[], nRef: number): boolean {
  const ref = round[0]?.ref;
  return round.length > 0 && round.every((s) => s.alive === 0 && s.inDendrite === 0) && !(ref && ref.mfpt[2]! < nRef);
}

/** Latest last-exit time over the shards, ms. */
function passageTime(round: ShardState[]): number {
  return round.reduce((a, s) => Math.max(a, s.passageTime), 0);
}

/** A copy of `a` zero-padded (or cut) to length `n`. */
function padTo(a: Float32Array, n: number): Float32Array {
  const out = new Float32Array(n);
  out.set(a.subarray(0, n));
  return out;
}

/** Concatenates typed arrays. */
function concat(parts: Float32Array[]): Float32Array {
  const out = new Float32Array(parts.reduce((a, p) => a + p.length, 0));
  let k = 0;
  for (const p of parts) {
    out.set(p, k);
    k += p.length;
  }
  return out;
}

/**
 * The particle solver spread over shard workers. Each round, every shard takes the same
 * number of steps; once all have replied the round is merged into the next frame. The loop
 * gets `'wait'` while a round is out and is woken when it completes.
 */
export class ShardedParticlesSolver implements WorkerSolver {
  private readonly params: Params;
  private readonly ports: ShardPort[] = [];
  private active = 0;
  private gen = 0;
  private replies: (ShardState | undefined)[] = [];
  private pending = 0;
  /** Completed rounds not yet merged (at most two: the shards run one round ahead). */
  private done: ShardState[][] = [];
  private budgetMs = 12;
  /**
   * The loop has asked for steps since the last round sent from `receive`, so a completed
   * round may be followed by the next one straight away. Without this a paused run would
   * keep stepping (paused wakes only call `frame`).
   */
  private ahead = false;
  private merged: ShardState[] = [];
  private survival: number[] = [0, 1];
  private survivalEvery = 1;
  private rounds = 0;
  private profile: {
    data: Float32Array;
    neckTop: number;
    neckRadius: number;
    dendriteRadius: number;
    dendriteHalfLength: number;
  } | null = null;
  private nRef = 0;
  /** Histogram range of the merged round. */
  private histTMax = 1;
  private error: Error | null = null;
  private wake: () => void = () => {};

  constructor(
    initial: Params,
    private readonly spawn: (index: number) => ShardPort,
    private readonly cores: number,
  ) {
    this.params = { ...defaultParams(PARTICLE_PARAMS), ...initial };
    this.restart();
  }

  setWake(wake: () => void): void {
    this.wake = wake;
  }

  advance(budgetMs: number): boolean | 'wait' {
    if (this.error) throw this.error;
    this.budgetMs = budgetMs;
    this.ahead = true;
    const merged = this.drain();
    if (this.pending === 0 && !finished(this.merged, this.nRef)) this.sendRound();
    if (merged) return true;
    return this.pending > 0 ? 'wait' : false;
  }

  frame(): SolverFrame {
    // Also called without `advance` when a paused solver wakes the loop (e.g. a new release).
    this.drain();
    const states = this.merged;
    const p = this.profile;
    const timeMs = states.reduce((a, s) => Math.max(a, s.timeMs), 0);
    const released = states.reduce((a, s) => a + s.released, 0);
    const alive = states.reduce((a, s) => a + s.alive, 0);
    const positions = concat(states.map((s) => s.positions));
    const regions = concat(states.map((s) => s.regions));
    const survival = Float32Array.from(this.survival);
    const fields: FrameField[] = [
      { name: 'positions', shape: [positions.length / 3, 3], data: positions },
      { name: 'regions', shape: [regions.length], data: regions },
      { name: 'survival', shape: [survival.length / 2, 2], data: survival },
      { name: 'histogram', shape: [HISTOGRAM_BINS], data: mergeHistograms(states, HISTOGRAM_BINS) },
    ];
    if (p) fields.push({ name: 'profile', shape: [p.data.length / 2, 2], data: p.data.slice() });
    const inside = states.reduce((a, s) => (s.mfpt[2]! > 0 ? a : a + s.alive * s.timeMs), 0);
    const m = mergeMfpt(
      states.map((s) => s.mfpt),
      inside,
    );
    const ref = states[0]?.ref;
    const stats: Record<string, number> = {
      alive,
      inDendrite: states.reduce((a, s) => a + s.inDendrite, 0),
      passageTime: passageTime(states),
      released,
      histTMax: this.histTMax,
      hasReference: ref ? 1 : 0,
      refReleased: ref ? this.nRef : 0,
      neckTop: p?.neckTop ?? 1,
      neckRadius: p?.neckRadius ?? 0.1,
      dendriteRadius: p?.dendriteRadius ?? 0,
      dendriteHalfLength: p?.dendriteHalfLength ?? 0,
      workers: this.active,
      mfpt: m[0]!,
      mfptSe: m[1]!,
      escaped: m[2]!,
      mfptCensored: m[3]!,
    };
    if (ref) {
      fields.push(
        { name: 'survivalRef', shape: [ref.survival.length / 2, 2], data: ref.survival.slice() },
        // Padded: the reply to a release carries a 1-bin histogram.
        { name: 'histogramRef', shape: [HISTOGRAM_BINS], data: padTo(ref.histogram, HISTOGRAM_BINS) },
      );
      Object.assign(stats, { refMfpt: ref.mfpt[0], refMfptSe: ref.mfpt[1], refEscaped: ref.mfpt[2], refMfptCensored: ref.mfpt[3] });
    }
    return { timeMs, fields, steps: states[0]?.steps ?? 0, stats };
  }

  setParams(changed: Params): void {
    Object.assign(this.params, changed);
    this.restart();
  }

  action(name: string): void {
    if (name === 'release') this.restart();
  }

  dispose(): void {
    for (const p of this.ports) p.terminate();
    this.ports.length = 0;
    this.pending = 0;
  }

  /** Releases a fresh population over as many shards as it needs. */
  private restart(): void {
    const n = Number(this.params.particles);
    this.active = shardCount(n, this.cores);
    while (this.ports.length < this.active) this.addPort();
    this.gen++;
    this.nRef = Math.min(n, REFERENCE_PARTICLES);
    this.survival = [0, 1];
    this.survivalEvery = 1;
    this.rounds = 0;
    this.merged = [];
    this.done = [];
    this.histTMax = 1;
    this.profile = null; // the shape may change; the start replies bring the new one
    const counts = splitEvenly(n, this.active);
    const geometry = String(this.params.geometry);
    const dv = signedDrop(Number(this.params.dv), String(this.params.direction));
    this.collect(this.active);
    this.ahead = false;
    this.error = null;
    counts.forEach((count, i) =>
      this.ports[i]!.post({
        type: 'start',
        gen: this.gen,
        geometry,
        dendrite: this.params.outlet === 'dendrite',
        dt: Number(this.params.dt),
        field: String(this.params.field),
        dv,
        n: count,
        nRef: i === 0 ? this.nRef : 0,
      }),
    );
    // Shards left over from a larger run go, with their memory (a wasm heap never shrinks).
    for (const p of this.ports.splice(this.active)) p.terminate();
  }

  private addPort(): void {
    const index = this.ports.length;
    const port = this.spawn(index);
    port.onMessage((msg) => this.receive(index, msg));
    this.ports.push(port);
  }

  private collect(k: number): void {
    this.replies = new Array<ShardState | undefined>(k);
    this.pending = k;
  }

  private receive(index: number, msg: FromShard): void {
    if (msg.type === 'error') {
      if (msg.gen !== undefined && msg.gen !== this.gen) return;
      this.error ??= new Error(`particle shard ${index}: ${msg.message}`);
      this.wake();
      return;
    }
    if (msg.gen !== this.gen || index >= this.active || this.replies[index]) return;
    this.replies[index] = msg;
    if (msg.profile)
      this.profile = {
        data: msg.profile,
        neckTop: msg.neckTop ?? 1,
        neckRadius: msg.neckRadius ?? 0.1,
        dendriteRadius: msg.dendriteRadius ?? 0,
        dendriteHalfLength: msg.dendriteHalfLength ?? 0,
      };
    if (--this.pending === 0) {
      const round = this.replies as ShardState[];
      this.done.push(round);
      // Keep the shards busy while the loop merges this round, but no further ahead.
      if (this.ahead && this.done.length < 2 && !finished(round, this.nRef)) {
        this.ahead = false;
        this.sendRound();
      }
      this.wake();
    }
  }

  /** Merges every completed round, oldest first; true if there were any. */
  private drain(): boolean {
    const any = this.done.length > 0;
    for (const round of this.done.splice(0)) this.mergeRound(round);
    return any;
  }

  private mergeRound(round: ShardState[]): void {
    this.merged = round;
    this.histTMax = round[0]?.tMax ?? 1;
    const alive = this.merged.reduce((a, s) => a + s.alive, 0);
    const released = this.merged.reduce((a, s) => a + s.released, 0);
    // Survival is sampled up to the last exit, not while escaped ions spread in the dendrite.
    const t = passageTime(this.merged);
    // The round where the last ion leaves is always sampled, so the curve ends at 0.
    const last = finished(round, this.nRef) || alive === 0;
    // `t` can keep growing with the reference's clock, so stop once the curve has reached 0.
    const open = this.survival[this.survival.length - 1]! > 0;
    if ((this.rounds++ % this.survivalEvery === 0 || last) && open && t > this.survival[this.survival.length - 2]!) {
      // Thin before pushing, so the newest sample (maybe the last exit) is kept.
      if (this.survival.length / 2 + 1 >= SURVIVAL_CAPACITY) {
        // Keep t = 0 and every other sample; sample half as often from now on.
        this.survival = this.survival.filter((_, i) => i % 4 < 2);
        this.survivalEvery *= 2;
      }
      this.survival.push(t, released > 0 ? alive / released : 0);
    }
  }

  private sendRound(): void {
    // The newest state the shards have reported: the last completed round, else the merged one.
    const latest = this.done[this.done.length - 1] ?? this.merged;
    const msPerStep = latest.reduce((a, s) => Math.max(a, s.msPerStep), 0);
    const steps = msPerStep > 0 ? Math.max(1, Math.min(MAX_STEPS_PER_ROUND, Math.floor(this.budgetMs / msPerStep))) : 4;
    // Exit-time axes end at the last exit, not when the last ion leaves the dendrite.
    // Pad while any ion (main or reference) is still inside; the passage clocks then run on.
    const exiting = latest.length === 0 || latest.some((s) => s.passageTime >= s.timeMs);
    const tMax = niceCeil(passageTime(latest) + (exiting ? steps * Number(this.params.dt) : 0));
    const display = Math.ceil(DISPLAY_MAX / this.active);
    this.collect(this.active);
    for (let i = 0; i < this.active; i++)
      this.ports[i]!.post({ type: 'step', gen: this.gen, steps, tMax, bins: HISTOGRAM_BINS, display });
  }
}
