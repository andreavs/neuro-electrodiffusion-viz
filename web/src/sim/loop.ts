import type { ScaleId } from '../scales/registry';
import type { FromWorker, Params, RuntimeStats, ToWorker } from './protocol';
import type { SolverFactory, WorkerSolver } from './solver';

/** Per-tick solver budget, leaving headroom in a 16.7 ms frame (see CONTRIBUTING). */
export const DEFAULT_BUDGET_MS = 12;
export const MAX_BUDGET_MS = 50;

/** Environment the loop runs in; the worker passes real timers, tests pass fakes. */
export interface LoopHost {
  post(msg: FromWorker): void;
  /** Run `fn` as a new macrotask so incoming messages can interleave; returns a cancel function. */
  schedule(fn: () => void): () => void;
  now(): number;
}

export interface RunLoopOptions {
  budgetMs?: number;
  /**
   * Ticks the solver may run ahead while a frame is unacknowledged. Keeps the
   * solver busy between frames, but stops it when the UI stops consuming
   * frames (e.g. a hidden tab pauses requestAnimationFrame).
   */
  maxTicksAhead?: number;
  /** Window over which steps/s and sim-ms/s are measured. */
  rateWindowMs?: number;
}

/**
 * Owns one solver and drives it: ticks while running, posts at most one
 * frame at a time, and applies control messages. Free of Worker APIs so it is
 * unit-testable; `worker.ts` is a thin adapter.
 */
export class RunLoop {
  private solver: WorkerSolver | null = null;
  private scale: ScaleId | null = null;
  private params: Params = {};
  private running = false;
  private budgetMs: number;
  private readonly maxTicksAhead: number;
  private readonly rateWindowMs: number;

  private cancelTick: (() => void) | null = null;
  private seq = 0;
  private awaitingAck = false;
  private ticksAhead = 0;
  private dirty = false;

  private solverMs = 0;
  private rate = { t: 0, simMs: 0, steps: 0, stepsPerSec: 0, simMsPerSec: 0 };

  constructor(
    private readonly host: LoopHost,
    private readonly factories: Record<ScaleId, SolverFactory>,
    opts: RunLoopOptions = {},
  ) {
    this.budgetMs = opts.budgetMs ?? DEFAULT_BUDGET_MS;
    this.maxTicksAhead = opts.maxTicksAhead ?? 2;
    this.rateWindowMs = opts.rateWindowMs ?? 500;
  }

  /** Handle every message except `init`, which needs the WASM module. */
  handle(msg: ToWorker): void {
    try {
      switch (msg.type) {
        case 'init':
          break;
        case 'create':
          this.scale = msg.scale;
          this.params = { ...msg.params };
          this.build();
          break;
        case 'reset':
          if (this.scale) this.build();
          break;
        case 'run':
          this.running = msg.running;
          this.resetRates();
          if (this.running) this.scheduleTick();
          else this.stopTicking();
          this.dirty = this.solver !== null; // publish the new running state
          this.flush();
          break;
        case 'setParams':
          Object.assign(this.params, msg.params);
          if (this.solver) {
            this.solver.setParams(msg.params);
            this.dirty = true;
            this.flush();
          }
          break;
        case 'action':
          if (this.solver?.action) {
            this.solver.action(msg.name);
            this.dirty = true;
            this.flush();
          }
          break;
        case 'setBudget':
          this.budgetMs = clampBudget(msg.budgetMs);
          break;
        case 'ack':
          if (msg.seq !== this.seq || !this.awaitingAck) break;
          this.awaitingAck = false;
          this.ticksAhead = 0;
          this.flush();
          if (this.running) this.scheduleTick();
          break;
        case 'dispose':
          this.dispose();
          break;
      }
    } catch (err) {
      this.fail(err);
    }
  }

  /** Stop ticking and free the solver. Safe to call repeatedly. */
  dispose(): void {
    this.stopTicking();
    this.running = false;
    this.solver?.dispose();
    this.solver = null;
    this.scale = null;
  }

  get isRunning(): boolean {
    return this.running;
  }

  get hasPendingTick(): boolean {
    return this.cancelTick !== null;
  }

  private build(): void {
    this.solver?.dispose();
    this.solver = null;
    this.solver = this.factories[this.scale!](this.params);
    this.solverMs = 0;
    this.resetRates();
    // Whatever frame was in flight belongs to the old solver and may never be
    // acked (the client drops it on create); its late ack is ignored by seq.
    this.awaitingAck = false;
    this.ticksAhead = 0;
    this.dirty = true;
    this.flush();
    if (this.running) this.scheduleTick();
  }

  private scheduleTick(): void {
    if (this.cancelTick || !this.solver) return;
    this.cancelTick = this.host.schedule(() => {
      this.cancelTick = null;
      this.tick();
    });
  }

  private stopTicking(): void {
    this.cancelTick?.();
    this.cancelTick = null;
  }

  private tick(): void {
    if (!this.running || !this.solver) return;
    // Ran far enough ahead of the renderer: idle until the next ack.
    if (this.awaitingAck && this.ticksAhead >= this.maxTicksAhead) return;
    try {
      const t0 = this.host.now();
      this.solver.advance(this.budgetMs);
      const spent = this.host.now() - t0;
      this.solverMs = this.solverMs === 0 ? spent : 0.9 * this.solverMs + 0.1 * spent;
      this.dirty = true;
      if (this.awaitingAck) this.ticksAhead++;
      this.flush();
    } catch (err) {
      this.fail(err);
      return;
    }
    this.scheduleTick();
  }

  /** Post a frame if there is something new and the previous frame was acknowledged. */
  private flush(): void {
    if (!this.dirty || this.awaitingAck || !this.solver) return;
    const f = this.solver.frame();
    const seq = this.seq + 1;
    // Post first: if it throws (e.g. DataCloneError) no frame is in flight,
    // so we must not wait for an ack.
    this.post({
      type: 'frame',
      seq,
      timeMs: f.timeMs,
      fields: f.fields,
      stats: f.stats,
      runtime: this.runtimeStats(f.timeMs, f.steps ?? 0),
    });
    this.seq = seq;
    this.awaitingAck = true;
    this.dirty = false;
  }

  private runtimeStats(simMs: number, steps: number): RuntimeStats {
    const now = this.host.now();
    const r = this.rate;
    if (!this.running) {
      r.stepsPerSec = 0;
      r.simMsPerSec = 0;
    } else if (r.t === 0) {
      Object.assign(r, { t: now, simMs, steps });
    } else if (now - r.t >= this.rateWindowMs) {
      const dt = (now - r.t) / 1000;
      r.stepsPerSec = Math.max(0, (steps - r.steps) / dt);
      r.simMsPerSec = Math.max(0, (simMs - r.simMs) / dt);
      Object.assign(r, { t: now, simMs, steps });
    }
    return {
      running: this.running,
      stepsPerSec: r.stepsPerSec,
      simMsPerSec: r.simMsPerSec,
      solverMs: this.solverMs,
    };
  }

  private resetRates(): void {
    this.rate = { t: 0, simMs: 0, steps: 0, stepsPerSec: 0, simMsPerSec: 0 };
  }

  private post(msg: FromWorker): void {
    this.host.post(msg);
  }

  private fail(err: unknown): void {
    this.running = false;
    this.stopTicking();
    this.post({ type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
}

export function clampBudget(ms: number): number {
  return Number.isFinite(ms) ? Math.min(MAX_BUDGET_MS, Math.max(0.5, ms)) : DEFAULT_BUDGET_MS;
}

