import type { Params } from '../protocol';
import type { SolverFrame, WorkerSolver } from '../solver';

/**
 * Deterministic solver for tests: each `advance` takes `stepsPerAdvance`
 * steps of `dt` ms of simulated time, and burns `costMs` on the injected clock.
 */
export class FakeSolver implements WorkerSolver {
  timeMs = 0;
  steps = 0;
  advances = 0;
  budgets: number[] = [];
  params: Params;
  actions: string[] = [];
  disposed = false;
  failOnAdvance = false;
  /** When true, `advance` reports that nothing changed. */
  finished = false;

  constructor(
    params: Params,
    private readonly clock?: { t: number },
    private readonly costMs = 0,
  ) {
    this.params = { ...params };
  }

  get dt(): number {
    return typeof this.params.dt === 'number' ? this.params.dt : 0.1;
  }

  advance(budgetMs: number): boolean {
    if (this.failOnAdvance) throw new Error('boom');
    this.advances++;
    this.budgets.push(budgetMs);
    if (this.finished) return false;
    const n = typeof this.params.stepsPerAdvance === 'number' ? this.params.stepsPerAdvance : 10;
    this.steps += n;
    this.timeMs += n * this.dt;
    if (this.clock) this.clock.t += this.costMs;
    return true;
  }

  frame(): SolverFrame {
    return {
      timeMs: this.timeMs,
      steps: this.steps,
      fields: [{ name: 'u', shape: [2], data: new Float32Array([this.timeMs, this.steps]) }],
      stats: { advances: this.advances },
    };
  }

  setParams(params: Params): void {
    Object.assign(this.params, params);
  }

  action(name: string): void {
    this.actions.push(name);
  }

  dispose(): void {
    this.disposed = true;
  }
}
