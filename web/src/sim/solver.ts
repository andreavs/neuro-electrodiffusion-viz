import type { FrameField, Params } from './protocol';

export interface SolverFrame {
  timeMs: number;
  fields: FrameField[];
  /** Total solver steps taken so far; lets the runtime report steps/s. */
  steps?: number;
  stats?: Record<string, number>;
}

/**
 * A scale's solver as seen from inside the worker. Implementations wrap the
 * scale's wasm-bindgen struct; one lives in `solvers/<scale>.ts`.
 */
export interface WorkerSolver {
  /**
   * Advance the simulation, spending at most about `budgetMs` of wall-clock time. Return
   * `false` when nothing changed (e.g. the run has finished): the loop then idles until the
   * next parameter change or action instead of re-posting the same frame.
   *
   * Return `'wait'` when the work runs elsewhere (e.g. in helper workers) and has not
   * produced anything new yet: the loop then stops ticking until the solver calls the
   * function passed to {@link WorkerSolver.setWake}. While paused, a wake instead posts a
   * frame, so `frame()` should include results that arrived since the last `advance`.
   */
  advance(budgetMs: number): void | boolean | 'wait';
  /**
   * Called once after construction with a function that asks the loop for another tick,
   * for solvers whose `advance` can return `'wait'`. Calls after the solver is replaced are
   * ignored.
   */
  setWake?(wake: () => void): void;
  /** Snapshot of the fields to render. Arrays must be fresh copies (they are transferred). */
  frame(): SolverFrame;
  /** Apply changed parameters (only the keys that changed are passed). */
  setParams(params: Params): void;
  /** Perform a one-shot action triggered by a `button` control. */
  action?(name: string): void;
  dispose(): void;
}

export type SolverFactory = (params: Params) => WorkerSolver;
