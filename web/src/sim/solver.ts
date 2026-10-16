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
  /** Advance the simulation, spending at most about `budgetMs` of wall-clock time. */
  advance(budgetMs: number): void;
  /** Snapshot of the fields to render. Arrays must be fresh copies (they are transferred). */
  frame(): SolverFrame;
  /** Apply changed parameters (only the keys that changed are passed). */
  setParams(params: Params): void;
  /** Perform a one-shot action triggered by a `button` control. */
  action?(name: string): void;
  dispose(): void;
}

export type SolverFactory = (params: Params) => WorkerSolver;
