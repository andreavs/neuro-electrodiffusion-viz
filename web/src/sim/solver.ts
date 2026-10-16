import type { FrameField } from './protocol';

/**
 * A scale's solver as seen from inside the worker. Implementations wrap the
 * scale's wasm-bindgen struct; one lives in `solvers/<scale>.ts`.
 */
export interface WorkerSolver {
  /** Advance the simulation, spending at most about `budgetMs` of wall-clock time. */
  advance(budgetMs: number): void;
  /** Snapshot of the fields to render. Arrays must be fresh copies (they are transferred). */
  frame(): { timeMs: number; fields: FrameField[]; stats?: Record<string, number> };
  setParams(params: Record<string, unknown>): void;
  dispose(): void;
}

export type SolverFactory = (params: Record<string, unknown>) => WorkerSolver;
