import type { ScaleId } from '../scales/registry';

/**
 * Messages between the UI thread and the simulation worker.
 *
 * Convention: the worker owns the solver and advances it with a wall-clock
 * budget per tick; it posts `frame` messages whose typed arrays are
 * transferred (not copied). At most one frame is in flight: the worker waits
 * for an `ack` carrying the frame's `seq` before posting the next one, and the
 * UI acks when its renderer takes the frame on requestAnimationFrame. A slow
 * renderer (or a hidden tab) therefore throttles the worker instead of letting
 * frames queue up.
 */
export type ToWorker =
  | { type: 'init' }
  | { type: 'create'; scale: ScaleId; params: Params }
  | { type: 'run'; running: boolean }
  | { type: 'setParams'; params: Params }
  /** A one-shot action from a `button` control (e.g. "stimulate"); not replayed on reset. */
  | { type: 'action'; name: string }
  /** Rebuild the solver from the creation params plus every `setParams` since. */
  | { type: 'reset' }
  /** Solver wall-clock time per tick, in ms (the speed control). */
  | { type: 'setBudget'; budgetMs: number }
  | { type: 'ack'; seq: number }
  | { type: 'dispose' };

export type ParamValue = number | string | boolean;
export type Params = Record<string, ParamValue>;

export interface FrameField {
  name: string;
  shape: number[];
  data: Float32Array;
}

/** Measured by the worker's run loop, independent of the solver. */
export interface RuntimeStats {
  running: boolean;
  /** Solver steps per wall-clock second (0 if the solver does not report steps). */
  stepsPerSec: number;
  /** Simulated ms per wall-clock second. */
  simMsPerSec: number;
  /** Smoothed wall-clock ms spent in `advance` per tick. */
  solverMs: number;
}

export interface FrameMessage {
  type: 'frame';
  seq: number;
  timeMs: number;
  fields: FrameField[];
  /** Solver-specific readouts (e.g. MFPT, max concentration). */
  stats?: Record<string, number>;
  runtime: RuntimeStats;
}

export type FromWorker =
  | { type: 'ready'; version: string }
  | FrameMessage
  | { type: 'error'; message: string };

/** Typed arrays to hand over as transferables when posting a message. */
export function transferablesOf(msg: FromWorker): Transferable[] {
  return msg.type === 'frame' ? msg.fields.map((f) => f.data.buffer as ArrayBuffer) : [];
}
