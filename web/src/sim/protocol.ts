import type { ScaleId } from '../scales/registry';

/**
 * Messages between the UI thread and the simulation worker.
 *
 * Convention: the worker owns the solver and advances it with a wall-clock
 * budget per tick; it posts `frame` messages whose typed arrays are
 * transferred (not copied). The UI renders only the newest frame on each
 * requestAnimationFrame, so a slow solver never blocks rendering.
 */
export type ToWorker =
  | { type: 'init' }
  | { type: 'create'; scale: ScaleId; params: Record<string, unknown> }
  | { type: 'run'; running: boolean }
  | { type: 'setParams'; params: Record<string, unknown> }
  | { type: 'dispose' };

export interface FrameField {
  name: string;
  shape: number[];
  data: Float32Array;
}

export type FromWorker =
  | { type: 'ready'; version: string }
  | { type: 'frame'; timeMs: number; fields: FrameField[]; stats?: Record<string, number> }
  | { type: 'error'; message: string };

/** Typed arrays to hand over as transferables when posting a message. */
export function transferablesOf(msg: FromWorker): Transferable[] {
  return msg.type === 'frame' ? msg.fields.map((f) => f.data.buffer as ArrayBuffer) : [];
}
