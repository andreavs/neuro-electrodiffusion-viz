import type { ScaleId } from '../scales/registry';
import type { FrameMessage, FromWorker, Params, ToWorker } from './protocol';

type Listener = (msg: FromWorker) => void;

/** The parts of `Worker` the client uses, so tests can pass a fake. */
export interface WorkerLike {
  postMessage(msg: ToWorker): void;
  onmessage: ((ev: MessageEvent<FromWorker>) => void) | null;
  onerror: ((ev: ErrorEvent) => void) | null;
  onmessageerror: ((ev: MessageEvent) => void) | null;
  terminate(): void;
}

/**
 * Holds the newest unconsumed frame. Because the worker waits for an ack
 * before sending another frame, at most one is ever outstanding; a frame
 * arriving before the previous one is taken (which only happens if a caller
 * acks out of band) replaces it.
 */
export class FrameMailbox {
  private latest: FrameMessage | null = null;

  put(frame: FrameMessage): void {
    this.latest = frame;
  }

  /** Remove and return the newest frame, or null if none arrived since the last take. */
  take(): FrameMessage | null {
    const f = this.latest;
    this.latest = null;
    return f;
  }

  clear(): void {
    this.latest = null;
  }
}

/** Sets `<html data-solver>` for the browser smoke test; a no-op outside the DOM (unit tests). */
function markSolver(state: 'ready' | 'error' | undefined): void {
  if (typeof document === 'undefined') return;
  if (state) document.documentElement.dataset.solver = state;
  else delete document.documentElement.dataset.solver;
}

function defaultWorker(): WorkerLike {
  return new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }) as WorkerLike;
}

/** UI-side handle to the simulation worker. One instance per mounted scale. */
export class SimClient {
  private readonly worker: WorkerLike;
  private readonly listeners = new Set<Listener>();
  private readonly disposers = new Set<() => void>();
  private readonly mailbox = new FrameMailbox();
  private terminated = false;

  constructor(makeWorker: () => WorkerLike = defaultWorker) {
    this.worker = makeWorker();
    this.worker.onmessage = (ev: MessageEvent<FromWorker>) => this.dispatch(ev.data);
    // Load failures and uncaught worker exceptions surface as `error` messages,
    // so `ready()` rejects instead of hanging.
    this.worker.onerror = (ev: ErrorEvent) => {
      ev.preventDefault?.();
      this.dispatch({ type: 'error', message: ev.message || 'Simulation worker failed' });
    };
    this.worker.onmessageerror = () =>
      this.dispatch({ type: 'error', message: 'Could not decode a message from the simulation worker' });
  }

  private dispatch(msg: FromWorker): void {
    if (msg.type === 'frame') this.mailbox.put(msg);
    for (const l of this.listeners) l(msg);
  }

  send(msg: ToWorker): void {
    if (!this.terminated) this.worker.postMessage(msg);
  }

  on(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /**
   * Load the WASM module; resolves with its version. Mirrors the outcome on
   * `<html data-solver="ready|error">`, which the browser smoke test (web/e2e) waits for.
   */
  ready(): Promise<string> {
    return new Promise((resolve, reject) => {
      const off = this.on((m) => {
        if (m.type === 'ready') {
          off();
          markSolver('ready');
          resolve(m.version);
        } else if (m.type === 'error') {
          off();
          markSolver('error');
          reject(new Error(m.message));
        }
      });
      this.send({ type: 'init' });
    });
  }

  create(scale: ScaleId, params: Params = {}): void {
    this.mailbox.clear();
    this.send({ type: 'create', scale, params });
  }

  run(running: boolean): void {
    this.send({ type: 'run', running });
  }

  setParams(params: Params): void {
    this.send({ type: 'setParams', params });
  }

  action(name: string): void {
    this.send({ type: 'action', name });
  }

  reset(): void {
    this.send({ type: 'reset' });
  }

  setBudget(budgetMs: number): void {
    this.send({ type: 'setBudget', budgetMs });
  }

  /**
   * Take the newest frame for rendering and acknowledge it, which lets the
   * worker send the next one. Call once per animation frame; returns null if
   * nothing new arrived.
   */
  takeFrame(): FrameMessage | null {
    const f = this.mailbox.take();
    if (f) this.send({ type: 'ack', seq: f.seq });
    return f;
  }

  /** Register cleanup (timers, listeners, render loops) to run on `terminate`. */
  addDisposer(fn: () => void): () => void {
    this.disposers.add(fn);
    return () => this.disposers.delete(fn);
  }

  terminate(): void {
    if (this.terminated) return;
    this.terminated = true;
    markSolver(undefined);
    for (const d of this.disposers) d();
    this.disposers.clear();
    this.listeners.clear();
    this.mailbox.clear();
    this.worker.onmessage = null;
    this.worker.onerror = null;
    this.worker.onmessageerror = null;
    this.worker.terminate();
  }
}
