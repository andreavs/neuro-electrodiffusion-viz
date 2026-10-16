import type { FromWorker, ToWorker } from './protocol';

type Listener = (msg: FromWorker) => void;

/** UI-side handle to the simulation worker. One instance per mounted scale. */
export class SimClient {
  private readonly worker: Worker;
  private readonly listeners = new Set<Listener>();

  constructor() {
    this.worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (ev: MessageEvent<FromWorker>) => {
      for (const l of this.listeners) l(ev.data);
    };
  }

  send(msg: ToWorker): void {
    this.worker.postMessage(msg);
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
          document.documentElement.dataset.solver = 'ready';
          resolve(m.version);
        } else if (m.type === 'error') {
          off();
          document.documentElement.dataset.solver = 'error';
          reject(new Error(m.message));
        }
      });
      this.send({ type: 'init' });
    });
  }

  terminate(): void {
    delete document.documentElement.dataset.solver;
    this.listeners.clear();
    this.worker.terminate();
  }
}
