/// <reference lib="webworker" />
import init, { core_version } from '../wasm/pkg/ed_wasm.js';
import { RunLoop, type LoopHost } from './loop';
import { transferablesOf, type FromWorker, type ToWorker } from './protocol';
import { SOLVERS } from './solvers';

declare const self: DedicatedWorkerGlobalScope;

function post(msg: FromWorker): void {
  self.postMessage(msg, transferablesOf(msg));
}

// MessageChannel gives a macrotask without setTimeout's 4 ms nesting clamp, so
// back-to-back ticks spend their time in the solver rather than waiting.
const channel = new MessageChannel();
const pending = new Map<number, () => void>();
let nextId = 0;
channel.port1.onmessage = (ev: MessageEvent<number>) => {
  const fn = pending.get(ev.data);
  pending.delete(ev.data);
  fn?.();
};

const host: LoopHost = {
  post,
  schedule(fn) {
    const id = nextId++;
    pending.set(id, fn);
    channel.port2.postMessage(id);
    return () => pending.delete(id);
  },
  now: () => performance.now(),
};

const loop = new RunLoop(host, SOLVERS);
let wasmReady: Promise<void> | null = null;

// Handle messages strictly in order: `create` must wait for the WASM module,
// and nothing after it may overtake it.
let queue = Promise.resolve();
self.onmessage = (ev: MessageEvent<ToWorker>) => {
  const msg = ev.data;
  queue = queue.then(async () => {
    try {
      if (msg.type === 'init') {
        wasmReady ??= init().then(() => undefined);
        await wasmReady;
        post({ type: 'ready', version: core_version() });
        return;
      }
      if (msg.type === 'create') {
        wasmReady ??= init().then(() => undefined);
        await wasmReady;
      }
      loop.handle(msg);
    } catch (err) {
      post({ type: 'error', message: err instanceof Error ? err.message : String(err) });
    }
  });
};
