/// <reference lib="webworker" />
import init, { core_version } from '../wasm/pkg/ed_wasm.js';
import { transferablesOf, type FromWorker, type ToWorker } from './protocol';

declare const self: DedicatedWorkerGlobalScope;

function post(msg: FromWorker): void {
  self.postMessage(msg, transferablesOf(msg));
}

self.onmessage = async (ev: MessageEvent<ToWorker>) => {
  const msg = ev.data;
  try {
    switch (msg.type) {
      case 'init':
        await init();
        post({ type: 'ready', version: core_version() });
        break;
      case 'create':
      case 'run':
      case 'setParams':
      case 'dispose':
        // Scale solvers are wired in by their own issues.
        break;
    }
  } catch (err) {
    post({ type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
};
