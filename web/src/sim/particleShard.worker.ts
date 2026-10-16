/// <reference lib="webworker" />
import init, { ParticlesSim } from '../wasm/pkg/ed_wasm.js';
import { ShardRunner, shardTransferables, type FromShard, type ToShard } from './solvers/particleShard';

declare const self: DedicatedWorkerGlobalScope;

// The seed arrives in the worker's name so each shard draws its own random stream.
const seed = Number(self.name) || 1;
const runner = new ShardRunner((geometry, dt, s) => new ParticlesSim(geometry, dt, s), seed);
const ready = init().then(() => undefined);

// Handle messages strictly in order, after the WASM module has loaded.
let queue = Promise.resolve();
self.onmessage = (ev: MessageEvent<ToShard>) => {
  const msg = ev.data;
  queue = queue.then(async () => {
    try {
      await ready;
      const state = runner.handle(msg);
      if (state) self.postMessage(state satisfies FromShard, shardTransferables(state));
    } catch (err) {
      self.postMessage({ type: 'error', message: err instanceof Error ? err.message : String(err) } satisfies FromShard);
    }
  });
};
