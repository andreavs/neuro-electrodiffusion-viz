import { describe, expect, it, vi } from 'vitest';
import { SimClient } from '../sim/client';
import { FakeWorker } from '../sim/testing/fakeWorker';
import type { StatsView } from './controls';
import { FpsMeter, startRenderLoop } from './renderLoop';


/** Manual requestAnimationFrame. */
function fakeRaf() {
  let next = 1;
  const cbs = new Map<number, FrameRequestCallback>();
  return {
    raf: (cb: FrameRequestCallback) => {
      cbs.set(next, cb);
      return next++;
    },
    cancelRaf: (id: number) => void cbs.delete(id),
    step(now: number) {
      const pending = [...cbs.entries()];
      cbs.clear();
      for (const [, cb] of pending) cb(now);
    },
    get pending() {
      return cbs.size;
    },
  };
}

describe('FpsMeter', () => {
  it('measures the rate of ticks over its window', () => {
    const m = new FpsMeter(1000);
    expect(m.tick(0)).toBe(0);
    for (let i = 1; i <= 120; i++) m.tick((i * 1000) / 60);
    expect(m.fpsAt(2000)).toBeCloseTo(60, 5);
    expect(m.fpsAt(5000)).toBe(0);
  });
});

describe('startRenderLoop', () => {
  it('draws each new frame once, acks it, and reports stats', () => {
    const worker = new FakeWorker();
    const client = new SimClient(() => worker);
    const r = fakeRaf();
    const draw = vi.fn();
    const stats: StatsView[] = [];
    startRenderLoop(client, { draw, stats: { updateStats: (s) => stats.push(s) }, raf: r.raf, cancelRaf: r.cancelRaf });

    r.step(0);
    expect(draw).not.toHaveBeenCalled();
    worker.emit({
      type: 'frame',
      seq: 1,
      timeMs: 4,
      fields: [],
      runtime: { running: true, stepsPerSec: 100, simMsPerSec: 2, solverMs: 3 },
    });
    r.step(16);
    r.step(33);
    expect(draw).toHaveBeenCalledTimes(1);
    expect(worker.sent).toEqual([{ type: 'ack', seq: 1 }]);
    expect(stats[0]).toMatchObject({ simTimeMs: 4, stepsPerSec: 100, simMsPerSec: 2 });
  });

  it('stops when the client terminates, leaving no animation frame pending', () => {
    const client = new SimClient(() => new FakeWorker());
    const r = fakeRaf();
    startRenderLoop(client, { draw: () => {}, raf: r.raf, cancelRaf: r.cancelRaf });
    r.step(0);
    expect(r.pending).toBe(1);
    client.terminate();
    expect(r.pending).toBe(0);
  });

  it('stop() is idempotent and unregisters from the client', () => {
    const client = new SimClient(() => new FakeWorker());
    const r = fakeRaf();
    const stop = startRenderLoop(client, { draw: () => {}, raf: r.raf, cancelRaf: r.cancelRaf });
    stop();
    stop();
    expect(r.pending).toBe(0);
    client.terminate();
  });
});
