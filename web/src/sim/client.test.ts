import { describe, expect, it, vi } from 'vitest';
import { FrameMailbox, SimClient, type WorkerLike } from './client';
import type { FrameMessage, FromWorker, ToWorker } from './protocol';

class FakeWorker implements WorkerLike {
  sent: ToWorker[] = [];
  terminated = false;
  onmessage: ((ev: MessageEvent<FromWorker>) => void) | null = null;
  postMessage(msg: ToWorker): void {
    this.sent.push(msg);
  }
  terminate(): void {
    this.terminated = true;
  }
  emit(msg: FromWorker): void {
    this.onmessage?.({ data: msg } as MessageEvent<FromWorker>);
  }
}

function frame(seq: number): FrameMessage {
  return {
    type: 'frame',
    seq,
    timeMs: seq,
    fields: [{ name: 'u', shape: [1], data: new Float32Array([seq]) }],
    runtime: { running: true, stepsPerSec: 0, simMsPerSec: 0, solverMs: 0 },
  };
}

function setup() {
  const worker = new FakeWorker();
  const client = new SimClient(() => worker);
  return { worker, client };
}

describe('FrameMailbox', () => {
  it('keeps only the newest frame and empties on take', () => {
    const box = new FrameMailbox();
    expect(box.take()).toBeNull();
    box.put(frame(1));
    box.put(frame(2));
    expect(box.take()?.seq).toBe(2);
    expect(box.take()).toBeNull();
  });
});

describe('SimClient', () => {
  it('sends control messages', () => {
    const { worker, client } = setup();
    client.create('knp', { dt: 0.1 });
    client.run(true);
    client.setParams({ g: 2 });
    client.action('stimulate');
    client.setBudget(6);
    client.reset();
    expect(worker.sent).toEqual([
      { type: 'create', scale: 'knp', params: { dt: 0.1 } },
      { type: 'run', running: true },
      { type: 'setParams', params: { g: 2 } },
      { type: 'action', name: 'stimulate' },
      { type: 'setBudget', budgetMs: 6 },
      { type: 'reset' },
    ]);
  });

  it('acks a frame exactly once, when the renderer takes it', () => {
    const { worker, client } = setup();
    worker.emit(frame(1));
    expect(worker.sent).toEqual([]);
    const f = client.takeFrame();
    expect(f?.seq).toBe(1);
    expect(worker.sent).toEqual([{ type: 'ack', seq: 1 }]);
    expect(client.takeFrame()).toBeNull();
    expect(worker.sent).toHaveLength(1);
  });

  it('hands the renderer the transferred buffer itself, not a copy', () => {
    const { worker, client } = setup();
    const f = frame(3);
    worker.emit(f);
    expect(client.takeFrame()!.fields[0]!.data).toBe(f.fields[0]!.data);
  });

  it('resolves ready() with the version', async () => {
    const { worker, client } = setup();
    const p = client.ready();
    expect(worker.sent).toEqual([{ type: 'init' }]);
    worker.emit({ type: 'ready', version: '1.2.3' });
    await expect(p).resolves.toBe('1.2.3');
  });

  it('terminate runs disposers, drops listeners and stops sending', () => {
    const { worker, client } = setup();
    const disposer = vi.fn();
    const listener = vi.fn();
    client.addDisposer(disposer);
    client.on(listener);
    client.terminate();
    client.terminate();
    expect(disposer).toHaveBeenCalledTimes(1);
    expect(worker.terminated).toBe(true);
    expect(worker.onmessage).toBeNull();
    client.run(true);
    expect(worker.sent).toEqual([]);
    expect(listener).not.toHaveBeenCalled();
  });

  it('removed disposers do not run', () => {
    const { client } = setup();
    const disposer = vi.fn();
    const off = client.addDisposer(disposer);
    off();
    client.terminate();
    expect(disposer).not.toHaveBeenCalled();
  });
});
