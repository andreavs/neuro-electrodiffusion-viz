import { beforeEach, describe, expect, it } from 'vitest';
import type { ScaleId } from '../scales/registry';
import { DEFAULT_BUDGET_MS, MAX_BUDGET_MS, RunLoop, clampBudget, type LoopHost } from './loop';
import type { FrameMessage, FromWorker } from './protocol';
import type { SolverFactory } from './solver';
import { FakeSolver } from './testing/fakeSolver';

/** Manual macrotask queue and clock standing in for the worker environment. */
class FakeHost implements LoopHost {
  clock = { t: 1000 };
  posted: FromWorker[] = [];
  tasks: { fn: () => void; cancelled: boolean }[] = [];

  post(msg: FromWorker): void {
    this.posted.push(msg);
  }
  schedule(fn: () => void): () => void {
    const task = { fn, cancelled: false };
    this.tasks.push(task);
    return () => {
      task.cancelled = true;
    };
  }
  now(): number {
    return this.clock.t;
  }
  get pendingTasks(): number {
    return this.tasks.filter((t) => !t.cancelled).length;
  }
  /** Run queued tasks (including ones they schedule) up to `limit`. */
  runTasks(limit = 100): number {
    let ran = 0;
    while (this.tasks.length && ran < limit) {
      const task = this.tasks.shift()!;
      if (task.cancelled) continue;
      task.fn();
      ran++;
    }
    return ran;
  }
  frames(): FrameMessage[] {
    return this.posted.filter((m): m is FrameMessage => m.type === 'frame');
  }
  lastFrame(): FrameMessage {
    const f = this.frames();
    return f[f.length - 1]!;
  }
}

describe('RunLoop', () => {
  let host: FakeHost;
  let solvers: FakeSolver[];
  let loop: RunLoop;

  beforeEach(() => {
    host = new FakeHost();
    solvers = [];
    const make: SolverFactory = (params) => {
      const s = new FakeSolver(params, host.clock, 5);
      solvers.push(s);
      return s;
    };
    const factories = { knp: make, emi: make, particles: make } satisfies Record<ScaleId, SolverFactory>;
    loop = new RunLoop(host, factories);
  });

  it('constructs the solver on create and posts an initial frame', () => {
    loop.handle({ type: 'create', scale: 'knp', params: { dt: 0.5 } });
    expect(solvers).toHaveLength(1);
    expect(solvers[0]!.params).toEqual({ dt: 0.5 });
    expect(host.frames()).toHaveLength(1);
    expect(host.lastFrame()).toMatchObject({ seq: 1, timeMs: 0, runtime: { running: false } });
    expect(host.pendingTasks).toBe(0);
  });

  it('ticks with the default budget only while running', () => {
    loop.handle({ type: 'create', scale: 'knp', params: {} });
    host.runTasks();
    expect(solvers[0]!.advances).toBe(0);
    loop.handle({ type: 'run', running: true });
    host.runTasks(1);
    expect(solvers[0]!.budgets).toEqual([DEFAULT_BUDGET_MS]);
  });

  it('never has more than one frame in flight (back-pressure)', () => {
    loop.handle({ type: 'create', scale: 'knp', params: {} });
    loop.handle({ type: 'run', running: true });
    host.runTasks();
    // Only the initial frame was posted; the solver ran maxTicksAhead (2) ticks and then idled.
    expect(host.frames()).toHaveLength(1);
    expect(solvers[0]!.advances).toBe(2);
    expect(host.pendingTasks).toBe(0);

    loop.handle({ type: 'ack', seq: 1 });
    // The ack immediately releases the work done meanwhile...
    expect(host.frames()).toHaveLength(2);
    expect(host.lastFrame()).toMatchObject({ seq: 2, timeMs: 2 }); // 2 advances × 10 steps × 0.1 ms
    // ...and resumes ticking.
    host.runTasks();
    expect(solvers[0]!.advances).toBe(4);
    expect(host.frames()).toHaveLength(2);
  });

  it('ignores stale and duplicate acks', () => {
    loop.handle({ type: 'create', scale: 'knp', params: {} });
    loop.handle({ type: 'run', running: true });
    host.runTasks();
    loop.handle({ type: 'ack', seq: 7 });
    expect(host.frames()).toHaveLength(1);
    loop.handle({ type: 'ack', seq: 1 });
    loop.handle({ type: 'ack', seq: 1 });
    expect(host.frames()).toHaveLength(2);
  });

  it('keeps the frame rate tied to acks', () => {
    loop.handle({ type: 'create', scale: 'knp', params: {} });
    loop.handle({ type: 'run', running: true });
    for (let i = 0; i < 10; i++) {
      host.runTasks();
      loop.handle({ type: 'ack', seq: host.lastFrame().seq });
    }
    const seqs = host.frames().map((f) => f.seq);
    expect(seqs).toEqual(Array.from({ length: 11 }, (_, i) => i + 1));
  });

  it('pauses: cancels the pending tick and reports not running', () => {
    loop.handle({ type: 'create', scale: 'knp', params: {} });
    loop.handle({ type: 'ack', seq: 1 });
    loop.handle({ type: 'run', running: true });
    expect(loop.hasPendingTick).toBe(true);
    loop.handle({ type: 'ack', seq: host.lastFrame().seq });
    loop.handle({ type: 'run', running: false });
    expect(loop.hasPendingTick).toBe(false);
    expect(host.runTasks()).toBe(0);
    expect(solvers[0]!.advances).toBe(0);
    expect(host.lastFrame().runtime.running).toBe(false);
  });

  it('forwards params and actions, and posts a frame for the change', () => {
    loop.handle({ type: 'create', scale: 'emi', params: { dt: 0.1, g: 1 } });
    loop.handle({ type: 'ack', seq: 1 });
    loop.handle({ type: 'setParams', params: { g: 4 } });
    expect(solvers[0]!.params).toEqual({ dt: 0.1, g: 4 });
    expect(host.frames()).toHaveLength(2);
    loop.handle({ type: 'ack', seq: 2 });
    loop.handle({ type: 'action', name: 'stimulate' });
    expect(solvers[0]!.actions).toEqual(['stimulate']);
    expect(host.frames()).toHaveLength(3);
  });

  it('reset rebuilds from creation params plus later setParams, but not actions', () => {
    loop.handle({ type: 'create', scale: 'particles', params: { dt: 0.1, n: 100 } });
    loop.handle({ type: 'setParams', params: { n: 500 } });
    loop.handle({ type: 'action', name: 'release' });
    loop.handle({ type: 'run', running: true });
    host.runTasks();
    loop.handle({ type: 'reset' });
    expect(solvers).toHaveLength(2);
    expect(solvers[0]!.disposed).toBe(true);
    expect(solvers[1]!.params).toEqual({ dt: 0.1, n: 500 });
    expect(solvers[1]!.actions).toEqual([]);
    // Still running after reset.
    loop.handle({ type: 'ack', seq: host.lastFrame().seq });
    host.runTasks();
    expect(solvers[1]!.advances).toBeGreaterThan(0);
  });

  it('applies the speed budget, clamped', () => {
    loop.handle({ type: 'create', scale: 'knp', params: {} });
    loop.handle({ type: 'setBudget', budgetMs: 3 });
    loop.handle({ type: 'run', running: true });
    host.runTasks(1);
    loop.handle({ type: 'setBudget', budgetMs: 1e9 });
    host.runTasks(1);
    expect(solvers[0]!.budgets).toEqual([3, MAX_BUDGET_MS]);
    expect(clampBudget(Number.NaN)).toBe(DEFAULT_BUDGET_MS);
    expect(clampBudget(0)).toBe(0.5);
  });

  it('measures steps/s and sim-ms/s from the injected clock', () => {
    loop.handle({ type: 'create', scale: 'knp', params: {} });
    loop.handle({ type: 'run', running: true });
    // Each advance costs 5 ms of fake wall time; 10 steps × 0.1 ms.
    for (let i = 0; i < 200; i++) {
      host.runTasks();
      loop.handle({ type: 'ack', seq: host.lastFrame().seq });
    }
    const rt = host.lastFrame().runtime;
    expect(rt.running).toBe(true);
    expect(rt.solverMs).toBeCloseTo(5);
    expect(rt.stepsPerSec).toBeCloseTo(2000, -1); // 10 steps per 5 ms
    expect(rt.simMsPerSec).toBeCloseTo(200, 0); // 1 sim-ms per 5 ms
  });

  it('reports solver errors and stops ticking', () => {
    loop.handle({ type: 'create', scale: 'knp', params: {} });
    solvers[0]!.failOnAdvance = true;
    loop.handle({ type: 'run', running: true });
    host.runTasks();
    expect(host.posted.at(-1)).toEqual({ type: 'error', message: 'boom' });
    expect(loop.isRunning).toBe(false);
    expect(loop.hasPendingTick).toBe(false);
  });

  it('reports factory errors', () => {
    const failing = new RunLoop(host, {
      knp: () => {
        throw new Error('not implemented');
      },
    } as unknown as Record<ScaleId, SolverFactory>);
    failing.handle({ type: 'create', scale: 'knp', params: {} });
    expect(host.posted).toEqual([{ type: 'error', message: 'not implemented' }]);
  });

  it('dispose frees the solver and leaves no pending ticks', () => {
    loop.handle({ type: 'create', scale: 'knp', params: {} });
    loop.handle({ type: 'run', running: true });
    loop.handle({ type: 'dispose' });
    expect(solvers[0]!.disposed).toBe(true);
    expect(loop.hasPendingTick).toBe(false);
    expect(host.runTasks()).toBe(0);
    loop.handle({ type: 'ack', seq: 1 });
    expect(host.frames()).toHaveLength(1);
  });

  it('create replaces an existing solver', () => {
    loop.handle({ type: 'create', scale: 'knp', params: {} });
    loop.handle({ type: 'create', scale: 'emi', params: { a: 1 } });
    expect(solvers[0]!.disposed).toBe(true);
    expect(solvers[1]!.params).toEqual({ a: 1 });
  });

  it('create or reset while a frame is unacknowledged still posts a frame (no deadlock)', () => {
    loop.handle({ type: 'create', scale: 'knp', params: {} });
    loop.handle({ type: 'run', running: true });
    host.runTasks();
    // The client dropped frame 1 without acking it, then re-created.
    loop.handle({ type: 'create', scale: 'knp', params: { dt: 0.2 } });
    expect(host.lastFrame()).toMatchObject({ seq: 2, timeMs: 0 });
    loop.handle({ type: 'ack', seq: 1 }); // late ack for the old frame is ignored
    loop.handle({ type: 'reset' });
    expect(host.lastFrame().seq).toBe(3);
    loop.handle({ type: 'ack', seq: 3 });
    host.runTasks();
    expect(solvers[2]!.advances).toBeGreaterThan(0);
  });

  it('reports errors thrown by frame() during a tick and stops', () => {
    loop.handle({ type: 'create', scale: 'knp', params: {} });
    loop.handle({ type: 'ack', seq: 1 });
    solvers[0]!.frame = () => {
      throw new Error('panic in frame');
    };
    loop.handle({ type: 'run', running: true });
    host.runTasks();
    expect(host.posted.at(-1)).toEqual({ type: 'error', message: 'panic in frame' });
    expect(loop.isRunning).toBe(false);
    expect(loop.hasPendingTick).toBe(false);
  });

  it('does not wait for an ack when posting a frame throws', () => {
    let throwOnce = true;
    const post = host.post.bind(host);
    host.post = (msg) => {
      if (msg.type === 'frame' && throwOnce) {
        throwOnce = false;
        throw new Error('DataCloneError');
      }
      post(msg);
    };
    loop.handle({ type: 'create', scale: 'knp', params: {} });
    expect(host.posted).toEqual([{ type: 'error', message: 'DataCloneError' }]);
    loop.handle({ type: 'setParams', params: { dt: 0.3 } });
    expect(host.lastFrame()).toMatchObject({ seq: 1 });
  });

  it('idles without posting once the solver reports nothing changed, and resumes on setParams', () => {
    loop.handle({ type: 'create', scale: 'particles', params: {} });
    loop.handle({ type: 'run', running: true });
    loop.handle({ type: 'ack', seq: host.lastFrame().seq });
    solvers[0]!.finished = true;
    host.runTasks();
    const posted = host.frames().length;
    loop.handle({ type: 'ack', seq: host.lastFrame().seq });
    host.runTasks();
    expect(host.frames()).toHaveLength(posted);
    expect(host.pendingTasks).toBe(0);
    solvers[0]!.finished = false;
    loop.handle({ type: 'setParams', params: { dt: 0.2 } });
    expect(host.frames()).toHaveLength(posted + 1);
    expect(host.pendingTasks).toBe(1);
  });
});
