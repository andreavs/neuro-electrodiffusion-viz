import { describe, expect, it } from 'vitest';
import { ShardRunner, type FromShard, type ToShard } from './particleShard';
import type { ParticlesWasm } from './particles';
import { DISPLAY_MAX, ShardedParticlesSolver, type ShardPort } from './shardedParticles';

/** Each step removes one ion; exit times are recorded so the MFPT is exact. */
class FakeSim implements ParticlesWasm {
  alive = 0;
  released = 0;
  ref = false;
  time = 0;
  exits: number[] = [];
  constructor(readonly geometry: string) {}
  set_field = (kind: string, dv: number) => void (this.ref = kind !== 'off' && dv !== 0);
  set_dt() {}
  release(n: number) {
    [this.alive, this.released, this.time, this.exits] = [n, n, 0, []];
  }
  step(n: number) {
    const k = Math.min(n, this.alive);
    for (let i = 0; i < k; i++) this.exits.push(++this.time);
    this.alive -= k;
    return k;
  }
  time_ms = () => this.time;
  n_alive = () => this.alive;
  n_released = () => this.released;
  has_reference = () => this.ref;
  positions = () => new Float32Array(3 * this.alive).fill(this.released);
  regions = () => new Float32Array(this.alive).fill(1);
  survival = () => Float32Array.of(0, 1);
  histogram = (_ref: boolean, bins: number) => new Float32Array(bins).fill(1 / bins);
  mfpt = () => {
    const n = this.exits.length;
    if (n === 0) return Float64Array.of(NaN, NaN, 0, NaN);
    const m = this.exits.reduce((a, b) => a + b, 0) / n;
    const v = n > 1 ? this.exits.reduce((a, t) => a + (t - m) ** 2, 0) / (n - 1) : NaN;
    return Float64Array.of(m, Math.sqrt(v / n), n, m);
  };
  profile = () => Float32Array.of(0.1, 0, 0, 1);
  neck_top = () => 1;
  neck_radius = () => 0.1;
  free() {}
}

/** In-memory shard ports; replies are queued until `flush`. */
function harness(params: Record<string, string | number>, cores = 5) {
  const queue: (() => void)[] = [];
  const sims: FakeSim[] = [];
  let wakes = 0;
  const spawn = (index: number): ShardPort => {
    const runner = new ShardRunner(
      (g) => {
        const s = new FakeSim(g);
        sims[index] = s;
        return s;
      },
      index + 1,
      () => 0,
    );
    let cb: (m: FromShard) => void = () => {};
    return {
      post: (msg: ToShard) =>
        queue.push(() => {
          const s = runner.handle(msg);
          if (s) cb(s);
        }),
      onMessage: (f) => {
        cb = f;
      },
      terminate: () => {},
    };
  };
  const solver = new ShardedParticlesSolver(params, spawn, cores);
  solver.setWake(() => wakes++);
  const flush = () => {
    while (queue.length) queue.shift()!();
  };
  return { solver, sims, flush, wakes: () => wakes };
}

describe('ShardedParticlesSolver', () => {
  it('waits for every shard, then merges the round into one frame', () => {
    const { solver, sims, flush, wakes } = harness({ particles: 25_000, field: 'off' });
    flush(); // the release, while paused: no steps follow
    expect(sims).toHaveLength(3);
    expect(sims.map((s) => s.released)).toEqual([8334, 8333, 8333]);
    expect(wakes()).toBe(1);
    expect(solver.frame().stats).toMatchObject({ released: 25_000, alive: 25_000 });
    expect(sims[0]!.alive).toBe(8334);
    expect(solver.advance(12)).toBe('wait'); // asks for steps
    flush(); // the shards answer and go straight on with one more round
    expect(wakes()).toBe(3);
    expect(solver.advance(12)).toBe(true);
    const f = solver.frame();
    const by = Object.fromEntries(f.fields.map((x) => [x.name, x]));
    expect(f.stats).toMatchObject({ released: 25_000, alive: 25_000 - 3 * f.steps!, workers: 3 });
    expect(by.positions!.shape[0]).toBe(f.stats!.alive);
    expect(by.profile!.data.length).toBe(4);
    expect(by.survival!.data.length).toBe(6); // t = 0 and two rounds
    expect(f.stats!.escaped).toBe(3 * f.steps!);
  });

  it('draws at most DISPLAY_MAX ions in total', () => {
    const { solver, flush } = harness({ particles: 1_000_000, field: 'off' });
    [flush(), solver.advance(12), flush(), solver.advance(12)];
    const n = solver.frame().fields.find((x) => x.name === 'positions')!.shape[0]!;
    expect(Math.abs(n - DISPLAY_MAX)).toBeLessThan(3);
  });

  it('drops replies from an earlier release and reports finished runs', () => {
    const { solver, flush } = harness({ particles: 1000, field: 'off' });
    flush();
    solver.advance(12);
    solver.setParams({ particles: '5000' }); // the step reply in flight is now stale
    flush();
    expect(solver.advance(12)).toBe(true);
    expect(solver.frame().stats).toMatchObject({ released: 5000, alive: 5000 });
    let guard = 0;
    for (let r: boolean | 'wait' = true; r !== false && guard < 10_000; guard++) {
      r = solver.advance(12);
      flush();
    }
    expect(solver.frame().stats).toMatchObject({ alive: 0, escaped: 5000 });
    expect(solver.advance(12)).toBe(false);
  });

  it('stops stepping when the loop stops asking (paused)', () => {
    const { solver, sims, flush } = harness({ particles: 1000, field: 'off' });
    flush();
    solver.advance(12);
    flush(); // one round, plus the one sent ahead
    solver.frame(); // a paused wake: merges, asks for nothing
    flush();
    const steps = sims[0]!.released - sims[0]!.alive;
    flush();
    solver.frame();
    flush();
    expect(sims[0]!.released - sims[0]!.alive).toBe(steps);
    expect(solver.frame().fields.find((x) => x.name === 'profile')).toBeDefined();
  });

  it('clears a shard error on the next release and drops leftover shards', () => {
    const { solver, sims, flush } = harness({ particles: 25_000, field: 'off' });
    flush();
    (solver as unknown as { error: Error }).error = new Error('too many particles');
    expect(() => solver.advance(12)).toThrow();
    solver.setParams({ particles: '1000' });
    flush();
    expect(solver.advance(12)).toBe(true);
    expect(sims.map((s) => s.released)).toEqual([1000, 8333, 8333]); // 1 and 2 were terminated
    expect(solver.frame().stats).toMatchObject({ workers: 1, released: 1000 });
  });

  it('surfaces shard errors', () => {
    const queue: (() => void)[] = [];
    const solver = new ShardedParticlesSolver(
      {},
      () => {
        let cb: (m: FromShard) => void = () => {};
        return {
          post: () => queue.push(() => cb({ type: 'error', message: 'no wasm' })),
          onMessage: (f) => {
            cb = f;
          },
          terminate: () => {},
        };
      },
      4,
    );
    queue.forEach((f) => f());
    expect(() => solver.advance(12)).toThrow(/no wasm/);
  });
});
