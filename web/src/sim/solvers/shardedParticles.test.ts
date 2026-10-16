import { describe, expect, it } from 'vitest';
import { ShardRunner, type FromShard, type ToShard } from './particleShard';
import type { ParticlesWasm } from './particles';
import { DISPLAY_MAX, ShardedParticlesSolver, type ShardPort } from './shardedParticles';

/**
 * Each step removes one ion; exit times are recorded so the MFPT is exact. With a dendrite,
 * each escaped ion lingers there until the spine is empty, then one leaves per step.
 */
class FakeSim implements ParticlesWasm {
  alive = 0;
  released = 0;
  ref = false;
  time = 0;
  passage = 0;
  moving = 0;
  exits: number[] = [];
  constructor(
    readonly geometry: string,
    readonly dendrite = false,
  ) {}
  set_field = (kind: string, dv: number) => void (this.ref = kind !== 'off' && dv !== 0);
  set_dt() {}
  release(n: number) {
    [this.alive, this.released, this.time, this.passage, this.moving, this.exits] = [n, n, 0, 0, 0, []];
  }
  step(n: number) {
    let k = 0;
    for (; k < n && this.alive + this.moving > 0; k++) {
      this.time++;
      if (this.alive > 0) {
        this.alive--;
        this.exits.push((this.passage = this.time));
        if (this.dendrite) this.moving++;
      } else this.moving--;
    }
    return k;
  }
  time_ms = () => this.time;
  passage_time_ms = () => this.passage;
  n_alive = () => this.alive;
  n_in_dendrite = () => this.moving;
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
  dendrite_radius = () => (this.dendrite ? 0.45 : 0);
  dendrite_half_length = () => (this.dendrite ? 3 : 0);
  free() {}
}

/** In-memory shard ports; replies are queued until `flush`. */
function harness(params: Record<string, string | number>, cores = 5) {
  const queue: (() => void)[] = [];
  const sims: FakeSim[] = [];
  let wakes = 0;
  const spawn = (index: number): ShardPort => {
    const runner = new ShardRunner(
      (g, d) => {
        const s = new FakeSim(g, d);
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

  it('keeps running while escaped ions spread in the dendrite, sampling survival only until the last exit', () => {
    const { solver, flush } = harness({ particles: 30_000, field: 'off', outlet: 'dendrite' });
    flush();
    let guard = 0;
    let midway: Record<string, number> | undefined;
    for (let r: boolean | 'wait' = true; r !== false && guard < 10_000; guard++) {
      r = solver.advance(12);
      flush();
      const st = solver.frame().stats!;
      if (st.alive === 0 && st.inDendrite! > 0) midway ??= st;
    }
    // Each shard empties its spine after 10 000 steps, then its dendrite after 10 000 more.
    expect(midway).toMatchObject({ escaped: 30_000, passageTime: 10_000, dendriteRadius: 0.45, dendriteHalfLength: 3 });
    const f = solver.frame();
    expect(f.timeMs).toBe(20_000);
    expect(f.stats).toMatchObject({ alive: 0, inDendrite: 0, passageTime: 10_000, histTMax: 10_000 });
    const surv = f.fields.find((x) => x.name === 'survival')!.data;
    expect(Array.from(surv.slice(-2))).toEqual([10_000, 0]);
    expect(solver.advance(12)).toBe(false);
    // Switching back to an absorbing base forgets the dendrite before the shards reply.
    solver.setParams({ outlet: 'absorb' });
    expect(solver.frame().stats).toMatchObject({ dendriteRadius: 0, dendriteHalfLength: 0 });
    flush();
    expect(solver.frame().stats).toMatchObject({ dendriteRadius: 0, released: 30_000 });
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
