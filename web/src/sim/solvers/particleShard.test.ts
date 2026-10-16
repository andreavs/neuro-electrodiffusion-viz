import { describe, expect, it } from 'vitest';
import { ShardRunner, shardTransferables } from './particleShard';
import type { ParticlesWasm } from './particles';

/** Each step removes one ion. */
function fakeSim(geometry: string, log: string[]): ParticlesWasm {
  let alive = 0;
  let released = 0;
  let time = 0;
  let ref = false;
  log.push(`new ${geometry}`);
  return {
    set_field: (kind, dv) => void (ref = kind !== 'off' && dv !== 0),
    set_dt: () => {},
    release: (n) => void (alive = released = n),
    step: (n) => {
      const k = Math.min(n, alive);
      alive -= k;
      time += k;
      return k;
    },
    time_ms: () => time,
    passage_time_ms: () => time,
    n_alive: () => alive,
    n_in_dendrite: () => 0,
    n_released: () => released,
    has_reference: () => ref,
    positions: () => new Float32Array(3 * alive),
    regions: () => new Float32Array(alive),
    survival: () => Float32Array.of(0, 1),
    histogram: (_r, bins) => new Float32Array(bins),
    mfpt: () => Float64Array.of(1, 0, 0, 1),
    profile: () => Float32Array.of(0.1, 0, 0, 1),
    neck_top: () => 1,
    neck_radius: () => 0.1,
    dendrite_radius: () => 0,
    dendrite_half_length: () => 0,
    free: () => void log.push('free'),
  };
}

describe('ShardRunner', () => {
  const start = { type: 'start', gen: 1, geometry: 'sharp', dt: 1e-3, field: 'neck', dv: 20, n: 500, nRef: 0 } as const;
  const step = { type: 'step', gen: 1, steps: 100, tMax: 2, bins: 8, display: 50 } as const;

  it('replies to a release with the wall profile, then steps and trims positions for display', () => {
    const log: string[] = [];
    const r = new ShardRunner((g) => fakeSim(g, log), 1, () => 0);
    const s0 = r.handle(start)!;
    expect(s0).toMatchObject({ gen: 1, steps: 0, alive: 500, released: 500, neckTop: 1 });
    expect(s0.profile).toHaveLength(4);
    expect(s0.positions).toHaveLength(0);
    const s1 = r.handle(step)!;
    expect(s1).toMatchObject({ steps: 100, alive: 400, tMax: 2 });
    expect(s1.profile).toBeUndefined();
    expect(s1.positions).toHaveLength(150);
    expect(s1.regions).toHaveLength(50);
    expect(s1.histogram).toHaveLength(8);
    expect(s1.ref).toBeDefined();
    expect(shardTransferables(s1)).toHaveLength(7);
  });

  it('ignores steps for another release and rebuilds only for a new geometry', () => {
    const log: string[] = [];
    const r = new ShardRunner((g) => fakeSim(g, log), 1, () => 0);
    expect(r.handle(step)).toBeNull();
    r.handle(start);
    expect(r.handle({ ...step, gen: 0 })).toBeNull();
    r.handle({ ...start, gen: 2 });
    r.handle({ ...start, gen: 3, geometry: 'funnel' });
    expect(log).toEqual(['new sharp', 'free', 'new funnel']);
  });
});
