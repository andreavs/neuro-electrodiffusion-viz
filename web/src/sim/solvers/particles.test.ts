import { describe, expect, it } from 'vitest';
import {
  HISTOGRAM_BINS,
  batchSize,
  makeParticlesSolver,
  niceCeil,
  packFrame,
  type ParticlesWasm,
} from './particles';

/** Records calls; each step removes one ion until none are left. */
class FakeSim implements ParticlesWasm {
  /** Wall-clock cost of one step on the shared fake clock. */
  static clock = { t: 0, msPerStep: 0 };
  calls: string[] = [];
  alive = 0;
  released = 0;
  ref = false;
  time = 0;
  freed = false;
  constructor(
    readonly geometry: string,
    readonly dendrite = false,
  ) {}
  set_field(kind: string, dv: number) {
    this.calls.push(`field ${kind} ${dv}`);
    this.ref = kind !== 'off' && dv !== 0;
  }
  set_dt(dt: number) {
    this.calls.push(`dt ${dt}`);
  }
  release(n: number, nRef: number) {
    this.calls.push(`release ${n} ${nRef}`);
    this.alive = this.released = n;
    this.time = 0;
  }
  step(n: number) {
    const k = Math.min(n, this.alive);
    this.alive -= k;
    FakeSim.clock.t += k * FakeSim.clock.msPerStep;
    this.time += k * 0.5;
    return k;
  }
  time_ms = () => this.time;
  /** Last exit time; follows the clock unless a test sets it. */
  passage: number | null = null;
  passage_time_ms = () => this.passage ?? this.time;
  n_alive = () => this.alive;
  n_in_dendrite = () => 0;
  n_released = () => this.released;
  has_reference = () => this.ref;
  positions = () => new Float32Array(3 * this.alive);
  regions = () => new Float32Array(this.alive);
  survival = () => Float32Array.of(0, 1, this.time, this.alive / this.released);
  histogram = (_ref: boolean, bins: number) => new Float32Array(bins);
  mfpt = (ref: boolean) => Float64Array.of(ref ? 2 : 1, 0.1, 3, 1.5);
  profile = () => Float32Array.of(0.1, 0, 0.1, 1, 0, 1.5);
  neck_top = () => 1;
  neck_radius = () => 0.1;
  dendrite_radius = () => (this.dendrite ? 0.45 : 0);
  dendrite_half_length = () => (this.dendrite ? 3 : 0);
  free() {
    this.freed = true;
  }
}

function setup(params = {}) {
  const sims: FakeSim[] = [];
  const solver = makeParticlesSolver(
    params,
    (geometry, dendrite) => {
      const s = new FakeSim(geometry, dendrite);
      sims.push(s);
      return s;
    },
    () => FakeSim.clock.t,
  );
  return { solver, sims };
}

describe('niceCeil', () => {
  it('rounds up to 1, 2 or 5 times a power of ten', () => {
    expect([0, 0.3, 1, 1.01, 2.5, 7, 10, 340].map(niceCeil)).toEqual([1, 0.5, 1, 2, 5, 10, 10, 500]);
  });
});

describe('batchSize', () => {
  it('fits the remaining budget, at least one step and at most 256', () => {
    expect(batchSize(10, 0)).toBe(1);
    expect(batchSize(10, 2)).toBe(5);
    expect(batchSize(1, 2)).toBe(1);
    expect(batchSize(1000, 0.001)).toBe(256);
  });
});

describe('packFrame', () => {
  it('packs positions, regions, curves and MFPT stats with consistent shapes', () => {
    const sim = new FakeSim('sharp');
    sim.set_field('neck', 20);
    sim.release(4, 2);
    sim.step(1);
    const f = packFrame(sim, 7, 2);
    expect(f.steps).toBe(7);
    expect(f.timeMs).toBe(0.5);
    const byName = Object.fromEntries(f.fields.map((x) => [x.name, x]));
    expect(Object.keys(byName).sort()).toEqual(
      ['histogram', 'histogramRef', 'positions', 'profile', 'regions', 'survival', 'survivalRef'].sort(),
    );
    expect(byName.positions!.shape).toEqual([3, 3]);
    expect(byName.regions!.shape).toEqual([3]);
    expect(byName.survival!.shape).toEqual([2, 2]);
    expect(byName.histogram!.shape).toEqual([HISTOGRAM_BINS]);
    expect(byName.profile!.shape).toEqual([3, 2]);
    for (const x of f.fields) expect(x.data.length).toBe(x.shape.reduce((a, b) => a * b, 1));
    expect(f.stats).toMatchObject({ alive: 3, released: 4, histTMax: 0.5, hasReference: 1, refReleased: 2, mfpt: 1, refMfpt: 2 });
  });

  it('ends the exit-time axis at the last exit, not the current time', () => {
    const sim = new FakeSim('sharp', true);
    sim.release(4, 0);
    sim.time = 7;
    sim.passage = 3;
    const f = packFrame(sim, 1, 0);
    expect(f.timeMs).toBe(7);
    expect(f.stats).toMatchObject({ histTMax: 5, passageTime: 3 });
  });

  it('omits the reference curves while the field is off', () => {
    const sim = new FakeSim('sharp');
    sim.set_field('off', 0);
    sim.release(2, 2);
    const f = packFrame(sim, 0, 2);
    expect(f.fields.map((x) => x.name)).not.toContain('survivalRef');
    expect(f.stats?.hasReference).toBe(0);
  });
});

describe('makeParticlesSolver', () => {
  it('starts with the schema defaults and a capped reference population', () => {
    const { sims } = setup({ particles: 50_000 });
    expect(sims[0]!.geometry).toBe('sharp');
    expect(sims[0]!.calls).toEqual(['dt 0.001', 'field neck 20', 'release 50000 2000']);
  });

  it('re-releases on parameter changes and rebuilds only for a new geometry', () => {
    const { solver, sims } = setup();
    solver.setParams({ direction: 'in' });
    expect(sims).toHaveLength(1);
    expect(sims[0]!.calls.slice(-2)).toEqual(['field neck -20', 'release 5000 2000']);
    solver.setParams({ geometry: 'funnel' });
    expect(sims).toHaveLength(2);
    expect(sims[0]!.freed).toBe(true);
    expect(sims[1]!.geometry).toBe('funnel');
    expect(sims[1]!.calls.at(-1)).toBe('release 5000 2000');
  });

  it('rebuilds with a dendrite when ions should spread into it', () => {
    const { solver, sims } = setup();
    expect(sims[0]!.dendrite).toBe(false);
    solver.setParams({ outlet: 'dendrite' });
    expect(sims).toHaveLength(2);
    expect(sims[1]!.dendrite).toBe(true);
    const f = solver.frame();
    expect(f.stats).toMatchObject({ dendriteRadius: 0.45, dendriteHalfLength: 3, inDendrite: 0 });
  });

  it('stops before a step that would overrun the budget', () => {
    FakeSim.clock.msPerStep = 2;
    const { solver } = setup();
    for (let tick = 0; tick < 5; tick++) {
      const t0 = FakeSim.clock.t;
      expect(solver.advance(9)).toBe(true);
      expect(FakeSim.clock.t - t0).toBeLessThanOrEqual(9);
      expect(FakeSim.clock.t - t0).toBeGreaterThanOrEqual(6);
    }
    // After a restart (cost unknown) a step costlier than the budget runs once.
    FakeSim.clock.msPerStep = 20;
    solver.action!('release');
    const t0 = FakeSim.clock.t;
    solver.advance(9);
    expect(FakeSim.clock.t - t0).toBe(20);
    FakeSim.clock.msPerStep = 0;
  });

  it('advances until every ion has escaped, then reports no change, and counts steps', () => {
    const { solver, sims } = setup({ particles: 1000 });
    while (solver.advance(50) !== false);
    expect(sims[0]!.alive).toBe(0);
    expect(solver.frame().steps).toBe(1000);
    solver.action!('release');
    expect(sims[0]!.alive).toBe(1000);
    expect(solver.frame().steps).toBe(0);
    solver.dispose();
    expect(sims[0]!.freed).toBe(true);
  });
});
