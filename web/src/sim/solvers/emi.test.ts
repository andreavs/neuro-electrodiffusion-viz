import { describe, expect, it } from 'vitest';
import { makeEmiSolver, packFrame, type EmiWasm, type PackState } from './emi';

/** Two cells, three membrane nodes, one membrane quad and two slice quads; records calls. */
class FakeSim implements EmiWasm {
  static clock = { t: 0, msPerStep: 4 };
  calls: string[] = [];
  time = 0;
  freed = false;
  constructor(readonly perSide: number) {}
  stimulate(which: string) {
    this.calls.push(`stimulate ${which}`);
  }
  set_dt(dt: number) {
    this.calls.push(`dt ${dt}`);
  }
  set_probe(x: number) {
    this.calls.push(`probe ${x}`);
  }
  step(n: number) {
    this.time += 0.1 * n;
    FakeSim.clock.t += n * FakeSim.clock.msPerStep;
    return n;
  }
  time_ms = () => this.time;
  n_cells = () => 2;
  probe_x = () => 150;
  stimulated = () => 2;
  extent = () => Float32Array.of(210, 1.4, 1.4);
  membrane_positions = () => new Float32Array(9);
  membrane_quads = () => Uint32Array.of(0, 1, 2, 0);
  phi_m = () => Float32Array.of(-65, -64, -63);
  slice_positions = (axis: number) => new Float32Array(24).fill(axis);
  slice_tags = () => Float32Array.of(0, 1);
  slice_values = (_axis: number, frac: number, field: string) =>
    new Float32Array(8).fill(field === 'k' ? 4 + frac : frac);
  take_traces = () => Float32Array.of(0, -65, -65, 0.1, -60, -65);
  free() {
    this.freed = true;
  }
}

function setup(params = {}) {
  const sims: FakeSim[] = [];
  FakeSim.clock.t = 0;
  const solver = makeEmiSolver(
    params,
    (perSide) => {
      const s = new FakeSim(perSide);
      sims.push(s);
      return s;
    },
    () => FakeSim.clock.t,
  );
  return { solver, sims };
}

const names = (f: { fields: { name: string }[] }) => f.fields.map((x) => x.name);

describe('packFrame', () => {
  it('sends the mesh and slice quads once, then only the changing fields', () => {
    const sim = new FakeSim(3);
    const state: PackState = { geometry: 1, slice: 1, geometrySent: false, sliceSent: false };
    const params = { slice: 'across', slicePos: 0.25, field: 'k' };
    const first = packFrame(sim, params, state, 7);
    expect(names(first)).toEqual(['nodes', 'quads', 'slicePositions', 'sliceTags', 'phiM', 'slice', 'traces']);
    const f = Object.fromEntries(first.fields.map((x) => [x.name, x]));
    expect(f.quads!.shape).toEqual([1, 4]);
    expect(Array.from(f.quads!.data)).toEqual([0, 1, 2, 0]);
    expect(f.slicePositions!.shape).toEqual([2, 4, 3]);
    expect(f.slicePositions!.data[0]).toBe(0); // across → normal along x
    expect(f.slice!.shape).toEqual([2, 4]);
    expect(f.slice!.data[0]).toBeCloseTo(4.25);
    expect(f.traces!.shape).toEqual([2, 3]);
    expect(first.steps).toBe(7);
    expect(first.stats).toMatchObject({ geometry: 1, slice: 1, nCells: 2, probeX: 150, stimulated: 2, extentX: 210 });

    expect(names(packFrame(sim, params, state, 8))).toEqual(['phiM', 'slice', 'traces']);
  });
});

describe('makeEmiSolver', () => {
  it('builds with the defaults, sets the probe and fires the stimulus at t = 0', () => {
    const { sims } = setup();
    expect(sims).toHaveLength(1);
    expect(sims[0]!.perSide).toBe(3);
    expect(sims[0]!.calls).toEqual(['probe 150', 'stimulate centre']);
  });

  it('steps at the playback rate, within the budget', () => {
    const { solver, sims } = setup({ rate: 20, dt: 0.1 });
    FakeSim.clock.msPerStep = 4;
    expect(solver.advance(12)).toBe(true); // first call only starts the clock
    expect(sims[0]!.time).toBe(0);
    FakeSim.clock.t += 50; // owes 1 ms = 10 steps, the budget allows 3
    solver.advance(12);
    expect(sims[0]!.time).toBeCloseTo(0.3);
    solver.advance(100); // the remaining 0.7 ms, plus what the 12 ms just spent added
    expect(sims[0]!.time).toBeCloseTo(1.2);
    FakeSim.clock.t += 10_000; // a long pause catches up on at most 100 ms of wall time
    solver.advance(1e6);
    expect(sims[0]!.time).toBeCloseTo(3.2, 0);
  });

  it('repeats the stimulus with the chosen period', () => {
    const { solver, sims } = setup({ rate: 20, dt: 0.1, period: 1 });
    FakeSim.clock.msPerStep = 0;
    solver.advance(12);
    for (let i = 0; i < 25; i++) {
      FakeSim.clock.t += 5;
      solver.advance(12);
    }
    expect(sims[0]!.time).toBeCloseTo(2.5);
    expect(sims[0]!.calls.filter((c) => c.startsWith('stimulate'))).toHaveLength(3);
  });

  it('rebuilds on geometry changes and resends the mesh', () => {
    const { solver, sims } = setup();
    const before = solver.frame().stats!;
    solver.setParams({ gap: 0.2 });
    expect(sims).toHaveLength(2);
    expect(sims[0]!.freed).toBe(true);
    const f = solver.frame();
    expect(names(f)).toContain('quads');
    // Versions never repeat, not even across solvers (a Reset builds a new one).
    expect(f.stats!.geometry).toBeGreaterThan(before.slice!);
    expect(setup().solver.frame().stats!.geometry).toBeGreaterThan(f.stats!.slice!);
  });

  it('updates in place for dt, probe, slice and stimulus', () => {
    const { solver, sims } = setup();
    const before = solver.frame().stats!;
    solver.setParams({ dt: 0.05, probe: 50, field: 'na' });
    solver.action!('stimulate');
    expect(sims).toHaveLength(1);
    expect(sims[0]!.calls.slice(2)).toEqual(['dt 0.05', 'probe 50', 'stimulate centre']);
    expect(names(solver.frame())).toEqual(['phiM', 'slice', 'traces']);
    solver.setParams({ slicePos: 0.3 });
    const f = solver.frame();
    expect(names(f)).toContain('slicePositions');
    expect(f.stats!.slice).toBeGreaterThan(before.slice!);
    expect(f.stats!.geometry).toBe(before.geometry);
  });

  it('keeps the old simulation when a rebuild fails', () => {
    let fail = false;
    const sims: FakeSim[] = [];
    const solver = makeEmiSolver({}, (n) => {
      if (fail) throw new Error('bad geometry');
      const s = new FakeSim(n);
      sims.push(s);
      return s;
    });
    fail = true;
    expect(() => solver.setParams({ axons: '4' })).toThrow('bad geometry');
    expect(sims[0]!.freed).toBe(false);
    solver.dispose();
    expect(sims[0]!.freed).toBe(true);
  });
});
