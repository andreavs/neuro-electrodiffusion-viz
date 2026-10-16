import { describe, expect, it } from 'vitest';
import { siteKey, type MeshGeometry } from '../../scales/emi/meshes';
import { makeEmiSolver, packFrame, type EmiGeometry, type EmiWasm, type PackState } from './emi';

/** Two cells, three membrane nodes, one membrane triangle and two slice triangles; records calls. */
class FakeSim implements EmiWasm {
  static clock = { t: 0, msPerStep: 4 };
  calls: string[] = [];
  time = 0;
  freed = false;
  constructor(readonly geometry: EmiGeometry) {}
  get perSide() {
    return this.geometry.kind === 'bundle' ? this.geometry.perSide : 0;
  }
  stimulate(which: string) {
    this.calls.push(`stimulate ${which}`);
  }
  stimulate_sphere(cell: number, x: number, y: number, z: number, r: number, g: number) {
    this.calls.push(`sphere ${cell} ${x},${y},${z} ${r} ${g}`);
    return 3;
  }
  add_probe(x: number, y: number, z: number, r: number) {
    this.calls.push(`add probe ${x},${y},${z} ${r}`);
    return 2;
  }
  n_probes = () => 2;
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
  bounds = () => Float32Array.of(0, 0, 0, 210, 1.4, 1.4);
  membrane_positions = () => new Float32Array(9);
  membrane_triangles = () => Uint32Array.of(0, 1, 2);
  phi_m = () => Float32Array.of(-65, -64, -63);
  slice_positions = (axis: number) => new Float32Array(18).fill(axis);
  slice_tags = () => Float32Array.of(0, 1);
  slice_values = (_axis: number, frac: number, field: string) =>
    new Float32Array(6).fill(field === 'k' ? 4 + frac : frac);
  take_traces = () => Float32Array.of(0, -65, -65, 0.1, -60, -65);
  free() {
    this.freed = true;
  }
}

const MESH: MeshGeometry = {
  bytes: new Uint8Array(4),
  meta: {
    label: 'Cell',
    description: '',
    stimuli: [
      { key: 'soma', label: 'Soma', cell: 1, centre: [0, 0, 0], radius: 5, g: 2 },
      { key: 'tip', label: 'Tip', cell: 1, centre: [0, 0, 9], radius: 1, g: 4 },
    ],
    probes: [{ label: 'Soma', centre: [1, 2, 3], radius: 4 }],
  },
};

function setup(params = {}, load = (_name: string) => Promise.resolve(MESH)) {
  const sims: FakeSim[] = [];
  FakeSim.clock.t = 0;
  const solver = makeEmiSolver(
    params,
    (geometry) => {
      const s = new FakeSim(geometry);
      sims.push(s);
      return s;
    },
    () => FakeSim.clock.t,
    load,
  );
  return { solver, sims };
}

const names = (f: { fields: { name: string }[] }) => f.fields.map((x) => x.name);

describe('packFrame', () => {
  it('sends the mesh and slice triangles once, then only the changing fields', () => {
    const sim = new FakeSim({ kind: 'bundle', perSide: 3, gap: 0.1 });
    const state: PackState = { geometry: 1, slice: 1, geometrySent: false, sliceSent: false };
    const params = { slice: 'across', slicePos: 0.25, field: 'k' };
    const first = packFrame(sim, params, state, 7);
    expect(names(first)).toEqual(['nodes', 'tris', 'slicePositions', 'sliceTags', 'phiM', 'slice', 'traces']);
    const f = Object.fromEntries(first.fields.map((x) => [x.name, x]));
    expect(f.tris!.shape).toEqual([1, 3]);
    expect(Array.from(f.tris!.data)).toEqual([0, 1, 2]);
    expect(f.slicePositions!.shape).toEqual([2, 3, 3]);
    expect(f.slicePositions!.data[0]).toBe(0); // across → normal along x
    expect(f.slice!.shape).toEqual([2, 3]);
    expect(f.slice!.data[0]).toBeCloseTo(4.25);
    expect(f.traces!.shape).toEqual([2, 3]);
    expect(first.steps).toBe(7);
    expect(first.stats).toMatchObject({ geometry: 1, slice: 1, nCells: 2, nProbes: 2, probeX: 150, stimulated: 2, loX: 0, hiX: 210 });

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
    expect(names(f)).toContain('tris');
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
    const solver = makeEmiSolver({}, (g) => {
      if (fail) throw new Error('bad geometry');
      const s = new FakeSim(g);
      sims.push(s);
      return s;
    });
    fail = true;
    expect(() => solver.setParams({ axons: '4' })).toThrow('bad geometry');
    expect(sims[0]!.freed).toBe(false);
    solver.dispose();
    expect(sims[0]!.freed).toBe(true);
  });

  it('loads a mesh geometry, then adds its probes and stimulates the chosen site', async () => {
    const loads: string[] = [];
    const { solver, sims } = setup({ geometry: 'cell', [siteKey('cell')]: 'tip' }, (name) => {
      loads.push(name);
      return Promise.resolve(MESH);
    });
    let woken = 0;
    solver.setWake!(() => woken++);
    expect(sims).toHaveLength(0);
    expect(solver.advance(12)).toBe('wait');
    expect(solver.frame().fields).toEqual([]);
    await Promise.resolve();
    expect(woken).toBe(1);
    expect(solver.advance(12)).toBe(true);
    expect(sims).toHaveLength(1);
    expect(sims[0]!.geometry.kind).toBe('mesh');
    expect(sims[0]!.calls).toEqual(['add probe 1,2,3 4', 'sphere 1 0,0,9 1 4']);
    expect(names(solver.frame())).toContain('tris');
    // Back to the bundle and to the mesh again: the mesh is not fetched twice.
    solver.setParams({ geometry: 'bundle' });
    solver.setParams({ geometry: 'cell', [siteKey('cell')]: 'nope' });
    expect(loads).toEqual(['cell']);
    expect(sims).toHaveLength(3);
    expect(sims[1]!.freed).toBe(true);
    expect(sims[2]!.calls.at(-1)).toBe('sphere 1 0,0,0 5 2'); // unknown site → the first
  });

  it('reports a failed mesh load on the next tick, unless the geometry moved on', async () => {
    const { solver } = setup({ geometry: 'cell' }, () => Promise.reject(new Error('404')));
    solver.setWake!(() => {});
    await Promise.resolve();
    await Promise.resolve();
    expect(() => solver.advance(12)).toThrow('404');

    const other = setup({ geometry: 'cell' }, () => Promise.reject(new Error('404'))).solver;
    other.setParams({ geometry: 'bundle' });
    await Promise.resolve();
    await Promise.resolve();
    expect(other.advance(12)).toBe(true);
  });

  it('builds a mesh as soon as it arrives, so a paused view gets it from frame()', async () => {
    const { solver, sims } = setup({ geometry: 'cell' });
    let woken = 0;
    solver.setWake!(() => woken++);
    await Promise.resolve();
    expect(woken).toBe(1);
    expect(sims).toHaveLength(1);
    expect(names(solver.frame())).toContain('tris');
  });

  it('reports a failed load from frame() too, and ignores loads that no longer matter', async () => {
    const failed = setup({ geometry: 'cell' }, () => Promise.reject(new Error('404'))).solver;
    failed.setWake!(() => {});
    await Promise.resolve();
    await Promise.resolve();
    expect(() => failed.frame()).toThrow('404');
    expect(failed.frame().stats).toEqual({ loading: 1 }); // reported once

    // A slow mesh A, then B: A arriving last must not replace B.
    const later: Record<string, (m: MeshGeometry) => void> = {};
    const { solver, sims } = setup({ geometry: 'a' }, (name) => new Promise((done) => (later[name] = done)));
    solver.setWake!(() => {});
    solver.setParams({ geometry: 'b' });
    later.b!(MESH);
    await Promise.resolve();
    later.a!(MESH);
    await Promise.resolve();
    expect(sims).toHaveLength(1);
    // Disposed while loading: a late arrival builds nothing.
    const gone = setup({ geometry: 'c' }, (name) => new Promise((done) => (later[name] = done)));
    gone.solver.dispose();
    later.c!(MESH);
    await Promise.resolve();
    expect(gone.sims).toHaveLength(0);
  });

  it('ignores bundle-only keys on a mesh geometry', async () => {
    const { solver, sims } = setup({ geometry: 'cell' });
    solver.setWake!(() => {});
    await Promise.resolve();
    solver.setParams({ axons: '4', gap: 0.3 });
    expect(sims).toHaveLength(1);
  });
});
