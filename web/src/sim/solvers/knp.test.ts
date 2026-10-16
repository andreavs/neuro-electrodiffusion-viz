import { describe, expect, it } from 'vitest';
import { decodeProbes, encodeProbes } from '../../scales/knp/params';
import { TraceBuffer, makeKnpSolver, type KnpWasm } from './knp';

/** A 2×2×2 grid whose fields are filled with the step count; one step costs `msPerStep`. */
class FakeSim implements KnpWasm {
  calls: string[] = [];
  n = 0;
  freed = false;
  constructor(readonly clock: { t: number; msPerStep: number }) {}
  set_syn_rate = (hz: number) => void this.calls.push(`syn ${hz}`);
  set_stim = (nA: number) => void this.calls.push(`stim ${nA}`);
  set_dt = (dt: number) => void this.calls.push(`dt ${dt}`);
  dt = () => 1;
  step = (k: number) => void ((this.n += k), (this.clock.t += k * this.clock.msPerStep));
  time_ms = () => this.n;
  spikes = () => 3;
  shape = () => Uint32Array.of(2, 2, 2);
  size = () => Float64Array.of(40, 40, 80);
  field = (name: string) => new Float32Array(8).fill(name === 'k' ? this.n : -1);
  v = () => Float32Array.of(-65, -60);
  compartments = () => new Float32Array(14);
  take_trace = () => Float64Array.of(this.n - 0.5, -65, this.n, -64);
  probe = (xyz: Float64Array) => new Float32Array((xyz.length / 3) * 5).fill(7);
  free = () => void (this.freed = true);
}

function setup(msPerStep: number) {
  const clock = { t: 0, msPerStep };
  let sim: FakeSim | undefined;
  const solver = makeKnpSolver({}, () => (sim = new FakeSim(clock)), () => clock.t);
  return { solver, sim: sim!, clock };
}

describe('knp solver adapter', () => {
  it('applies the default parameters on creation', () => {
    const { sim } = setup(1);
    expect(sim.calls).toEqual(['syn 8000', 'stim 0', 'dt 1']);
  });
  it('takes at least one step, and as many as fit the budget', () => {
    const slow = setup(20);
    slow.solver.advance(12);
    expect(slow.sim.n).toBe(1);
    const fast = setup(2);
    fast.solver.advance(12);
    expect(fast.sim.n).toBe(6);
  });
  it('packs the shown field, the neuron, the trace and the probes', () => {
    const { solver, sim } = setup(5);
    solver.setParams({ probes: encodeProbes([[1, 2, 3], [4, 5, 6]]) });
    solver.advance(12);
    const f = solver.frame();
    const byName = Object.fromEntries(f.fields.map((x) => [x.name, x]));
    expect(byName.k!.shape).toEqual([2, 2, 2]);
    expect(Array.from(byName.k!.data)).toEqual(new Array(8).fill(sim.n));
    expect(byName.v!.shape).toEqual([2]);
    expect(byName.compartments!.shape).toEqual([2, 7]);
    expect(byName.trace!.shape).toEqual([2 * sim.n, 2]);
    expect(byName.probes!.shape).toEqual([2, 5]);
    expect(f.steps).toBe(sim.n);
    expect(f.stats).toMatchObject({ spikes: 3, sizeZ: 80 });
    // Every array has its own buffer, so the frame can be transferred.
    expect(new Set(f.fields.map((x) => x.data.buffer)).size).toBe(f.fields.length);
    // The trace is handed over once.
    expect(solver.frame().fields.find((x) => x.name === 'trace')!.shape).toEqual([0, 2]);
  });
  it('switches field and forwards only changed parameters', () => {
    const { solver, sim } = setup(5);
    sim.calls = [];
    solver.setParams({ field: 'phi', synRate: 2000 });
    expect(sim.calls).toEqual(['syn 2000']);
    expect(solver.frame().fields[0]!.name).toBe('phi');
    solver.dispose();
    expect(sim.freed).toBe(true);
  });
});

describe('probe encoding', () => {
  it('round-trips and drops malformed entries', () => {
    const s = encodeProbes([[1, 2.25, 3], [10, 20, 30]]);
    expect(decodeProbes(s)).toEqual([1, 2.3, 3, 10, 20, 30]);
    expect(decodeProbes('')).toEqual([]);
    expect(decodeProbes('1,2;a,b,c;4,5,6')).toEqual([4, 5, 6]);
  });
});

describe('TraceBuffer', () => {
  it('keeps the newest samples in order', () => {
    const b = new TraceBuffer(3);
    b.push(Float64Array.of(0, 10, 1, 11));
    b.push(Float64Array.of(2, 12, 3, 13));
    expect(Array.from(b.take())).toEqual([1, 11, 2, 12, 3, 13]);
    expect(b.take().length).toBe(0);
  });
});
