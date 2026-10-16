import { describe, expect, it } from 'vitest';
import { makeSpreadingSolver, type SdWasm } from './spreading';

/** Each fake step costs 5 ms of wall clock. */
const clock = { t: 0 };

class FakeSd implements SdWasm {
  calls: string[] = [];
  n = 0;
  freed = false;
  constructor(
    readonly neurons: number,
    readonly gNap: number,
  ) {}
  set_drive = (count: number, g: number) => void this.calls.push(`drive ${count} ${g}`);
  set_pump = (s: number) => void this.calls.push(`pump ${s}`);
  set_buffering = (r: number) => void this.calls.push(`buffering ${r}`);
  step = (k: number) => void ((this.n += k), (clock.t += 5 * k));
  dt = () => 4;
  time_ms = () => 4 * this.n;
  substeps_per_neuron = () => 1;
  depolarised = () => 0.5;
  shape = () => Uint32Array.of(2, 2, 1);
  size = () => Float64Array.of(1000, 1000, 200);
  field = (name: string) => new Float32Array(4).fill(name === 'k' ? 3 : 0);
  somata = () => new Float32Array(3 * this.neurons);
  v_soma = () => new Float32Array(this.neurons).fill(-65);
  driven = () => Uint32Array.of(0);
  free = () => void (this.freed = true);
}

function setup() {
  const sims: FakeSd[] = [];
  const solver = makeSpreadingSolver(
    { neurons: 3 },
    (_seed, neurons, gNap) => {
      const s = new FakeSd(neurons, gNap);
      sims.push(s);
      return s;
    },
    () => clock.t,
  );
  return { solver, sims };
}

const names = (f: { fields: { name: string }[] }) => f.fields.map((x) => x.name);

describe('spreading-depression solver adapter', () => {
  it('creates the tissue with the defaults and converts buffering to 1/ms', () => {
    const { sims } = setup();
    expect(sims).toHaveLength(1);
    expect([sims[0]!.neurons, sims[0]!.gNap]).toEqual([3, 0.25]);
    expect(sims[0]!.calls).toEqual(['drive 1 20', 'pump 1', 'buffering 0']);
    const { solver, sims: s2 } = setup();
    solver.setParams({ buffering: 2 });
    expect(s2[0]!.calls.at(-1)).toBe('buffering 0.002');
  });

  it('sends the neuron positions once per build and after the drive changes', () => {
    const { solver } = setup();
    solver.advance(10);
    const first = solver.frame();
    expect(names(first)).toEqual(['k', 'v', 'somata', 'driven']);
    expect(first.fields[2]!.shape).toEqual([3, 3]);
    expect(first.stats).toMatchObject({ generation: 1, depolarised: 0.5, sizeZ: 200 });
    expect(names(solver.frame())).toEqual(['k', 'v']);
    solver.setParams({ driven: 5 });
    expect(names(solver.frame())).toContain('somata');
  });

  it('rebuilds (and frees the old tissue) only for parameters that fix the rest state', () => {
    const { solver, sims } = setup();
    solver.setParams({ pump: 1.5 });
    expect(sims).toHaveLength(1);
    solver.setParams({ gNap: 0.1 });
    expect(sims).toHaveLength(2);
    expect(sims[0]!.freed).toBe(true);
    expect(sims[1]!.gNap).toBe(0.1);
    expect(sims[1]!.calls).toContain('pump 1.5');
    expect(solver.frame().stats!.generation).toBe(2);
  });

  it('rejects an unknown field and keeps the old one', () => {
    const { solver } = setup();
    expect(() => solver.setParams({ field: 'q' })).toThrow();
    expect(solver.frame().fields[0]!.name).toBe('k');
  });
});
