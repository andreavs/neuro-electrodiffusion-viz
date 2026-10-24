import { describe, expect, it } from 'vitest';
import { makeSpreadingSolver, type SdWasm } from './spreading';

/** Each fake step costs 5 ms of wall clock. */
const clock = { t: 0 };

class FakeSd implements SdWasm {
  calls: string[] = [];
  n = 0;
  freed = false;
  /** Makes `set_pump` reject its value, as the wasm range check does. */
  static badPump = false;
  constructor(
    readonly neurons: number,
    readonly gNap: number,
    readonly glia = 0,
    readonly coupling = 1,
  ) {}
  drivenCount = 0;
  set_drive = (count: number, g: number) => {
    this.drivenCount = count;
    this.calls.push(`drive ${count} ${g}`);
  };
  set_pump = (s: number) => {
    if (FakeSd.badPump) throw new Error('pump out of range');
    this.calls.push(`pump ${s}`);
  };
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
  driven = () => Uint32Array.from({ length: this.drivenCount }, (_, i) => i);
  free = () => void (this.freed = true);
}

function setup() {
  const sims: FakeSd[] = [];
  const solver = makeSpreadingSolver(
    { neurons: 3 },
    (_seed, neurons, gNap, glia, coupling) => {
      const s = new FakeSd(neurons, gNap, glia, coupling);
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
    expect([sims[0]!.neurons, sims[0]!.gNap, sims[0]!.glia, sims[0]!.coupling]).toEqual([3, 0.25, 0, 1]);
    expect(sims[0]!.calls).toEqual(['drive 1 20', 'pump 1', 'buffering 0']);
    const { solver, sims: s2 } = setup();
    solver.setParams({ buffering: 2 });
    expect(s2[0]!.calls.at(-1)).toBe('buffering 0.002');
  });

  it('sends the positions once per build and the driven cells after each drive change', () => {
    const { solver, sims } = setup();
    solver.advance(10);
    const first = solver.frame();
    expect(names(first)).toEqual(['k', 'v', 'somata', 'driven']);
    expect(first.fields[2]!.shape).toEqual([3, 3]);
    expect(first.stats).toMatchObject({ depolarised: 0.5, sizeZ: 200 });
    expect(names(solver.frame())).toEqual(['k', 'v']);
    solver.setParams({ driven: 2 });
    expect(sims[0]!.calls.at(-1)).toBe('drive 2 20');
    const next = solver.frame();
    expect(names(next)).toEqual(['k', 'v', 'driven']);
    expect([...next.fields[2]!.data]).toEqual([0, 1]);
  });

  it('rejects counts that wasm would wrap', () => {
    const { solver } = setup();
    expect(() => solver.setParams({ driven: -1 })).toThrow();
    expect(() => solver.setParams({ neurons: 2.5 })).toThrow();
  });

  it('keeps the old tissue and parameters when a rebuild fails', () => {
    const { solver, sims } = setup();
    const gen = solver.frame().stats!.generation;
    FakeSd.badPump = true;
    try {
      expect(() => solver.setParams({ gNap: 0.1, pump: 5 })).toThrow();
    } finally {
      FakeSd.badPump = false;
    }
    expect(sims).toHaveLength(2);
    expect(sims[1]!.freed).toBe(true);
    expect(sims[0]!.freed).toBe(false);
    // The failed build changed nothing: the next rebuild still starts from the old gNap.
    solver.setParams({ neurons: 4 });
    expect(sims[2]!.gNap).toBe(0.25);
    expect(solver.frame().stats!.generation).toBeGreaterThan(gen as number);
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
    const { solver: other } = setup();
    // Generations are unique across adapters, so a new `create` also reads as a rebuild.
    expect(new Set([solver.frame().stats!.generation, other.frame().stats!.generation]).size).toBe(2);
  });

  it('rebuilds with astrocytes and shows their fields only when they exist', () => {
    const { solver, sims } = setup();
    // Without astrocytes their fields fall back to [K⁺]e.
    solver.setParams({ field: 'vg' });
    expect(solver.frame().fields[0]!.name).toBe('k');
    expect(sims).toHaveLength(1);
    solver.setParams({ glia: 2, coupling: 0.5 });
    expect(sims).toHaveLength(2);
    expect([sims[1]!.glia, sims[1]!.coupling]).toEqual([2, 0.5]);
    expect(solver.frame().fields[0]!.name).toBe('vg');
    solver.setParams({ glia: 0 });
    expect(solver.frame().fields[0]!.name).toBe('k');
  });

  it('rejects an unknown field and keeps the old one', () => {
    const { solver } = setup();
    expect(() => solver.setParams({ field: 'q' })).toThrow();
    expect(solver.frame().fields[0]!.name).toBe('k');
  });
});
