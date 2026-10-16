import type { ParamSchema } from '../../ui/controls';

/** Fields the tissue solver can send; values match `KnpSim.field` names. */
export const KNP_FIELDS = [
  { value: 'k', label: '[K⁺]', units: 'mM' },
  { value: 'na', label: '[Na⁺]', units: 'mM' },
  { value: 'ca', label: '[Ca²⁺]', units: 'mM' },
  { value: 'x', label: '[X⁻]', units: 'mM' },
  { value: 'phi', label: 'φ', units: 'mV' },
] as const;

export type KnpField = (typeof KNP_FIELDS)[number]['value'];

/** Order of the values reported per probe (see `KnpSim.probe`). */
export const PROBE_ORDER: readonly KnpField[] = ['k', 'na', 'ca', 'x', 'phi'];

/** Controls of the tissue-scale simulation; read by both the view and the worker solver. */
export const KNP_PARAMS = [
  {
    kind: 'select',
    key: 'field',
    label: 'Shown field',
    options: KNP_FIELDS.map(({ value, label }) => ({ value, label })),
    default: 'k',
  },
  {
    kind: 'number',
    key: 'synRate',
    label: 'Synaptic input',
    min: 0,
    max: 30_000,
    step: 500,
    default: 8000,
    unit: 'Hz',
    help: 'Total rate of excitatory synaptic events on the dendrites (about 2 kHz gives 5 spikes/s)',
  },
  {
    kind: 'number',
    key: 'stim',
    label: 'Somatic current',
    min: -1,
    max: 3,
    step: 0.1,
    default: 0,
    unit: 'nA',
    help: 'Current injected into the soma through an electrode',
  },
  {
    kind: 'number',
    key: 'dt',
    label: 'Time step',
    min: 0.25,
    max: 5,
    step: 0.25,
    default: 1,
    unit: 'ms',
    help: 'KNP step; the neuron always steps at 0.025 ms. Larger runs faster',
  },
] as const satisfies ParamSchema;

/** Encodes probe positions (µm, grid coordinates) as the `probes` parameter. */
export function encodeProbes(points: readonly (readonly [number, number, number])[]): string {
  return points.map((p) => p.map((v) => v.toFixed(1)).join(',')).join(';');
}

/** Parses the `probes` parameter; malformed entries are dropped. */
export function decodeProbes(s: string): number[] {
  const out: number[] = [];
  for (const part of s.split(';')) {
    const v = part.split(',').map(Number);
    if (v.length === 3 && v.every(Number.isFinite)) out.push(...v);
  }
  return out;
}
