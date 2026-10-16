import type { ParamSchema } from '../../ui/controls';

/** Length of the axons, µm (fixed by the bindings). */
export const AXON_LENGTH = 200;
/** x where the axons start (extracellular space before their ends), µm. */
export const AXON_START = 5;

/** Controls of the cell-scale simulation; read by both the view and the worker solver. */
export const EMI_PARAMS = [
  {
    kind: 'select',
    key: 'axons',
    label: 'Axons',
    options: [
      { value: '2', label: '4 (2 × 2)' },
      { value: '3', label: '9 (3 × 3)' },
      { value: '4', label: '16 (4 × 4)' },
    ],
    default: '3',
  },
  {
    kind: 'number',
    key: 'gap',
    label: 'Spacing',
    min: 0.05,
    max: 0.5,
    step: 0.05,
    default: 0.1,
    unit: 'µm',
    help: 'Extracellular gap between neighbouring axons (the axons are 0.2 µm wide)',
  },
  {
    kind: 'select',
    key: 'stimulus',
    label: 'Stimulate',
    options: [
      { value: 'centre', label: 'Centre axon' },
      { value: 'corner', label: 'Corner axon' },
      { value: 'all', label: 'All axons' },
    ],
    default: 'centre',
    help: 'Which axons get a synaptic input at their left end',
  },
  {
    kind: 'number',
    key: 'period',
    label: 'Repeat stimulus every',
    min: 0,
    max: 50,
    step: 5,
    default: 20,
    unit: 'ms',
    help: '0 fires only once, at the start',
  },
  { kind: 'button', key: 'stimulate', label: 'Fire stimulus', help: 'Open the synapse on the chosen axons now' },
  {
    kind: 'select',
    key: 'field',
    label: 'Slice shows',
    options: [
      { value: 'phi', label: 'Potential φ' },
      { value: 'k', label: '[K⁺]' },
      { value: 'na', label: '[Na⁺]' },
      { value: 'cl', label: '[Cl⁻]' },
    ],
    default: 'phi',
  },
  {
    kind: 'select',
    key: 'slice',
    label: 'Slice',
    options: [
      { value: 'along', label: 'Along the axons' },
      { value: 'across', label: 'Across the bundle' },
    ],
    default: 'along',
  },
  {
    kind: 'number',
    key: 'slicePos',
    label: 'Slice position',
    min: 0,
    max: 1,
    step: 0.01,
    default: 0.5,
    help: 'Height of the lengthwise slice, or position of the cross-section along the axons',
  },
  { kind: 'toggle', key: 'inside', label: 'Show the slice inside the axons', default: false },
  {
    kind: 'number',
    key: 'probe',
    label: 'Probe position',
    min: 0,
    max: AXON_LENGTH,
    step: 5,
    default: 150,
    unit: 'µm',
    help: 'Where along the axons the membrane potential traces are recorded',
  },
  {
    kind: 'number',
    key: 'rate',
    label: 'Playback',
    min: 0.5,
    max: 20,
    step: 0.5,
    default: 2,
    unit: 'ms/s',
    help: 'Simulated milliseconds per second of wall time (an action potential lasts about 2 ms)',
  },
  {
    kind: 'number',
    key: 'dt',
    label: 'Time step',
    min: 0.025,
    max: 0.2,
    step: 0.025,
    default: 0.05,
    unit: 'ms',
  },
] as const satisfies ParamSchema;

/** Slice normal axis: z for the lengthwise slice, x for the cross-section. */
export function sliceAxis(slice: string): number {
  return slice === 'across' ? 0 : 2;
}

/** Slice fields in the order the frame's `sliceField` stat refers to. */
export const SLICE_FIELDS = ['phi', 'k', 'na', 'cl'];
