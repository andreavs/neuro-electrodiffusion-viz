import type { ParamSchema } from '../../ui/controls';

/** Fields the spreading-depression solver can send; values match `SdSim.field` names. */
export const SD_FIELDS = [
  { value: 'k', label: '[K⁺]', units: 'mM' },
  { value: 'na', label: '[Na⁺]', units: 'mM' },
  { value: 'x', label: '[X⁻]', units: 'mM' },
  { value: 'phi', label: 'φ', units: 'mV' },
] as const;

/** Parameters that fix the calibrated resting state: changing one rebuilds the tissue. */
export const SD_REBUILD_KEYS = ['neurons', 'gNap'] as const;

/** Controls of the spreading-depression scenario; read by the view and the worker solver. */
export const SD_PARAMS = [
  {
    kind: 'select',
    key: 'field',
    label: 'Shown field',
    options: SD_FIELDS.map(({ value, label }) => ({ value, label })),
    default: 'k',
  },
  {
    kind: 'number',
    key: 'driven',
    label: 'Driven neurons',
    min: 0,
    max: 1000,
    step: 1,
    default: 1,
    help: 'How many neurons at the centre get the excitatory drive',
  },
  {
    kind: 'number',
    key: 'drive',
    label: 'Drive',
    min: 0,
    max: 50,
    step: 1,
    default: 20,
    unit: 'nS',
    help: 'Na⁺ conductance added to each driven soma (20 nS holds it in depolarisation block)',
  },
  {
    kind: 'number',
    key: 'gNap',
    label: 'Persistent Na⁺',
    min: 0,
    max: 0.3,
    step: 0.01,
    default: 0.25,
    unit: 'mS/cm²',
    help: 'Persistent Na⁺ conductance of every cell; sets how close the tissue is to its threshold (restarts)',
  },
  {
    kind: 'number',
    key: 'pump',
    label: 'Na⁺/K⁺ pump',
    min: 0.5,
    max: 2,
    step: 0.05,
    default: 1,
    unit: '×',
    help: 'Pump capacity relative to the one that balances the resting cell',
  },
  {
    kind: 'number',
    key: 'buffering',
    label: 'Glial K⁺ uptake',
    min: 0,
    max: 5,
    step: 0.25,
    default: 0,
    unit: '1/s',
    help: 'Rate at which glia take up excess K⁺ (with Cl⁻)',
  },
  {
    kind: 'number',
    key: 'neurons',
    label: 'Neurons',
    min: 1000,
    max: 20_000,
    step: 1000,
    default: 10_000,
    help: 'Number of cells in the 1 × 1 × 0.2 mm slab (restarts)',
  },
] as const satisfies ParamSchema;
