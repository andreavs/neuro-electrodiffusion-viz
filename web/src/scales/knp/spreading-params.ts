import type { ParamSchema } from '../../ui/controls';

/** Fields the spreading-depression solver can send; values match `SdSim.field` names. */
export const SD_FIELDS = [
  { value: 'k', label: '[K⁺]e', units: 'mM' },
  { value: 'na', label: '[Na⁺]', units: 'mM' },
  { value: 'x', label: '[X⁻]', units: 'mM' },
  { value: 'phi', label: 'φ', units: 'mV' },
  { value: 'vg', label: 'Astrocyte V', units: 'mV' },
  { value: 'kg', label: 'Astrocyte [K⁺]', units: 'mM' },
] as const;

/** Fields that only exist with astrocytes; without them the solver shows [K⁺]e instead. */
export const SD_GLIA_FIELDS: readonly (typeof SD_FIELDS)[number]['value'][] = ['vg', 'kg'];

/**
 * Parameters that fix the calibrated resting state: changing one rebuilds the tissue
 * (`coupling` only while there are astrocytes).
 */
export const SD_REBUILD_KEYS = ['neurons', 'gNap', 'glia', 'coupling'] as const;

/** Controls of the spreading-depression scenario; read by the view and the worker solver. */
export const SD_PARAMS = [
  {
    kind: 'select',
    key: 'field',
    label: 'Shown field',
    options: SD_FIELDS.map(({ value, label }) => ({ value, label })),
    default: 'k',
    help: 'The astrocyte fields need Astrocytes above 0; without them [K⁺]e is shown',
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
    key: 'glia',
    label: 'Astrocytes',
    min: 0,
    max: 8,
    step: 0.5,
    default: 0,
    unit: 'µm²/µm³',
    help: 'Astrocytic membrane area per tissue volume, 0 for none (Halnes et al. 2013 use 8). Astrocytes take up K⁺ where it is high and release it through their gap junctions where it is low. They raise the ignition threshold: add driven neurons or persistent Na⁺ to start a wave (restarts)',
  },
  {
    kind: 'number',
    key: 'coupling',
    label: 'Gap junctions',
    min: 0.01,
    max: 1,
    step: 0.01,
    default: 1,
    unit: '×',
    help: 'Coupling of the astrocytic syncytium; lower it to mimic blocked gap junctions (restarts)',
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
