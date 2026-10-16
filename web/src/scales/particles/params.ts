import type { ParamSchema } from '../../ui/controls';

/** Controls of the molecular-scale simulation; read by both the view and the worker solver. */
export const PARTICLE_PARAMS = [
  {
    kind: 'select',
    key: 'geometry',
    label: 'Spine shape',
    options: [
      { value: 'sharp', label: 'Sharp neck-head junction' },
      { value: 'funnel', label: 'Funneled junction' },
    ],
    default: 'sharp',
  },
  {
    kind: 'select',
    key: 'field',
    label: 'Electric field',
    options: [
      { value: 'off', label: 'Off' },
      { value: 'neck', label: 'Across the neck' },
      { value: 'uniform', label: 'Uniform along the spine' },
    ],
    default: 'neck',
  },
  {
    kind: 'number',
    key: 'dv',
    label: 'Potential drop',
    min: 0,
    max: 60,
    step: 1,
    default: 20,
    unit: 'mV',
    help: 'Potential difference between the spine head and the base of the neck',
  },
  {
    kind: 'select',
    key: 'direction',
    label: 'Field direction',
    options: [
      { value: 'out', label: 'Out (pushes Ca²⁺ out)' },
      { value: 'in', label: 'In (holds Ca²⁺ back)' },
    ],
    default: 'out',
  },
  {
    kind: 'number',
    key: 'particles',
    label: 'Ca²⁺ ions',
    min: 1000,
    max: 100_000,
    step: 1000,
    default: 5000,
    help: 'Rendering stays smooth at 100 000 ions, but the simulation runs slower',
  },
  {
    kind: 'number',
    key: 'dt',
    label: 'Time step',
    min: 1e-4,
    max: 2e-3,
    step: 1e-4,
    default: 1e-3,
    unit: 'ms',
    help: 'Smaller is more accurate near the walls; larger runs faster',
  },
  { kind: 'button', key: 'release', label: 'Release ions', help: 'Start a new release from the head centre' },
] as const satisfies ParamSchema;

/** Zero-field comparison population (only simulated while a field is on). */
export const REFERENCE_PARTICLES = 2000;

/** Signed head-minus-base potential drop in mV; positive pushes cations out. */
export function signedDrop(dv: number, direction: string): number {
  return direction === 'in' ? -dv : dv;
}
