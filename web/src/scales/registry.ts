import type { SimClient } from '../sim/client';

export type ScaleId = 'knp' | 'emi' | 'particles';

export interface ScaleInfo {
  id: ScaleId;
  title: string;
  scale: string;
  summary: string;
}

/** A mounted simulation view. Each scale implements one. */
export interface ScaleView {
  /** Build DOM/WebGL inside `root` and start driving `client`. */
  mount(root: HTMLElement, client: SimClient): void | Promise<void>;
  /** Stop rendering, release GPU resources and listeners. */
  unmount(): void;
}

export const SCALES: readonly ScaleInfo[] = [
  {
    id: 'knp',
    title: 'Tissue scale: KNP with point-source neurons',
    scale: '~1 mm³ of tissue',
    summary:
      'Neurons act as point sources and sinks of ions in a tissue-averaged extracellular space. ' +
      'Concentrations evolve by the Kirchhoff-Nernst-Planck equations, with the potential set by electroneutrality.',
  },
  {
    id: 'emi',
    title: 'Cell scale: KNP-EMI with explicit membranes',
    scale: '~100 µm, axon bundles',
    summary:
      'Intracellular and extracellular spaces are resolved explicitly, coupled through a Hodgkin-Huxley membrane. ' +
      'Shows action potentials, ion accumulation and ephaptic coupling between neighbouring axons.',
  },
  {
    id: 'particles',
    title: 'Molecular scale: Brownian electrodiffusion',
    scale: '~1 µm, dendritic spine',
    summary:
      'Individual Ca²⁺ ions random-walk inside a dendritic spine with an electric drift term. ' +
      'Measure how the field changes the mean first passage time out of the spine.',
  },
];

export function getScale(id: string): ScaleInfo | undefined {
  return SCALES.find((s) => s.id === id);
}
