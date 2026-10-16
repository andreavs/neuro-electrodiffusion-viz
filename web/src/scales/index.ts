import { createEmiView } from './emi/view';
import { createTissueView } from './knp/scenario';
import { createParticlesView } from './particles/view';
import type { ScaleId, ScaleView } from './registry';

const VIEWS: Record<ScaleId, () => ScaleView> = {
  knp: createTissueView,
  emi: createEmiView,
  particles: createParticlesView,
};

export function createView(id: ScaleId): ScaleView {
  return VIEWS[id]();
}
