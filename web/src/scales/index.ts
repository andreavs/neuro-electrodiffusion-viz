import { createEmiView } from './emi/view';
import { createKnpView } from './knp/view';
import { createParticlesView } from './particles/view';
import type { ScaleId, ScaleView } from './registry';

const VIEWS: Record<ScaleId, () => ScaleView> = {
  knp: createKnpView,
  emi: createEmiView,
  particles: createParticlesView,
};

export function createView(id: ScaleId): ScaleView {
  return VIEWS[id]();
}
