import { placeholderView } from '../placeholder';
import { getScale, type ScaleView } from '../registry';

/** Replace with the real particles visualization. */
export function createParticlesView(): ScaleView {
  return placeholderView(getScale('particles')!);
}
