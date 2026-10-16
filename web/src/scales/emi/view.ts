import { placeholderView } from '../placeholder';
import { getScale, type ScaleView } from '../registry';

/** Replace with the real emi visualization. */
export function createEmiView(): ScaleView {
  return placeholderView(getScale('emi')!);
}
