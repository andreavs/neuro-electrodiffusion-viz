import { placeholderView } from '../placeholder';
import { getScale, type ScaleView } from '../registry';

/** Replace with the real knp visualization. */
export function createKnpView(): ScaleView {
  return placeholderView(getScale('knp')!);
}
