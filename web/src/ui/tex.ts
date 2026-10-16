import temml from 'temml';
import 'temml/dist/Temml-Local.css';

/** TeX → MathML. Browsers lay out MathML natively, so no fonts or layout pass are needed. */
export function renderTex(tex: string, display: boolean): string {
  return temml.renderToString(tex, { displayMode: display, throwOnError: false });
}
