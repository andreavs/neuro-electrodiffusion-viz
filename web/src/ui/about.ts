import type { ScaleId } from '../scales/registry';
import './about.css';

/**
 * Theory content for one scale. Prose fields are trusted HTML and may contain inline TeX
 * between `$…$`; `equations[].tex` is display-mode TeX. Math is rendered to MathML.
 */
export interface AboutContent {
  /** One or two paragraphs on what the model describes. */
  intro: string;
  equations: { tex: string; caption: string }[];
  assumptions: string[];
  parameters: { symbol: string; description: string; value: string }[];
  references: { text: string; href?: string }[];
}

export type TexRenderer = (tex: string, display: boolean) => string;

/** Each scale's content lives in `scales/<scale>/about.ts` and is loaded on demand. */
const CONTENT: Record<ScaleId, () => Promise<{ about: AboutContent }>> = {
  knp: () => import('../scales/knp/about'),
  emi: () => import('../scales/emi/about'),
  particles: () => import('../scales/particles/about'),
};

const INLINE_MATH = /\$([^$]+)\$/g;

function prose(html: string, tex: TexRenderer): string {
  return html.replace(INLINE_MATH, (_, src: string) => tex(src, false));
}

/** Build the panel body as HTML. Pure, so it can be tested without a DOM. */
export function renderAboutBody(c: AboutContent, tex: TexRenderer): string {
  const p = (html: string) => prose(html, tex);
  return `
    ${c.intro
      .split(/\n\s*\n/)
      .map((para) => `<p>${p(para.trim())}</p>`)
      .join('')}
    <h3>Equations</h3>
    ${c.equations
      .map(
        (e) => `
      <figure class="about-eq">
        <div class="about-scroll">${tex(e.tex, true)}</div>
        <figcaption>${p(e.caption)}</figcaption>
      </figure>`,
      )
      .join('')}
    <h3>Assumptions</h3>
    <ul>${c.assumptions.map((a) => `<li>${p(a)}</li>`).join('')}</ul>
    <h3>Parameters</h3>
    <div class="about-scroll">
      <table class="about-params">
        <thead><tr><th scope="col">Symbol</th><th scope="col">Meaning</th><th scope="col">Value</th></tr></thead>
        <tbody>${c.parameters
          .map(
            (r) =>
              `<tr><td>${tex(r.symbol, false)}</td><td>${p(r.description)}</td><td>${p(r.value)}</td></tr>`,
          )
          .join('')}</tbody>
      </table>
    </div>
    <h3>References</h3>
    <ul class="about-refs">${c.references
      .map((r) => `<li>${r.href ? `<a href="${r.href}" target="_blank" rel="noopener">${r.text}</a>` : r.text}</li>`)
      .join('')}</ul>`;
}

/** Load a scale's content and the TeX renderer (both split into lazy chunks). */
export async function loadAboutHtml(id: ScaleId): Promise<string> {
  const [{ about }, { renderTex }] = await Promise.all([CONTENT[id](), import('./tex')]);
  return renderAboutBody(about, renderTex);
}

/**
 * Collapsible "About this model" panel. Views append it wherever it fits their layout.
 *
 * The content and math renderer are fetched on first hover/focus, and the panel only opens
 * once its body is fully rendered, so opening never reflows half-typeset math.
 */
export function createAboutPanel(
  id: ScaleId,
  load: (id: ScaleId) => Promise<string> = loadAboutHtml,
): HTMLDetailsElement {
  const details = document.createElement('details');
  details.className = 'about';
  details.innerHTML =
    '<summary>About this model <span class="about-status" aria-live="polite"></span></summary>' +
    '<div class="about-body"></div>';
  const summary = details.querySelector('summary')!;
  const body = details.querySelector<HTMLElement>('.about-body')!;
  const status = summary.querySelector<HTMLElement>('.about-status')!;

  let state: 'idle' | 'loading' | 'ready' | 'error' = 'idle';
  let pending: Promise<void> | null = null;
  let selfOpened = false; // the click handler opened it after loading; don't reload on toggle
  const ensure = (): Promise<void> =>
    (pending ??= ((state = 'loading'), load(id)).then(
      (html) => {
        body.innerHTML = html;
        state = 'ready';
      },
      (err: unknown) => {
        const msg = document.createElement('p');
        msg.className = 'muted';
        msg.textContent = `Could not load the model description (${String(err)}).`;
        body.replaceChildren(msg);
        state = 'error';
        pending = null; // retry on the next open
      },
    ));

  summary.addEventListener('pointerenter', () => {
    if (state === 'idle') void ensure();
  });
  summary.addEventListener('focus', () => {
    if (state === 'idle') void ensure();
  });
  // Enter/Space on a <summary> dispatch click, so this covers keyboard users too.
  summary.addEventListener('click', (e) => {
    if (details.open || state === 'ready') return;
    e.preventDefault();
    summary.setAttribute('aria-busy', 'true');
    status.textContent = 'Loading…';
    void ensure().then(() => {
      summary.removeAttribute('aria-busy');
      status.textContent = '';
      selfOpened = true;
      details.open = true;
    });
  });
  // Opened some other way (e.g. programmatically) before anything was loaded: fill it in.
  details.addEventListener('toggle', () => {
    if (selfOpened) {
      selfOpened = false;
      return;
    }
    if (details.open && (state === 'idle' || state === 'error')) void ensure();
  });
  return details;
}
