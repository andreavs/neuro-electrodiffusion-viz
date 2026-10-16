import type { Params } from '../../sim/protocol';

/** One stop of the guided tour: a preset for the controls and what it shows. */
export interface TourStep {
  title: string;
  /** Trusted HTML (written here, not user input). */
  body: string;
  params: Params;
}

const cite = (label: string, doi: string) => `<a href="https://doi.org/${doi}" target="_blank" rel="noopener">${label}</a>`;
const HARNETT = cite('Harnett et al. 2012', '10.1038/nature11554');
const JAYANT = cite('Jayant et al. 2017', '10.1038/nnano.2016.268');
const SABATINI = cite('Sabatini et al. 2002', '10.1016/S0896-6273(02)00573-1');
const HOLCMAN = cite('Holcman &amp; Schuss 2011', '10.1186/2190-8567-1-10');
const TONNESEN = cite('Tønnesen et al. 2014', '10.1038/nn.3682');
const POPOVIC = cite('Popovic et al. 2015', '10.1038/ncomms9436');

/** Every step sets all of these, so steps do not depend on the order they are visited in. */
const BASE: Params = { geometry: 'sharp', field: 'off', dv: 25, direction: 'out', particles: '20000', dt: 1e-3 };

export const TOUR: readonly TourStep[] = [
  {
    title: 'A handful of ions',
    body: `A spine head holds only a few free Ca²⁺ ions: at the resting 70 nM, a typical 0.05 µm³ head has about 2 (the larger head drawn here, about 20), and after an action potential about 100 stay free while the rest are buffered (${SABATINI}). With so few ions, each release escapes on its own schedule. Press <em>Release ions</em> a few times and watch the MFPT jump.`,
    params: { ...BASE, particles: '100' },
  },
  {
    title: 'The neck is a bottleneck',
    body: `With 20 000 ions the randomness averages out. Ions released in the head need about 46 ms to find the narrow neck and leave, close to the narrow-escape estimate V·L/(πa²D) ≈ 42 ms (${HOLCMAN}). This is how spines keep Ca²⁺, the trigger for plasticity, to themselves.`,
    params: { ...BASE },
  },
  {
    title: 'The neck is also a resistor',
    body: `During an EPSP the spine head can depolarise by about 25 mV while the dendrite barely moves (${HARNETT}; ${JAYANT}). That drop sits across a neck about 1 µm long. For Ca²⁺ (z = 2) the drift beats diffusion once zΔV exceeds the thermal voltage, at about 13 mV. Here the MFPT roughly halves, to about 22 ms; the dashed curves show the same spine without the field.`,
    params: { ...BASE, field: 'neck' },
  },
  {
    title: 'Held back',
    body: `Reverse the drop (head more negative than the dendrite) and the field traps the ions: the MFPT grows to about 130 ms. Diffusion alone cannot tell these cases apart; only electrodiffusion responds to the sign of the voltage.`,
    params: { ...BASE, field: 'neck', direction: 'in' },
  },
  {
    title: 'A tighter neck, a stronger push',
    body: `Narrowing the neck slows diffusive escape (τ ∝ L/a²) but raises the neck resistance by the same factor, and with it the voltage drop for a given synaptic current. When drift dominates, the two cancel: the escape time then depends on the head volume and the current, not the neck shape. Necks are plastic, widening by about 30% after LTP (${TONNESEN}). The size of the effect hinges on the neck resistance, which is still debated, from about 30 MΩ (${POPOVIC}) to about 500 MΩ (${HARNETT}). Try 5 mV against 40 mV with the slider. This step also switches to the paper's funnelled junction, which changes little next to the neck itself.`,
    params: { ...BASE, field: 'neck', geometry: 'funnel' },
  },
  {
    title: 'Checking against the paper',
    body: `Publication 3 validated the method with a 30 kV/m field along the whole spine, about a 60 mV drop over its 2 µm. MCell gave an MFPT of 8.1 ms for this spine against 46.8 ms without the field; this solver gives about 8.0 ms.`,
    params: { ...BASE, field: 'uniform', dv: 60 },
  },
  {
    title: 'A million ions',
    body: `More ions make the statistics sharper: the uncertainty of the MFPT falls as 1/√N, so a million ions pin it down to about 0.1% and resolve the tail of the exit-time distribution. The ions are spread over one worker per spare CPU core; 100 000 of them are drawn. Expect the clock to run slowly: each step moves every ion.`,
    params: { ...BASE, field: 'neck', particles: '1000000' },
  },
];

export interface TourOptions {
  /** Apply a step's parameters; returns whether anything changed. */
  apply(params: Params): boolean;
  /** Start a fresh release (used when a step's parameters were already set). */
  release(): void;
}

/**
 * The tour box: an intro, then the current step's text and buttons to move between steps.
 * Choosing a step also resumes a paused run, since a step is something to watch.
 */
export function createTour(opts: TourOptions, steps: readonly TourStep[] = TOUR): HTMLElement {
  const el = document.createElement('section');
  el.className = 'pv-tour';
  el.setAttribute('aria-label', 'Guided tour');
  el.innerHTML = `
    <div class="pv-tour-head">
      <span class="pv-tour-label"></span>
      <span class="pv-tour-nav">
        <button type="button" class="ctl-button" data-tour="prev">Back</button>
        <button type="button" class="ctl-button" data-tour="close">Close</button>
        <button type="button" class="ctl-button ctl-primary" data-tour="next"></button>
      </span>
    </div>
    <div class="pv-tour-step" aria-live="polite"><h3></h3><p></p></div>`;
  const q = <T extends HTMLElement>(sel: string) => el.querySelector<T>(sel)!;
  const label = q('.pv-tour-label');
  const prev = q<HTMLButtonElement>('[data-tour="prev"]');
  const close = q<HTMLButtonElement>('[data-tour="close"]');
  const next = q<HTMLButtonElement>('[data-tour="next"]');
  const stepEl = q('.pv-tour-step');
  let index = -1;

  // The buttons are kept (only their text and state change), so keyboard focus stays put.
  const render = () => {
    const touring = index >= 0;
    prev.hidden = close.hidden = stepEl.hidden = !touring;
    next.textContent = touring ? 'Next' : 'Start the tour';
    if (!touring) {
      label.innerHTML = `<strong>Why does charge matter in a spine?</strong>
        <span class="muted">A ${steps.length}-step tour of what the electric field does to Ca²⁺ in a dendritic spine.</span>`;
      next.disabled = false;
      return;
    }
    const step = steps[index]!;
    label.innerHTML = `<span class="muted">Guided tour · ${index + 1} of ${steps.length}</span>`;
    prev.disabled = index === 0;
    next.disabled = index === steps.length - 1;
    q('h3').textContent = step.title;
    q('p').innerHTML = step.body;
    // Keep focus on a usable button when the one just pressed is disabled at either end.
    if (document.activeElement instanceof HTMLButtonElement && document.activeElement.disabled)
      (prev.disabled ? next : prev).focus();
  };

  const show = (i: number) => {
    index = i;
    render();
    if (!opts.apply(steps[i]!.params)) opts.release();
  };

  el.addEventListener('click', (ev) => {
    const b = (ev.target as HTMLElement).closest<HTMLButtonElement>('[data-tour]');
    if (!b) return;
    if (b.dataset.tour === 'close') {
      index = -1;
      render();
      next.focus();
      return;
    }
    show(Math.max(0, Math.min(steps.length - 1, index + (b.dataset.tour === 'next' ? 1 : -1))));
  });
  render();
  return el;
}
