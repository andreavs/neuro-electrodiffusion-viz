import { routeHref } from '../router';
import { SCALES } from '../scales/registry';

export function renderMenu(root: HTMLElement): void {
  root.innerHTML = `
    <section class="menu">
      <p class="lead">
        Ion concentrations shape how neurons signal. Pick a scale to simulate electrodiffusion
        in the browser, from millimetres of tissue down to single ions.
      </p>
      <div class="intro">
        <p>
          Neurons signal by letting ions such as Na⁺, K⁺, Cl⁻ and Ca²⁺ cross their membranes.
          Standard models treat the concentrations on either side as fixed and track only voltage.
          In reality the ions move by <strong>electrodiffusion</strong>: they diffuse down their
          concentration gradients and drift in the electric field that the ions themselves create.
          Most of the brain is packed with cells and only about 20% is extracellular space, so
          activity can shift concentrations noticeably.
        </p>
        <p>
          Sometimes that matters a lot. In cortical spreading depression, associated with migraine
          aura, a wave of hyperactivity sweeps across the cortex at 1.5 to 9.5 millimetres per minute.
          It spreads to whichever neurons are nearby rather than along synaptic connections, and
          its front is marked by a sharp rise in extracellular K⁺. Capturing effects like this
          needs models in which concentrations are part of the dynamics.
        </p>
      </div>
      <div class="cards">
        ${SCALES.map(
          (s) => `
          <a class="card" href="${routeHref({ kind: 'scale', id: s.id })}">
            <span class="card-scale">${s.scale}</span>
            <h2>${s.title}</h2>
            <p>${s.summary}</p>
          </a>`,
        ).join('')}
      </div>
      <p class="muted">
        Based on the PhD thesis <em>Modeling Electrical and Diffusive Transports in Neural Tissue</em>
        (A. V. Solbrå, University of Oslo, 2019).
      </p>
    </section>`;
}
