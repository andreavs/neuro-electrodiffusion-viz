import { routeHref } from '../router';
import { SCALES } from '../scales/registry';

export function renderMenu(root: HTMLElement): void {
  root.innerHTML = `
    <section class="menu">
      <p class="lead">
        Ion concentrations shape how neurons signal. Pick a scale to simulate electrodiffusion
        in the browser, from millimetres of tissue down to single ions.
      </p>
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
