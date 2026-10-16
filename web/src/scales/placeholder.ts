import type { SimClient } from '../sim/client';
import type { ScaleInfo, ScaleView } from './registry';

/** Stand-in view until a scale's real visualization lands. */
export function placeholderView(info: ScaleInfo): ScaleView {
  return {
    async mount(root: HTMLElement, client: SimClient) {
      root.innerHTML = `
        <section class="placeholder" data-placeholder>
          <h2>${info.title}</h2>
          <p>${info.summary}</p>
          <p class="muted" data-status>Loading solver…</p>
        </section>`;
      const status = root.querySelector<HTMLElement>('[data-status]')!;
      try {
        const version = await client.ready();
        status.textContent = `Solver core v${version} loaded. This simulation is under construction.`;
      } catch (err) {
        status.textContent = `Failed to load solver: ${String(err)}`;
      }
    },
    unmount() {},
  };
}
