import type { SimClient } from '../../sim/client';
import type { ScaleView } from '../registry';
import { createSpreadingView } from './spreading-view';
import { createKnpView } from './view';

/** Scenarios of the tissue view, chosen by the route (`#/knp`, `#/knp/spreading`). */
const SCENARIOS = [
  { path: '', label: 'One pyramidal cell', create: createKnpView },
  { path: 'spreading', label: 'Spreading depression', create: createSpreadingView },
] as const;

/** The scenario named by `hash` (`#/knp/<path>`), defaulting to the first. */
export function scenarioFor(hash: string): (typeof SCENARIOS)[number] {
  const path = hash.replace(/^#\/?/, '').split(/[?]/)[0]!.split('/')[1] ?? '';
  return SCENARIOS.find((s) => s.path === path) ?? SCENARIOS[0];
}

/** The tissue view: the scenario from the route, with links to switch between them. */
export function createTissueView(): ScaleView {
  const scenario = scenarioFor(location.hash);
  const inner = scenario.create();
  return {
    mount(root: HTMLElement, client: SimClient) {
      // Views fill `root` before their first await, so the bar can go on top straight away.
      const done = inner.mount(root, client);
      const nav = document.createElement('nav');
      nav.className = 'kv-scenarios';
      nav.setAttribute('aria-label', 'Scenario');
      for (const s of SCENARIOS) {
        const a = document.createElement('a');
        a.href = s.path ? `#/knp/${s.path}` : '#/knp';
        a.textContent = s.label;
        if (s === scenario) a.setAttribute('aria-current', 'page');
        nav.append(a);
      }
      root.prepend(nav);
      return done;
    },
    unmount: () => inner.unmount(),
  };
}
