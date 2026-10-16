import './style.css';
import { parseRoute, routeHref } from './router';
import { createView } from './scales';
import type { ScaleView } from './scales/registry';
import { SimClient } from './sim/client';
import { renderMenu } from './ui/menu';

const app = document.querySelector<HTMLDivElement>('#app')!;
app.innerHTML = `
  <header class="topbar">
    <a href="${routeHref({ kind: 'menu' })}" class="brand">Neural Electrodiffusion</a>
  </header>
  <main id="view"></main>`;
const viewRoot = app.querySelector<HTMLElement>('#view')!;

let current: { view: ScaleView; client: SimClient } | null = null;

function teardown(): void {
  if (!current) return;
  current.view.unmount();
  current.client.terminate();
  current = null;
  viewRoot.replaceChildren();
}

function render(): void {
  teardown();
  const route = parseRoute(location.hash);
  if (route.kind === 'menu') {
    renderMenu(viewRoot);
    return;
  }
  const client = new SimClient();
  const view = createView(route.id);
  current = { view, client };
  void view.mount(viewRoot, client);
}

window.addEventListener('hashchange', render);
render();
