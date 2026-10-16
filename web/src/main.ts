import './style.css';
import { parseRoute, routeHref } from './router';
import { createView } from './scales';
import type { ScaleView } from './scales/registry';
import { SimClient } from './sim/client';
import { renderMenu } from './ui/menu';
import { RENDER_STYLES, setRenderStyle, watchRenderStyle, type RenderStyle } from './render/style';

const app = document.querySelector<HTMLDivElement>('#app')!;
app.innerHTML = `
  <header class="topbar">
    <a href="${routeHref({ kind: 'menu' })}" class="brand">Neural Electrodiffusion</a>
    <label class="style-pick">3-D style
      <select aria-label="3-D rendering style">
        ${RENDER_STYLES.map((s) => `<option value="${s.value}">${s.label}</option>`).join('')}
      </select>
    </label>
  </header>
  <main id="view"></main>`;
const viewRoot = app.querySelector<HTMLElement>('#view')!;
const stylePick = app.querySelector<HTMLSelectElement>('.style-pick select')!;
watchRenderStyle((s) => (stylePick.value = s));
stylePick.addEventListener('change', () => setRenderStyle(stylePick.value as RenderStyle));

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
