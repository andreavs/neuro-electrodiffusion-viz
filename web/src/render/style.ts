/**
 * The app-wide 3-D rendering style: `classic` (flat colormapped slices and
 * surfaces, as in the thesis figures) or `glow` (dark stage, emissive
 * surfaces and bloom). Remembered per browser.
 */
export type RenderStyle = 'classic' | 'glow';

export const RENDER_STYLES: readonly { value: RenderStyle; label: string }[] = [
  { value: 'glow', label: 'Glow' },
  { value: 'classic', label: 'Classic' },
];

const KEY = 'ed-render-style';
const listeners = new Set<(style: RenderStyle) => void>();
let current: RenderStyle = load();

function load(): RenderStyle {
  try {
    const v = globalThis.localStorage?.getItem(KEY);
    if (v === 'classic' || v === 'glow') return v;
  } catch {
    // Storage blocked (private mode, sandboxed frame): use the default.
  }
  return 'glow';
}

export function getRenderStyle(): RenderStyle {
  return current;
}

export function setRenderStyle(style: RenderStyle): void {
  if (style === current) return;
  current = style;
  try {
    globalThis.localStorage?.setItem(KEY, style);
  } catch {
    // Not persisted; the choice still applies to this page.
  }
  for (const f of [...listeners]) f(style);
}

/** Call `f` now and on every change. Returns an unsubscribe function. */
export function watchRenderStyle(f: (style: RenderStyle) => void): () => void {
  listeners.add(f);
  f(current);
  return () => listeners.delete(f);
}
