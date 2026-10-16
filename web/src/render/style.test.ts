import { describe, expect, it } from 'vitest';
import { getRenderStyle, setRenderStyle, watchRenderStyle, type RenderStyle } from './style';

describe('render style', () => {
  it('defaults to glow and notifies watchers until they unsubscribe', () => {
    expect(getRenderStyle()).toBe('glow');
    const seen: RenderStyle[] = [];
    const stop = watchRenderStyle((s) => seen.push(s));
    setRenderStyle('classic');
    setRenderStyle('classic');
    stop();
    setRenderStyle('glow');
    expect(seen).toEqual(['glow', 'classic']);
    expect(getRenderStyle()).toBe('glow');
  });
});
