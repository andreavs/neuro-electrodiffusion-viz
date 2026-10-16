import { describe, expect, it } from 'vitest';
import { parseRoute, routeHref } from './router';
import { SCALES } from './scales/registry';

describe('parseRoute', () => {
  it('defaults to the menu', () => {
    expect(parseRoute('')).toEqual({ kind: 'menu' });
    expect(parseRoute('#/')).toEqual({ kind: 'menu' });
    expect(parseRoute('#/nonsense')).toEqual({ kind: 'menu' });
  });

  it('round-trips every scale', () => {
    for (const s of SCALES) {
      const route = { kind: 'scale', id: s.id } as const;
      expect(parseRoute(routeHref(route))).toEqual(route);
    }
  });

  it('ignores trailing segments and query', () => {
    expect(parseRoute('#/emi/extra?x=1')).toEqual({ kind: 'scale', id: 'emi' });
  });
});
