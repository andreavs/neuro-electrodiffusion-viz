import { getScale, type ScaleId } from './scales/registry';

export type Route = { kind: 'menu' } | { kind: 'scale'; id: ScaleId };

/** Hash routes (`#/knp`) so deep links work on GitHub Pages without server rewrites. */
export function parseRoute(hash: string): Route {
  const id = hash.replace(/^#\/?/, '').split(/[/?]/)[0] ?? '';
  const scale = getScale(id);
  return scale ? { kind: 'scale', id: scale.id } : { kind: 'menu' };
}

export function routeHref(route: Route): string {
  return route.kind === 'menu' ? '#/' : `#/${route.id}`;
}
