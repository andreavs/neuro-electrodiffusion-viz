/** Loaders for the realistic cell geometries in public/meshes (made by tools/meshes/generate.py). */

/** A sphere on the membrane, µm. */
export interface MeshSite {
  label: string;
  centre: [number, number, number];
  radius: number;
}

/** A mesh file's companion JSON (written by tools/meshes/generate.py). */
export interface MeshMeta {
  label: string;
  description: string;
  /** Synapse sites: `g` mS/cm² on the membrane of `cell` within the sphere. */
  stimuli: (MeshSite & { key: string; cell: number; g: number })[];
  /** Probes: mean φ_M over the membrane within the sphere. */
  probes: MeshSite[];
}

export interface MeshGeometry {
  bytes: Uint8Array;
  meta: MeshMeta;
}

export type MeshLoader = (name: string) => Promise<MeshGeometry>;

const moduleUrl = import.meta.url;

/**
 * Where the public file `meshes/<name>.<ext>` is served. With the relative base of the build,
 * resolve from this module's chunk in assets/ (a worker's own URL is not the page's).
 */
function meshUrl(name: string, ext: string): string {
  if (!/^[a-z0-9-]+$/.test(name)) throw new Error(`unknown geometry ${JSON.stringify(name)}`);
  const base = import.meta.env.BASE_URL;
  const root = base.startsWith('/') ? new URL(base, globalThis.location?.origin) : new URL('..', moduleUrl);
  return new URL(`meshes/${name}.${ext}`, root).href;
}

const metas = new Map<string, Promise<MeshMeta>>();

/** The JSON of mesh `name` (fetched once). */
export function fetchMeshMeta(name: string): Promise<MeshMeta> {
  let meta = metas.get(name);
  if (!meta) {
    meta = fetch(meshUrl(name, 'json')).then((r) => {
      if (!r.ok) throw new Error(`could not load the ${name} mesh (HTTP ${r.status})`);
      return r.json() as Promise<MeshMeta>;
    });
    meta.catch(() => metas.delete(name));
    metas.set(name, meta);
  }
  return meta;
}

/** Param holding the chosen stimulus site of mesh geometry `name`. */
export const siteKey = (name: string) => `site:${name}`;

const geometries = new Map<string, Promise<MeshGeometry>>();

/** Fetches `meshes/<name>.edmesh` and its JSON from the app's public folder (once). */
export const fetchMesh: MeshLoader = (name) => {
  let mesh = geometries.get(name);
  if (!mesh) {
    mesh = Promise.all([fetch(meshUrl(name, 'edmesh')), fetchMeshMeta(name)]).then(async ([bin, meta]) => {
      if (!bin.ok) throw new Error(`could not load the ${name} mesh (HTTP ${bin.status})`);
      return { bytes: new Uint8Array(await bin.arrayBuffer()), meta };
    });
    mesh.catch(() => geometries.delete(name));
    geometries.set(name, mesh);
  }
  return mesh;
};
