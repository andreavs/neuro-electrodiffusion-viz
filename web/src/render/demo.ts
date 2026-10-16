/**
 * Dev-only demo of the rendering toolkit (not linked from the app menu).
 * Run `npm run dev` and open http://localhost:5173/src/render/demo.html
 *
 * Animates a synthetic 64×64×32 field on three orthogonal slices, 1e5 points
 * random-walking in the box, a colorbar and a probe time series. Query
 * parameters: `?points=N` to change the particle count.
 */
import * as THREE from 'three';
import { Colorbar, PointCloud, SlicePlane, TimeSeriesPlot, coolwarm, createScene, setRange, viridis } from './index';
import type { Shape3 } from './field';

const NX = 64;
const NY = 64;
const NZ = 32;
const shape: Shape3 = [NX, NY, NZ];
const extent = new THREE.Vector3(2, 2, 1);
const nPoints = Number(new URLSearchParams(location.search).get('points') ?? 1e5);

const canvas = document.querySelector<HTMLCanvasElement>('#view')!;
const side = document.querySelector<HTMLElement>('#side')!;
const statsEl = document.querySelector<HTMLElement>('#stats')!;

const view = createScene(canvas, { background: 0x111216 });
view.frameBox(extent);
view.scene.add(new THREE.Box3Helper(new THREE.Box3(extent.clone().multiplyScalar(-0.5), extent.clone().multiplyScalar(0.5)), 0x666a78));

const field = new Float32Array(NX * NY * NZ);
const slices = (['x', 'y', 'z'] as const).map((axis) => {
  const s = new SlicePlane({ shape, extent, axis, colormap: coolwarm, smooth: view.floatLinear, opacity: 0.92 });
  view.scene.add(s.mesh);
  return s;
});
const colorbar = new Colorbar({
  colormap: coolwarm,
  label: 'φ',
  units: 'mV',
  onChange: (r) => slices.forEach((s) => setRange(s, r)),
});
side.append(colorbar.el);

const cloud = new PointCloud({ capacity: nPoints, size: 0.012, colormap: viridis, range: { min: -0.5, max: 0.5 } });
view.scene.add(cloud.points);
const xyz = new Float32Array(nPoints * 3);
const zval = new Float32Array(nPoints);
for (let i = 0; i < nPoints; i++) {
  xyz[3 * i] = (Math.random() - 0.5) * extent.x;
  xyz[3 * i + 1] = (Math.random() - 0.5) * extent.y;
  xyz[3 * i + 2] = (Math.random() - 0.5) * extent.z;
}

const plot = new TimeSeriesPlot(document.querySelector<HTMLCanvasElement>('#plot')!, {
  traces: [
    { name: 'φ at centre', color: '#7aa2ff' },
    { name: 'φ at corner', color: '#ff8a5c' },
  ],
  windowMs: 2000,
  xLabel: 'time (ms)',
  yLabel: 'mV',
});

/** Two orbiting charged blobs plus a travelling wave, in mV. */
function fillField(t: number): void {
  const ax = 0.5 * Math.cos(t * 1.3);
  const ay = 0.5 * Math.sin(t * 1.3);
  const bx = 0.6 * Math.cos(-t * 0.8 + 2);
  const by = 0.4 * Math.sin(-t * 0.8 + 2);
  let i = 0;
  for (let z = 0; z < NZ; z++) {
    const pz = ((z + 0.5) / NZ - 0.5) * extent.z;
    for (let y = 0; y < NY; y++) {
      const py = ((y + 0.5) / NY - 0.5) * extent.y;
      for (let x = 0; x < NX; x++, i++) {
        const px = ((x + 0.5) / NX - 0.5) * extent.x;
        const da = (px - ax) ** 2 + (py - ay) ** 2 + pz * pz;
        const db = (px - bx) ** 2 + (py - by) ** 2 + (pz - 0.2) ** 2;
        field[i] = 20 * Math.exp(-da * 12) - 14 * Math.exp(-db * 18) + 2 * Math.sin(6 * px - 3 * t);
      }
    }
  }
}

// xorshift32 in [-0.5, 0.5): cheaper than Math.random for 3e5 draws per frame.
let seed = 0x9e3779b9;
function rand(): number {
  seed ^= seed << 13;
  seed ^= seed >>> 17;
  seed ^= seed << 5;
  return (seed >>> 0) / 4294967296 - 0.5;
}

/** Random-walk steps with reflecting walls. */
function stepPoints(dt: number): void {
  const s = Math.sqrt(dt) * 0.14;
  const hx = extent.x / 2;
  const hy = extent.y / 2;
  const hz = extent.z / 2;
  for (let i = 0; i < nPoints; i++) {
    const j = 3 * i;
    let x = xyz[j] + s * rand();
    let y = xyz[j + 1] + s * rand();
    let z = xyz[j + 2] + s * rand();
    if (x > hx) x = 2 * hx - x;
    else if (x < -hx) x = -2 * hx - x;
    if (y > hy) y = 2 * hy - y;
    else if (y < -hy) y = -2 * hy - y;
    if (z > hz) z = 2 * hz - z;
    else if (z < -hz) z = -2 * hz - z;
    xyz[j] = x;
    xyz[j + 1] = y;
    xyz[j + 2] = z;
    zval[i] = z;
  }
}

const centre = NX / 2 + NX * (NY / 2 + NY * (NZ / 2));
const corner = 4 + NX * (4 + NY * (NZ / 2));
let simT = 0;
let frames = 0;
let fpsT0 = performance.now();
let synthAcc = 0;
let toolkitAcc = 0;
// synthMs: generating the fake data (the solver worker's job in the real app).
// toolkitMs: uploading it through the toolkit (slices, colorbar, points, plot).
const stats = { fps: 0, synthMs: 0, toolkitMs: 0, frames: 0 };
(window as unknown as { __demoStats: typeof stats }).__demoStats = stats;

view.start((dtMs, now) => {
  const c0 = performance.now();
  const dt = Math.min(dtMs, 50) / 1000;
  simT += dt;
  fillField(simT);
  stepPoints(dt);
  const c1 = performance.now();
  for (const s of slices) s.update(field);
  colorbar.update(field);
  cloud.update(xyz, nPoints, zval);
  plot.push(0, simT * 1000, field[centre]);
  plot.push(1, simT * 1000, field[corner]);
  plot.draw();
  const c2 = performance.now();
  synthAcc += c1 - c0;
  toolkitAcc += c2 - c1;
  frames++;
  stats.frames++;
  if (now - fpsT0 >= 1000) {
    stats.fps = (frames * 1000) / (now - fpsT0);
    stats.synthMs = synthAcc / frames;
    stats.toolkitMs = toolkitAcc / frames;
    statsEl.textContent =
      `${stats.fps.toFixed(1)} fps\n` +
      `synthetic data ${stats.synthMs.toFixed(2)} ms/frame\ntoolkit update ${stats.toolkitMs.toFixed(2)} ms/frame\n` +
      `${nPoints.toLocaleString()} points, ${NX}×${NY}×${NZ} field\nDPR ${view.renderer.getPixelRatio()}`;
    frames = 0;
    synthAcc = 0;
    toolkitAcc = 0;
    fpsT0 = now;
  }
});

// Slice scrubbing with the keyboard: x/y/z then [ and ].
let active = 2;
window.addEventListener('keydown', (e) => {
  if (e.key === 'x' || e.key === 'y' || e.key === 'z') active = 'xyz'.indexOf(e.key);
  if (e.key === '[') slices[active].index--;
  if (e.key === ']') slices[active].index++;
});

import.meta.hot?.dispose(() => {
  view.dispose();
  colorbar.dispose();
  plot.dispose();
});
