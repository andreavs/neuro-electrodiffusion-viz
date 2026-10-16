/**
 * Colormaps as 256-entry RGBA8 lookup tables.
 *
 * - `viridis`: perceptually uniform sequential map for concentrations.
 * - `inferno`: sequential map from black, for the glow style's dark stage.
 * - `icefire`: diverging map through black, for changes of either sign on a dark stage.
 * - `coolwarm`: diverging blue → light grey → red map for potentials, built by
 *   linear interpolation in CIELAB so lightness rises monotonically to the
 *   centre and falls monotonically after it.
 *
 * Pure module (no three.js) so it can be unit-tested in Node.
 */

export const LUT_SIZE = 256;

export interface Colormap {
  name: string;
  /** `LUT_SIZE` RGBA8 entries, row-major. */
  lut: Uint8Array;
  /** True when the map is meant to be centred on zero. */
  diverging: boolean;
}

type RGB = [number, number, number];

/** Polynomial fit to matplotlib's viridis (sRGB in [0, 1]), by Matt Zucker (CC0). */
function viridisAt(t: number): RGB {
  const c0: RGB = [0.2777273272234177, 0.005407344544966578, 0.3340998053353061];
  const c1: RGB = [0.1050930431085774, 1.404613529898575, 1.384590162594685];
  const c2: RGB = [-0.3308618287255563, 0.214847559468213, 0.09509516302823659];
  const c3: RGB = [-4.634230498983486, -5.799100973351585, -19.33244095627987];
  const c4: RGB = [6.228269936347081, 14.17993336680509, 56.69055260068105];
  const c5: RGB = [4.776384997670288, -13.74514537774601, -65.35303263337234];
  const c6: RGB = [-5.435455855934631, 4.645852612178535, 26.3124352495832];
  const out: RGB = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    out[i] = c0[i] + t * (c1[i] + t * (c2[i] + t * (c3[i] + t * (c4[i] + t * (c5[i] + t * c6[i])))));
  }
  return out;
}

/** Polynomial fit to matplotlib's inferno (sRGB in [0, 1]), by Matt Zucker (CC0). */
function infernoAt(t: number): RGB {
  const c0: RGB = [0.0002189403691192265, 0.001651004631001012, -0.01948089843709184];
  const c1: RGB = [0.1065134194856116, 0.5639564367884091, 3.932712388889277];
  const c2: RGB = [11.60249308247187, -3.972853965665698, -15.9423941062914];
  const c3: RGB = [-41.70399613139459, 17.43639888205313, 44.35414519872813];
  const c4: RGB = [77.162935699427, -33.40235894210092, -81.80730925738993];
  const c5: RGB = [-71.31942824499214, 32.62606426397723, 73.20951985803202];
  const c6: RGB = [25.13112622477341, -12.24266895238567, -23.07032500287172];
  const out: RGB = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    out[i] = c0[i] + t * (c1[i] + t * (c2[i] + t * (c3[i] + t * (c4[i] + t * (c5[i] + t * c6[i])))));
  }
  return out;
}

function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

function linearToSrgb(c: number): number {
  return c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055;
}

// D65 white point.
const XN = 0.95047;
const YN = 1.0;
const ZN = 1.08883;

function labF(t: number): number {
  return t > 216 / 24389 ? Math.cbrt(t) : (24389 / 27 * t + 16) / 116;
}

function labFInv(t: number): number {
  return t ** 3 > 216 / 24389 ? t ** 3 : (116 * t - 16) / (24389 / 27);
}

/** sRGB in [0, 1] → CIELAB. */
export function srgbToLab([r, g, b]: RGB): RGB {
  const R = srgbToLinear(r);
  const G = srgbToLinear(g);
  const B = srgbToLinear(b);
  const x = (0.4124564 * R + 0.3575761 * G + 0.1804375 * B) / XN;
  const y = (0.2126729 * R + 0.7151522 * G + 0.072175 * B) / YN;
  const z = (0.0193339 * R + 0.119192 * G + 0.9503041 * B) / ZN;
  const fx = labF(x);
  const fy = labF(y);
  const fz = labF(z);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

function labToSrgb([L, a, b]: RGB): RGB {
  const fy = (L + 16) / 116;
  const x = XN * labFInv(fy + a / 500);
  const y = YN * labFInv(fy);
  const z = ZN * labFInv(fy - b / 200);
  const R = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
  const G = -0.969266 * x + 1.8760108 * y + 0.041556 * z;
  const B = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;
  return [linearToSrgb(Math.max(0, R)), linearToSrgb(Math.max(0, G)), linearToSrgb(Math.max(0, B))];
}

/** CIELAB lightness L* (0-100) of an RGBA8 LUT entry. */
export function lightnessAt(lut: Uint8Array, i: number): number {
  return srgbToLab([lut[4 * i] / 255, lut[4 * i + 1] / 255, lut[4 * i + 2] / 255])[0];
}

function buildLut(f: (t: number) => RGB): Uint8Array {
  const lut = new Uint8Array(LUT_SIZE * 4);
  for (let i = 0; i < LUT_SIZE; i++) {
    const c = f(i / (LUT_SIZE - 1));
    for (let k = 0; k < 3; k++) lut[4 * i + k] = Math.round(Math.min(1, Math.max(0, c[k])) * 255);
    lut[4 * i + 3] = 255;
  }
  return lut;
}

function coolwarmAt(t: number): RGB {
  // Endpoints and centre from Moreland's "cool to warm" map.
  const lo = srgbToLab([59 / 255, 76 / 255, 192 / 255]);
  const mid = srgbToLab([221 / 255, 221 / 255, 221 / 255]);
  const hi = srgbToLab([180 / 255, 4 / 255, 38 / 255]);
  const [a, b, s] = t < 0.5 ? [lo, mid, t * 2] : [mid, hi, (t - 0.5) * 2];
  return labToSrgb([a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, a[2] + (b[2] - a[2]) * s]);
}

/** Piecewise-linear interpolation through evenly spaced sRGB stops. */
function stopsAt(stops: RGB[]): (t: number) => RGB {
  return (t) => {
    const x = Math.min(1, Math.max(0, t)) * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(x));
    const f = x - i;
    const [a, b] = [stops[i]!, stops[i + 1]!];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
  };
}

const icefireAt = stopsAt([
  [0.75, 0.97, 1.0],
  [0.2, 0.62, 0.95],
  [0.12, 0.2, 0.55],
  [0.0, 0.0, 0.0],
  [0.55, 0.08, 0.12],
  [0.95, 0.42, 0.12],
  [1.0, 0.93, 0.55],
]);

export const viridis: Colormap = { name: 'viridis', lut: buildLut(viridisAt), diverging: false };
/** Black → purple → orange → pale yellow: reads as glowing on a dark stage. */
export const inferno: Colormap = { name: 'inferno', lut: buildLut(infernoAt), diverging: false };
/** Diverging through black (cyan ← blue ← black → red → gold): both signs glow on a dark stage. */
export const icefire: Colormap = { name: 'icefire', lut: buildLut(icefireAt), diverging: true };
export const coolwarm: Colormap = { name: 'coolwarm', lut: buildLut(coolwarmAt), diverging: true };

/** Map `t` in [0, 1] (clamped) to an RGBA8 entry, written into `out`. */
export function sampleColormap(cm: Colormap, t: number, out: Uint8Array | number[] = [0, 0, 0, 0]) {
  const x = Number.isFinite(t) ? Math.min(1, Math.max(0, t)) : 0;
  const i = Math.round(x * (LUT_SIZE - 1)) * 4;
  out[0] = cm.lut[i];
  out[1] = cm.lut[i + 1];
  out[2] = cm.lut[i + 2];
  out[3] = cm.lut[i + 3];
  return out;
}

/** CSS `linear-gradient` stops (bottom → top when used with `to top`). */
export function colormapGradient(cm: Colormap, stops = 16): string {
  const parts: string[] = [];
  for (let s = 0; s < stops; s++) {
    const t = s / (stops - 1);
    const [r, g, b] = sampleColormap(cm, t);
    parts.push(`rgb(${r} ${g} ${b}) ${(t * 100).toFixed(1)}%`);
  }
  return parts.join(', ');
}
