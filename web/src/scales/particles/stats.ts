/** Pure helpers for the particle view's plots and readouts. */

export interface MfptReadout {
  /** Best current estimate, ms. */
  mean: number;
  se: number;
  /** True once every ion has escaped, so the estimate is the sample mean. */
  final: boolean;
}

/**
 * MFPT ± SE from frame stats (`prefix` '' for the main population, 'ref' for the reference).
 * While ions remain inside it uses the censored estimate (time spent inside by all ions /
 * escapes), whose SE for an exponential tail is mean / √escapes.
 */
export function mfptReadout(stats: Record<string, number>, prefix: '' | 'ref', remaining: number): MfptReadout | null {
  const key = (k: string) => (prefix ? prefix + k : k[0]!.toLowerCase() + k.slice(1));
  const escaped = stats[key('Escaped')] ?? 0;
  if (!(escaped > 0)) return null;
  if (remaining === 0) return { mean: stats[key('Mfpt')]!, se: stats[key('MfptSe')]!, final: true };
  const mean = stats[key('MfptCensored')]!;
  return { mean, se: mean / Math.sqrt(escaped), final: false };
}

export function formatMfpt(r: MfptReadout | null): string {
  if (!r) return '–';
  const digits = r.mean >= 100 ? 0 : r.mean >= 10 ? 1 : 2;
  const se = Number.isFinite(r.se) ? ` ± ${r.se.toFixed(digits)}` : '';
  return `${r.final ? '' : '≈ '}${r.mean.toFixed(digits)}${se} ms`;
}

/** Outline of a histogram over [0, tMax] as interleaved (t, y) vertices of a step line. */
export function histogramSteps(bins: ArrayLike<number>, tMax: number): Float32Array {
  const n = bins.length;
  const out = new Float32Array(4 * n + 4);
  let k = 0;
  out[k++] = 0;
  out[k++] = 0;
  for (let i = 0; i < n; i++) {
    out[k++] = (i * tMax) / n;
    out[k++] = bins[i]!;
    out[k++] = ((i + 1) * tMax) / n;
    out[k++] = bins[i]!;
  }
  out[k++] = tMax;
  out[k++] = 0;
  return out;
}

/** Largest value in `arrays`, at least `floor`. */
export function maxOf(arrays: (ArrayLike<number> | undefined)[], floor: number): number {
  let m = floor;
  for (const a of arrays) if (a) for (let i = 0; i < a.length; i++) if (a[i]! > m) m = a[i]!;
  return m;
}
