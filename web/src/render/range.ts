/** Value ranges for colour mapping. Pure, allocation-free. */

export interface Range {
  min: number;
  max: number;
}

/**
 * Finite min/max of `data` (NaN/±Inf ignored). With `symmetric`, returns
 * [-m, m] where m = max |v|, so a diverging map stays centred on zero.
 * Returns {0, 1} for empty or all-non-finite input and pads degenerate ranges.
 */
export function autoRange(data: ArrayLike<number>, symmetric = false, out: Range = { min: 0, max: 1 }): Range {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  // Comparisons with NaN are false, so NaNs never update lo/hi; ±Inf would.
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
    lo = Infinity;
    hi = -Infinity;
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (!Number.isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (lo > hi) {
    out.min = 0;
    out.max = 1;
    return out;
  }
  if (symmetric) {
    const m = Math.max(Math.abs(lo), Math.abs(hi));
    lo = -m;
    hi = m;
  }
  if (hi - lo <= 1e-12 * Math.max(1, Math.abs(hi))) {
    const pad = Math.max(1e-6, Math.abs(hi) * 1e-3);
    lo -= pad;
    hi += pad;
  }
  out.min = lo;
  out.max = hi;
  return out;
}

/**
 * A range that grows to fit new data but never shrinks until `reset()`, so
 * auto-scaled axes and colour bars stay still instead of jittering frame to
 * frame. Each expansion adds `headroom` (a fraction of the new span) beyond
 * the new extreme, so slow drifts don't widen the range on every frame.
 */
export class ExpandingRange implements Range {
  min = Infinity;
  max = -Infinity;

  constructor(readonly headroom = 0.05) {}

  get empty(): boolean {
    return !(this.max >= this.min);
  }

  reset(): void {
    this.min = Infinity;
    this.max = -Infinity;
  }

  /** Grow to cover [lo, hi] (non-finite bounds are ignored). Returns true if the range changed. */
  include(lo: number, hi: number): boolean {
    if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo > hi) return false;
    if (this.empty) {
      this.min = lo;
      this.max = hi;
      return true;
    }
    if (lo >= this.min && hi <= this.max) return false;
    // The first fit is exact; later expansions overshoot a little.
    const pad = (Math.max(hi, this.max) - Math.min(lo, this.min)) * this.headroom;
    if (lo < this.min) this.min = lo - pad;
    if (hi > this.max) this.max = hi + pad;
    return true;
  }
}

/**
 * Colour range centred on `rest`, wide enough for the fraction `q` of the deviations
 * from `rest` in every `stride`-th value of `field` outside the masked cells (such as point
 * sources, whose values are far off the rest), and at least ±0.1 % of `rest`.
 * `scratch` is reused between calls.
 */
export function restRange(
  field: Float32Array,
  mask: Uint8Array,
  rest: number,
  q: number,
  stride: number,
  scratch: Float32Array,
): [number, number] {
  let n = 0;
  for (let i = 0; i < field.length && n < scratch.length; i += stride) {
    const d = Math.abs(field[i]! - rest);
    if (!mask[i] && Number.isFinite(d)) scratch[n++] = d;
  }
  const devs = scratch.subarray(0, n).sort();
  const d = Math.max(n ? devs[Math.min(n - 1, Math.floor(q * n))]! : 0, Math.abs(rest) * 1e-3, 1e-9);
  return [rest - d, rest + d];
}
