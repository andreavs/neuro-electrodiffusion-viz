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
