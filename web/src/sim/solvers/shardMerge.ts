/** Pure helpers for splitting an ion population over shard workers and pooling the results. */

/** Fewer ions than this per shard and the messaging costs more than the extra core gains. */
const MIN_IONS_PER_SHARD = 10_000;

/**
 * Shards for `n` ions on a machine with `cores` hardware threads. Two threads are left for
 * the page and the browser's compositor; busier shards starve them and the page stutters.
 */
export function shardCount(n: number, cores: number): number {
  const byCores = Math.max(1, Math.min(16, cores - 2));
  return Math.max(1, Math.min(byCores, Math.ceil(n / MIN_IONS_PER_SHARD)));
}

/** Splits `n` into `k` near-equal parts, larger ones first. */
export function splitEvenly(n: number, k: number): number[] {
  return Array.from({ length: k }, (_, i) => Math.floor(n / k) + (i < n % k ? 1 : 0));
}

/**
 * Combines per-shard `[mean, se, n, censored_mean]` exit-time summaries into the summary of
 * the pooled sample (the SE from the pooled sum of squared deviations).
 */
export function mergeMfpt(parts: Float64Array[]): Float64Array {
  const used = parts.filter((p) => p[2]! > 0);
  const n = used.reduce((a, p) => a + p[2]!, 0);
  if (n === 0) return Float64Array.of(NaN, NaN, 0, NaN);
  const mean = used.reduce((a, p) => a + p[0]! * p[2]!, 0) / n;
  const censored = used.reduce((a, p) => a + p[3]! * p[2]!, 0) / n;
  let ss = 0;
  for (const p of used) {
    const [m, se, k] = [p[0]!, p[1]!, p[2]!];
    // se² = s²/k with s² = Σ(t − m)²/(k − 1); a single exit has no spread of its own.
    if (k > 1) ss += se * se * k * (k - 1);
    ss += k * (m - mean) ** 2;
  }
  const se = n > 1 ? Math.sqrt(ss / (n - 1) / n) : NaN;
  return Float64Array.of(mean, se, n, censored);
}

/** Sum of per-shard histograms (fractions of each shard's ions), as a fraction of all ions. */
export function mergeHistograms(parts: { histogram: Float32Array; released: number }[], bins: number): Float32Array {
  const out = new Float32Array(bins);
  const total = parts.reduce((a, p) => a + p.released, 0) || 1;
  for (const p of parts) for (let i = 0; i < bins; i++) out[i]! += ((p.histogram[i] ?? 0) * p.released) / total;
  return out;
}
