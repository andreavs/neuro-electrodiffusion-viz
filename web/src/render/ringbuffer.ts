/**
 * Fixed-capacity ring buffer of (t, value) samples for time-series probes.
 * Pushing beyond capacity overwrites the oldest sample. No allocation after
 * construction.
 */
export class RingBuffer {
  readonly capacity: number;
  private readonly t: Float64Array;
  private readonly v: Float32Array;
  private head = 0; // next write position
  private count = 0;

  constructor(capacity: number) {
    if (!Number.isInteger(capacity) || capacity < 1) throw new RangeError('capacity must be a positive integer');
    this.capacity = capacity;
    this.t = new Float64Array(capacity);
    this.v = new Float32Array(capacity);
  }

  get length(): number {
    return this.count;
  }

  push(t: number, value: number): void {
    this.t[this.head] = t;
    this.v[this.head] = value;
    this.head = (this.head + 1) % this.capacity;
    if (this.count < this.capacity) this.count++;
  }

  clear(): void {
    this.head = 0;
    this.count = 0;
  }

  /** i-th sample in chronological order (0 = oldest). */
  timeAt(i: number): number {
    return this.t[this.physical(i)];
  }

  valueAt(i: number): number {
    return this.v[this.physical(i)];
  }

  /** Time of the newest sample, or NaN when empty. */
  get lastTime(): number {
    return this.count ? this.timeAt(this.count - 1) : NaN;
  }

  /** Index (chronological) of the first sample with time >= `t0`. */
  firstIndexAtOrAfter(t0: number): number {
    let lo = 0;
    let hi = this.count;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (this.timeAt(mid) < t0) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  private physical(i: number): number {
    if (i < 0 || i >= this.count) throw new RangeError(`index ${i} out of range [0, ${this.count})`);
    return (this.head - this.count + i + this.capacity) % this.capacity;
  }
}
