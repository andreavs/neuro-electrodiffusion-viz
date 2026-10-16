import { ExpandingRange } from './range';
import { RingBuffer } from './ringbuffer';
import { formatTick, niceTicks } from './ticks';

export interface TraceOptions {
  name: string;
  color: string;
}

export interface TimeSeriesOptions {
  traces: TraceOptions[];
  /** Visible time window ending at the newest sample, in ms. */
  windowMs: number;
  /** Samples kept per trace. Default 4096. */
  capacity?: number;
  xLabel?: string;
  yLabel?: string;
  /**
   * Fixed y range. When omitted the range fits the visible samples and then only grows,
   * so the axis stays still; `clear()` starts it over.
   */
  yRange?: [number, number];
}

/**
 * Lightweight canvas-2D plot of several traces over a rolling time window.
 * Call `push` as samples arrive and `draw` once per animation frame; drawing
 * is skipped when nothing changed.
 */
export class TimeSeriesPlot {
  readonly canvas: HTMLCanvasElement;
  readonly buffers: RingBuffer[];
  private readonly opts: TimeSeriesOptions;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly observer: ResizeObserver;
  private dirty = true;
  private fg = '';
  private readonly yHeld = new ExpandingRange(0.1);

  constructor(canvas: HTMLCanvasElement, opts: TimeSeriesOptions) {
    this.canvas = canvas;
    this.opts = opts;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('2D canvas context unavailable');
    this.ctx = ctx;
    this.buffers = opts.traces.map(() => new RingBuffer(opts.capacity ?? 4096));
    this.observer = new ResizeObserver(() => {
      this.dirty = true;
      this.fg = ''; // re-read the text colour (theme may have changed)
    });
    this.observer.observe(canvas);
  }

  push(trace: number, t: number, value: number): void {
    this.buffers[trace].push(t, value);
    this.dirty = true;
  }

  clear(): void {
    for (const b of this.buffers) b.clear();
    this.yHeld.reset();
    this.dirty = true;
  }

  draw(): void {
    if (!this.dirty) return;
    this.dirty = false;
    const { canvas, ctx, opts } = this;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cw = Math.max(1, canvas.clientWidth);
    const ch = Math.max(1, canvas.clientHeight);
    if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) {
      canvas.width = Math.round(cw * dpr);
      canvas.height = Math.round(ch * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);

    if (!this.fg) this.fg = getComputedStyle(canvas).color || '#222';
    const fg = this.fg;
    ctx.font = '11px system-ui, sans-serif';

    let tMax = -Infinity;
    for (const b of this.buffers) if (b.length && b.lastTime > tMax) tMax = b.lastTime;
    if (!Number.isFinite(tMax)) tMax = opts.windowMs;
    const tMin = tMax - opts.windowMs;

    let yMin = Infinity;
    let yMax = -Infinity;
    if (opts.yRange) {
      [yMin, yMax] = opts.yRange;
      if (!(yMax > yMin)) {
        const pad = Math.max(1e-6, Math.abs(yMax) * 0.05);
        yMin -= pad;
        yMax += pad;
      }
    } else {
      for (const b of this.buffers) {
        for (let i = b.firstIndexAtOrAfter(tMin); i < b.length; i++) {
          const v = b.valueAt(i);
          if (v < yMin) yMin = v;
          if (v > yMax) yMax = v;
        }
      }
      if (yMax >= yMin) {
        const pad = (yMax - yMin) * 0.08 || Math.max(1e-6, Math.abs(yMax) * 0.05) || 1;
        this.yHeld.include(yMin - pad, yMax + pad);
      }
      [yMin, yMax] = this.yHeld.empty ? [0, 1] : [this.yHeld.min, this.yHeld.max];
    }

    const left = 48;
    const right = 8;
    const top = opts.traces.length > 1 ? 18 : 8;
    const bottom = opts.xLabel ? 32 : 20;
    const pw = Math.max(1, cw - left - right);
    const ph = Math.max(1, ch - top - bottom);
    const sx = (t: number) => left + ((t - tMin) / opts.windowMs) * pw;
    const sy = (v: number) => top + (1 - (v - yMin) / (yMax - yMin)) * ph;

    // Axes and grid.
    ctx.strokeStyle = fg;
    ctx.globalAlpha = 0.15;
    ctx.lineWidth = 1;
    ctx.beginPath();
    const yt = niceTicks(yMin, yMax, 4);
    const xt = niceTicks(tMin, tMax, 5);
    for (const v of yt) {
      ctx.moveTo(left, Math.round(sy(v)) + 0.5);
      ctx.lineTo(left + pw, Math.round(sy(v)) + 0.5);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillStyle = fg;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (const v of yt) ctx.fillText(formatTick(v), left - 4, sy(v));
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (const t of xt) ctx.fillText(formatTick(t), sx(t), top + ph + 4);
    if (opts.xLabel) ctx.fillText(opts.xLabel, left + pw / 2, top + ph + 18);
    if (opts.yLabel) {
      ctx.save();
      ctx.translate(10, top + ph / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textBaseline = 'middle';
      ctx.fillText(opts.yLabel, 0, 0);
      ctx.restore();
    }

    // Traces, clipped to the plot area.
    ctx.save();
    ctx.beginPath();
    ctx.rect(left, top, pw, ph);
    ctx.clip();
    ctx.lineWidth = 1.5;
    this.buffers.forEach((b, k) => {
      if (!b.length) return;
      ctx.strokeStyle = opts.traces[k].color;
      ctx.beginPath();
      const start = Math.max(0, b.firstIndexAtOrAfter(tMin) - 1);
      // Min/max per pixel column: keeps spikes (e.g. an AP peak) when there
      // are more samples than pixels, and is stable as the window scrolls.
      let col = Math.floor(sx(b.timeAt(start)));
      let lo = Infinity;
      let hi = -Infinity;
      let first = true;
      const flush = () => {
        if (lo > hi) return;
        if (first) ctx.moveTo(col, sy(lo));
        else ctx.lineTo(col, sy(lo));
        if (hi !== lo) ctx.lineTo(col, sy(hi));
        first = false;
      };
      for (let i = start; i < b.length; i++) {
        const c = Math.floor(sx(b.timeAt(i)));
        const v = b.valueAt(i);
        if (c !== col) {
          flush();
          col = c;
          lo = Infinity;
          hi = -Infinity;
        }
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      flush();
      ctx.stroke();
    });
    ctx.restore();

    // Legend.
    if (opts.traces.length > 1) {
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      let x = left;
      for (const tr of opts.traces) {
        ctx.fillStyle = tr.color;
        ctx.fillRect(x, 5, 10, 3);
        ctx.fillStyle = fg;
        ctx.fillText(tr.name, x + 14, 1);
        x += 22 + ctx.measureText(tr.name).width;
      }
    }
  }

  dispose(): void {
    this.observer.disconnect();
  }
}
