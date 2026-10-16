import { formatTick, niceTicks } from '../../render';

export interface PlotSeries {
  /** Interleaved (x, y) points. */
  xy: ArrayLike<number>;
  color: string;
  dashed?: boolean;
  /** Fill the area under the line (for histograms). */
  fill?: boolean;
}

export interface PlotFrame {
  x: [number, number];
  y: [number, number];
  series: PlotSeries[];
}

/** Small canvas-2D line plot with fixed axes, redrawn whenever `draw` is called. */
export class LinePlot {
  private readonly ctx: CanvasRenderingContext2D;
  private last: PlotFrame | null = null;
  private readonly observer: ResizeObserver;
  /** CSS size and text colour, refreshed on resize so `draw` forces no style or layout pass. */
  private size = { w: 1, h: 1 };
  private fg = '';

  constructor(
    readonly canvas: HTMLCanvasElement,
    private readonly labels: { x: string; y: string },
  ) {
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('2D canvas context unavailable');
    this.ctx = ctx;
    this.observer = new ResizeObserver(() => {
      this.measure();
      if (this.last) this.draw(this.last);
    });
    this.observer.observe(canvas);
  }

  private measure(): void {
    const { canvas } = this;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.size = { w: Math.max(1, canvas.clientWidth), h: Math.max(1, canvas.clientHeight) };
    canvas.width = Math.round(this.size.w * dpr);
    canvas.height = Math.round(this.size.h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.fg = getComputedStyle(canvas).color || '#222';
  }

  draw(frame: PlotFrame): void {
    this.last = frame;
    const { ctx, fg } = this;
    if (!fg) return; // not measured yet; the ResizeObserver redraws
    const { w: cw, h: ch } = this.size;
    ctx.clearRect(0, 0, cw, ch);
    ctx.font = '11px system-ui, sans-serif';

    const [left, right, top, bottom] = [44, 8, 8, 32];
    const pw = Math.max(1, cw - left - right);
    const ph = Math.max(1, ch - top - bottom);
    const [x0, x1] = frame.x;
    const [y0, y1] = frame.y;
    const sx = (x: number) => left + ((x - x0) / (x1 - x0 || 1)) * pw;
    const sy = (y: number) => top + (1 - (y - y0) / (y1 - y0 || 1)) * ph;

    ctx.strokeStyle = fg;
    ctx.globalAlpha = 0.15;
    ctx.beginPath();
    const yt = niceTicks(y0, y1, 4);
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
    for (const v of niceTicks(x0, x1, 5)) ctx.fillText(formatTick(v), sx(v), top + ph + 4);
    ctx.fillText(this.labels.x, left + pw / 2, top + ph + 18);
    ctx.save();
    ctx.translate(10, top + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textBaseline = 'middle';
    ctx.fillText(this.labels.y, 0, 0);
    ctx.restore();

    ctx.save();
    ctx.beginPath();
    ctx.rect(left, top, pw, ph);
    ctx.clip();
    ctx.lineWidth = 1.5;
    for (const s of frame.series) {
      const n = Math.floor(s.xy.length / 2);
      if (n < 2) continue;
      ctx.beginPath();
      ctx.moveTo(sx(s.xy[0]!), sy(s.xy[1]!));
      for (let i = 1; i < n; i++) ctx.lineTo(sx(s.xy[2 * i]!), sy(s.xy[2 * i + 1]!));
      if (s.fill) {
        ctx.fillStyle = s.color;
        ctx.globalAlpha = 0.35;
        ctx.fill();
        ctx.globalAlpha = 1;
      }
      ctx.setLineDash(s.dashed ? [5, 4] : []);
      ctx.strokeStyle = s.color;
      ctx.stroke();
    }
    ctx.restore();
  }

  dispose(): void {
    this.observer.disconnect();
  }
}
