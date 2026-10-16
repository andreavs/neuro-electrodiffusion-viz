import { colormapGradient, type Colormap } from './colormap';
import { ExpandingRange, autoRange, type Range } from './range';
import { formatTick, niceTicks } from './ticks';

export interface ColorbarOptions {
  colormap: Colormap;
  label: string;
  units: string;
  /**
   * Start in auto mode: the range follows the data passed to `update`, growing to fit
   * new extremes but never shrinking until `resetAuto()`, so the colours stay steady.
   * Default true.
   */
  auto?: boolean;
  /** Keep the auto range centred on zero. Defaults to `colormap.diverging`. */
  symmetric?: boolean;
  range?: Range;
  /** Called whenever the range changes (auto or user edit). */
  onChange?: (range: Range) => void;
}

/**
 * Vertical colour legend with tick labels, units and an auto / fixed range.
 * In fixed mode the min and max fields are editable. DOM is only touched when
 * the displayed range actually changes.
 */
export class Colorbar {
  readonly el: HTMLElement;
  readonly range: Range;
  private auto: boolean;
  private readonly symmetric: boolean;
  private readonly onChange?: (range: Range) => void;
  private readonly bar: HTMLElement;
  private readonly ticks: HTMLElement;
  private readonly autoBox: HTMLInputElement;
  private readonly minInput: HTMLInputElement;
  private readonly maxInput: HTMLInputElement;
  private readonly scratch: Range = { min: 0, max: 1 };
  private readonly held = new ExpandingRange();
  private shown = '';

  constructor(opts: ColorbarOptions) {
    this.auto = opts.auto ?? true;
    this.symmetric = opts.symmetric ?? opts.colormap.diverging;
    this.onChange = opts.onChange;
    this.range = { ...(opts.range ?? { min: 0, max: 1 }) };

    this.el = document.createElement('div');
    this.el.className = 'colorbar';
    this.el.style.cssText =
      'display:inline-flex;flex-direction:column;gap:4px;font:12px system-ui,sans-serif;color:var(--text,#222);' +
      'background:var(--surface,#fff);border:1px solid var(--border,#ddd);border-radius:8px;padding:8px;min-width:92px';
    const title = document.createElement('div');
    title.style.fontWeight = '600';
    title.textContent = opts.units ? `${opts.label} (${opts.units})` : opts.label;

    const body = document.createElement('div');
    body.style.cssText = 'display:flex;gap:6px;height:160px';
    this.bar = document.createElement('div');
    this.bar.style.cssText = 'width:14px;border-radius:3px;border:1px solid var(--border,#ccc)';
    this.ticks = document.createElement('div');
    this.ticks.style.cssText = 'position:relative;flex:1;min-width:44px;font-variant-numeric:tabular-nums';
    body.append(this.bar, this.ticks);

    const field = (name: string) => {
      const input = document.createElement('input');
      input.type = 'number';
      input.step = 'any';
      input.ariaLabel = `${opts.label} ${name}`;
      input.style.cssText = 'width:84px;font:inherit';
      input.addEventListener('change', () => this.onUserEdit());
      return input;
    };
    this.maxInput = field('max');
    this.minInput = field('min');
    const autoLabel = document.createElement('label');
    autoLabel.style.cssText = 'display:flex;align-items:center;gap:4px';
    this.autoBox = document.createElement('input');
    this.autoBox.type = 'checkbox';
    this.autoBox.checked = this.auto;
    this.autoBox.addEventListener('change', () => this.setAuto(this.autoBox.checked));
    autoLabel.append(this.autoBox, 'auto range');

    this.el.append(title, this.maxInput, body, this.minInput, autoLabel);
    this.setColormap(opts.colormap);
    this.syncInputs();
    this.redraw();
  }

  get isAuto(): boolean {
    return this.auto;
  }

  setColormap(cm: Colormap): void {
    this.bar.style.background = `linear-gradient(to top, ${colormapGradient(cm)})`;
  }

  setAuto(auto: boolean): void {
    if (auto && !this.auto) this.held.reset();
    this.auto = auto;
    this.autoBox.checked = auto;
    this.syncInputs();
  }

  /** Fix the range (switches off auto mode). */
  setRange(min: number, max: number): void {
    this.auto = false;
    this.autoBox.checked = false;
    this.apply(min, max);
    this.syncInputs();
  }

  /** Forget the auto range; the next `update` fits its data afresh (e.g. after a simulation reset). */
  resetAuto(): void {
    this.held.reset();
  }

  /** In auto mode, grow the range to cover `data`. Returns the current range. */
  update(data: ArrayLike<number>): Range {
    if (this.auto && data.length > 0) {
      const r = autoRange(data, this.symmetric, this.scratch);
      if (this.held.include(r.min, r.max)) {
        if (this.symmetric) {
          const m = Math.max(-this.held.min, this.held.max);
          this.held.min = -m;
          this.held.max = m;
        }
        this.apply(this.held.min, this.held.max);
        this.syncInputs();
      }
    }
    return this.range;
  }

  dispose(): void {
    this.el.remove();
  }

  private onUserEdit(): void {
    const min = Number(this.minInput.value);
    const max = Number(this.maxInput.value);
    if (Number.isFinite(min) && Number.isFinite(max) && max > min) this.setRange(min, max);
    else this.syncInputs();
  }

  private apply(min: number, max: number): void {
    if (min === this.range.min && max === this.range.max) return;
    this.range.min = min;
    this.range.max = max;
    this.redraw();
    this.onChange?.(this.range);
  }

  private syncInputs(): void {
    this.minInput.disabled = this.auto;
    this.maxInput.disabled = this.auto;
    const lo = String(Number(this.range.min.toPrecision(4)));
    const hi = String(Number(this.range.max.toPrecision(4)));
    if (this.minInput.value !== lo && document.activeElement !== this.minInput) this.minInput.value = lo;
    if (this.maxInput.value !== hi && document.activeElement !== this.maxInput) this.maxInput.value = hi;
  }

  private redraw(): void {
    const { min, max } = this.range;
    const ticks = niceTicks(min, max, 5);
    // Tick positions depend on min/max too; quantise so tiny drifts don't redraw.
    const q = (max - min) * 1e-3 || 1;
    const key = `${Math.round(min / q)}|${Math.round(max / q)}|${ticks.map(formatTick).join('|')}`;
    if (key === this.shown) return;
    this.shown = key;
    this.ticks.replaceChildren(
      ...ticks.map((t) => {
        const s = document.createElement('span');
        const y = max > min ? (t - min) / (max - min) : 0.5;
        s.style.cssText = `position:absolute;left:0;bottom:${(y * 100).toFixed(2)}%;transform:translateY(50%);white-space:nowrap`;
        s.textContent = `– ${formatTick(t)}`;
        return s;
      }),
    );
  }
}
