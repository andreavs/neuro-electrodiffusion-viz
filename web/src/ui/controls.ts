import type { SimClient } from '../sim/client';
import { DEFAULT_BUDGET_MS } from '../sim/loop';
import type { ParamValue, Params } from '../sim/protocol';
import './controls.css';

/** One control in a scale's parameter schema. */
export type ParamSpec =
  | {
      kind: 'number';
      key: string;
      label: string;
      min: number;
      max: number;
      step: number;
      default: number;
      unit?: string;
      help?: string;
    }
  | {
      kind: 'select';
      key: string;
      label: string;
      options: readonly { value: string; label: string }[];
      default: string;
      help?: string;
    }
  | { kind: 'toggle'; key: string; label: string; default: boolean; help?: string }
  /** A one-shot action (sent as an `action` message, not a parameter). */
  | { kind: 'button'; key: string; label: string; help?: string };

export type ParamSchema = readonly ParamSpec[];

/** Initial parameter values for a schema (buttons carry no value). */
export function defaultParams(schema: ParamSchema): Params {
  const out: Params = {};
  for (const s of schema) if (s.kind !== 'button') out[s.key] = s.default;
  return out;
}

const escapeMap: Record<string, string> = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;',
};

export function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => escapeMap[c]!);
}

/** Display a number with precision matching the slider step. */
export function formatNumber(value: number, step: number, unit?: string): string {
  const decimals = step >= 1 ? 0 : Math.min(6, Math.ceil(-Math.log10(step) - 1e-9));
  const text = value.toFixed(decimals);
  return unit ? `${text} ${unit}` : text;
}

/** Snap a slider value into range and onto the step grid (relative to min). */
export function clampToStep(value: number, min: number, max: number, step: number): number {
  if (!Number.isFinite(value)) return min;
  const snapped = min + Math.round((value - min) / step) * step;
  const clamped = Math.min(max, Math.max(min, snapped));
  // Strip float noise such as 0.30000000000000004.
  return Number(clamped.toPrecision(12));
}

/** Convert what an input element holds into the parameter's value. */
export function readControl(
  spec: ParamSpec,
  input: { value: string; checked?: boolean },
): ParamValue | undefined {
  switch (spec.kind) {
    case 'number':
      return clampToStep(Number(input.value), spec.min, spec.max, spec.step);
    case 'select':
      return spec.options.some((o) => o.value === input.value) ? input.value : spec.default;
    case 'toggle':
      return Boolean(input.checked);
    case 'button':
      return undefined;
  }
}

/** Markup for one control. Pure, so the schema → DOM mapping is unit-testable. */
export function renderControl(spec: ParamSpec, value: ParamValue | undefined): string {
  const key = escapeHtml(spec.key);
  const label = escapeHtml(spec.label);
  const title = spec.help ? ` title="${escapeHtml(spec.help)}"` : '';
  switch (spec.kind) {
    case 'number': {
      const v = typeof value === 'number' ? value : spec.default;
      return `
        <label class="ctl ctl-number"${title}>
          <span class="ctl-label">${label}
            <output data-out="${key}">${escapeHtml(formatNumber(v, spec.step, spec.unit))}</output>
          </span>
          <input type="range" data-key="${key}" min="${spec.min}" max="${spec.max}" step="${spec.step}" value="${v}" />
        </label>`;
    }
    case 'select': {
      const v = typeof value === 'string' ? value : spec.default;
      const options = spec.options
        .map(
          (o) =>
            `<option value="${escapeHtml(o.value)}"${o.value === v ? ' selected' : ''}>${escapeHtml(o.label)}</option>`,
        )
        .join('');
      return `
        <label class="ctl ctl-select"${title}>
          <span class="ctl-label">${label}</span>
          <select data-key="${key}">${options}</select>
        </label>`;
    }
    case 'toggle': {
      const v = typeof value === 'boolean' ? value : spec.default;
      return `
        <label class="ctl ctl-toggle"${title}>
          <input type="checkbox" data-key="${key}"${v ? ' checked' : ''} />
          <span class="ctl-label">${label}</span>
        </label>`;
    }
    case 'button':
      return `<button type="button" class="ctl ctl-button" data-action="${key}"${title}>${label}</button>`;
  }
}

/** Speed is a fraction of the default per-tick solver budget. */
export const SPEED_SPEC = {
  kind: 'number',
  key: 'speed',
  label: 'Speed',
  min: 0.05,
  max: 1,
  step: 0.05,
  default: 1,
  unit: '×',
  help: 'Fraction of each frame the solver may spend computing',
} as const satisfies ParamSpec;

export function speedToBudgetMs(speed: number): number {
  return DEFAULT_BUDGET_MS * speed;
}

export interface StatsView {
  simTimeMs: number;
  stepsPerSec: number;
  simMsPerSec: number;
  fps: number;
}

/** Format a simulated time span compactly (µs below 1 ms, s above 10 s). */
export function formatSimTime(ms: number): string {
  if (ms < 1) return `${(ms * 1000).toFixed(1)} µs`;
  if (ms < 10_000) return `${ms.toFixed(ms < 10 ? 3 : 1)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

export function formatRate(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e4) return `${(n / 1e3).toFixed(0)}k`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return n.toFixed(n < 10 ? 1 : 0);
}

export function renderPanel(schema: ParamSchema, values: Params, running: boolean): string {
  return `
    <div class="ctl-transport">
      <button type="button" class="ctl-button ctl-primary" data-transport="play">${running ? 'Pause' : 'Play'}</button>
      <button type="button" class="ctl-button" data-transport="reset">Reset</button>
    </div>
    <div class="ctl-group">
      ${renderControl(SPEED_SPEC, values[SPEED_SPEC.key])}
      ${schema.map((s) => renderControl(s, values[s.key])).join('')}
    </div>
    <dl class="ctl-stats">
      <dt>Sim time</dt><dd data-stat="time">–</dd>
      <dt>Steps/s</dt><dd data-stat="steps">–</dd>
      <dt>Sim ms/s</dt><dd data-stat="rate">–</dd>
      <dt>Render</dt><dd data-stat="fps">–</dd>
    </dl>`;
}

export interface ControlPanelOptions {
  /** Start values; defaults come from the schema. */
  values?: Params;
  running?: boolean;
}

/**
 * Play/pause, reset, speed, the scale's parameters and a stats readout.
 * Parameter changes go straight to the worker as `setParams`.
 */
export class ControlPanel {
  readonly el: HTMLElement;
  private readonly values: Params;
  private running: boolean;
  private readonly specs: Map<string, ParamSpec>;
  private readonly stat: Record<'time' | 'steps' | 'rate' | 'fps', HTMLElement>;
  private readonly playButton: HTMLButtonElement;
  private readonly unregister: () => void;
  private destroyed = false;

  constructor(
    parent: HTMLElement,
    private readonly client: SimClient,
    schema: ParamSchema,
    opts: ControlPanelOptions = {},
  ) {
    this.values = { ...defaultParams(schema), [SPEED_SPEC.key]: SPEED_SPEC.default, ...opts.values };
    this.running = opts.running ?? false;
    this.specs = new Map([SPEED_SPEC, ...schema].map((s) => [s.key, s]));

    this.el = document.createElement('div');
    this.el.className = 'ctl-panel';
    this.el.innerHTML = renderPanel(schema, this.values, this.running);
    parent.append(this.el);

    const q = (sel: string) => this.el.querySelector<HTMLElement>(sel)!;
    this.stat = {
      time: q('[data-stat="time"]'),
      steps: q('[data-stat="steps"]'),
      rate: q('[data-stat="rate"]'),
      fps: q('[data-stat="fps"]'),
    };
    this.playButton = q('[data-transport="play"]') as HTMLButtonElement;

    this.el.addEventListener('input', this.onInput);
    this.el.addEventListener('change', this.onInput);
    this.el.addEventListener('click', this.onClick);
    this.unregister = client.addDisposer(() => this.destroy());
    client.setBudget(speedToBudgetMs(this.values[SPEED_SPEC.key] as number));
  }

  /** Current parameter values (excluding speed), e.g. for `client.create`. */
  params(): Params {
    const { [SPEED_SPEC.key]: _speed, ...rest } = this.values;
    return rest;
  }

  get isRunning(): boolean {
    return this.running;
  }

  setRunning(running: boolean): void {
    this.running = running;
    this.playButton.textContent = running ? 'Pause' : 'Play';
    this.client.run(running);
  }

  updateStats(s: StatsView): void {
    this.stat.time.textContent = formatSimTime(s.simTimeMs);
    this.stat.steps.textContent = formatRate(s.stepsPerSec);
    this.stat.rate.textContent = formatRate(s.simMsPerSec);
    this.stat.fps.textContent = `${s.fps.toFixed(0)} fps`;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.el.removeEventListener('input', this.onInput);
    this.el.removeEventListener('change', this.onInput);
    this.el.removeEventListener('click', this.onClick);
    this.unregister();
    this.el.remove();
  }

  private readonly onInput = (ev: Event): void => {
    const input = ev.target as HTMLInputElement | HTMLSelectElement;
    const key = input.dataset.key;
    const spec = key ? this.specs.get(key) : undefined;
    if (!spec) return;
    const value = readControl(spec, input as HTMLInputElement);
    if (value === undefined || value === this.values[spec.key]) return;
    this.values[spec.key] = value;
    if (spec.kind === 'number') {
      const out = this.el.querySelector(`[data-out="${CSS.escape(spec.key)}"]`);
      if (out) out.textContent = formatNumber(value as number, spec.step, spec.unit);
    }
    if (spec.key === SPEED_SPEC.key) this.client.setBudget(speedToBudgetMs(value as number));
    else this.client.setParams({ [spec.key]: value });
  };

  private readonly onClick = (ev: Event): void => {
    const button = (ev.target as HTMLElement).closest<HTMLButtonElement>('button');
    if (!button || !this.el.contains(button)) return;
    if (button.dataset.transport === 'play') this.setRunning(!this.running);
    else if (button.dataset.transport === 'reset') this.client.reset();
    else if (button.dataset.action) this.client.action(button.dataset.action);
  };
}
