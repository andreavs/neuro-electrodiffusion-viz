import { describe, expect, it } from 'vitest';
import {
  SPEED_SPEC,
  clampToStep,
  defaultParams,
  escapeHtml,
  formatNumber,
  formatRate,
  formatSimTime,
  readControl,
  renderControl,
  renderPanel,
  speedToBudgetMs,
  type ParamSchema,
} from './controls';
import { DEFAULT_BUDGET_MS } from '../sim/loop';

const schema = [
  { kind: 'number', key: 'gsyn', label: 'Synaptic g', min: 0, max: 10, step: 0.5, default: 4, unit: 'mS/cm²' },
  {
    kind: 'select',
    key: 'geom',
    label: 'Spine',
    options: [
      { value: 'sharp', label: 'Sharp neck' },
      { value: 'funnel', label: 'Funneled' },
    ],
    default: 'sharp',
  },
  { kind: 'toggle', key: 'field', label: 'Electric field', default: true },
  { kind: 'button', key: 'stimulate', label: 'Stimulate' },
] as const satisfies ParamSchema;

describe('defaultParams', () => {
  it('collects defaults and skips buttons', () => {
    expect(defaultParams(schema)).toEqual({ gsyn: 4, geom: 'sharp', field: true });
  });
});

describe('renderControl', () => {
  it('renders a number spec as a range slider with min/max/step and a formatted unit', () => {
    const html = renderControl(schema[0], 2.5);
    expect(html).toContain('type="range"');
    expect(html).toContain('data-key="gsyn"');
    expect(html).toContain('min="0" max="10" step="0.5" value="2.5"');
    expect(html).toContain('2.5 mS/cm²');
  });

  it('falls back to the default when the value has the wrong type', () => {
    expect(renderControl(schema[0], 'x')).toContain('value="4"');
  });

  it('renders a select with the current option selected', () => {
    const html = renderControl(schema[1], 'funnel');
    expect(html).toContain('<select data-key="geom">');
    expect(html).toContain('<option value="funnel" selected>Funneled</option>');
    expect(html).not.toContain('value="sharp" selected');
  });

  it('renders a toggle as a checkbox', () => {
    expect(renderControl(schema[2], true)).toMatch(/type="checkbox" data-key="field" checked/);
    expect(renderControl(schema[2], false)).not.toContain('checked');
  });

  it('renders a button as an action', () => {
    expect(renderControl(schema[3], undefined)).toContain('data-action="stimulate"');
  });

  it('escapes labels', () => {
    const html = renderControl({ kind: 'toggle', key: 'k', label: '<b>"x"</b>', default: false }, false);
    expect(html).toContain('&lt;b&gt;&quot;x&quot;&lt;/b&gt;');
    expect(escapeHtml(`&'`)).toBe('&amp;&#39;');
  });
});

describe('renderPanel', () => {
  it('includes transport, speed, every control and the stats readout', () => {
    const html = renderPanel(schema, defaultParams(schema), false);
    expect(html).toContain('data-transport="play">Play<');
    expect(html).toContain('data-transport="reset"');
    expect(html).toContain('data-key="speed"');
    for (const s of schema) expect(html).toContain(`"${s.key}"`);
    for (const stat of ['time', 'steps', 'rate', 'fps']) expect(html).toContain(`data-stat="${stat}"`);
    expect(renderPanel(schema, {}, true)).toContain('>Pause<');
  });
});

describe('readControl', () => {
  it('parses and snaps numbers', () => {
    expect(readControl(schema[0], { value: '3.7' })).toBe(3.5);
    expect(readControl(schema[0], { value: '99' })).toBe(10);
    expect(readControl(schema[0], { value: 'nope' })).toBe(0);
  });

  it('rejects unknown select values', () => {
    expect(readControl(schema[1], { value: 'funnel' })).toBe('funnel');
    expect(readControl(schema[1], { value: 'evil' })).toBe('sharp');
  });

  it('reads toggles from checked', () => {
    expect(readControl(schema[2], { value: 'on', checked: false })).toBe(false);
    expect(readControl(schema[2], { value: 'on', checked: true })).toBe(true);
  });

  it('gives no value for buttons', () => {
    expect(readControl(schema[3], { value: '' })).toBeUndefined();
  });
});

describe('formatting helpers', () => {
  it('formats numbers to the step precision', () => {
    expect(formatNumber(1, 1)).toBe('1');
    expect(formatNumber(0.3, 0.1, 'ms')).toBe('0.3 ms');
    expect(formatNumber(0.25, 0.05)).toBe('0.25');
    expect(formatNumber(1e-4, 1e-4)).toBe('0.0001');
  });

  it('clamps to step without float noise', () => {
    expect(clampToStep(0.30000000000000004, 0, 1, 0.1)).toBe(0.3);
    expect(clampToStep(0.12, 0.05, 1, 0.05)).toBe(0.1);
  });

  it('formats sim time and rates', () => {
    expect(formatSimTime(0.0005)).toBe('0.5 µs');
    expect(formatSimTime(2.5)).toBe('2.500 ms');
    expect(formatSimTime(123.45)).toBe('123.5 ms');
    expect(formatSimTime(12_000)).toBe('12.00 s');
    expect(formatRate(5)).toBe('5.0');
    expect(formatRate(250)).toBe('250');
    expect(formatRate(2500)).toBe('2.5k');
    expect(formatRate(25_000)).toBe('25k');
    expect(formatRate(2_500_000)).toBe('2.5M');
  });

  it('maps speed to a budget up to the default', () => {
    expect(speedToBudgetMs(SPEED_SPEC.max)).toBe(DEFAULT_BUDGET_MS);
    expect(speedToBudgetMs(0.5)).toBe(DEFAULT_BUDGET_MS / 2);
  });
});
