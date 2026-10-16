// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { SimClient } from '../sim/client';
import { DEFAULT_BUDGET_MS } from '../sim/loop';
import { FakeWorker } from '../sim/testing/fakeWorker';
import { ControlPanel, type ParamSchema } from './controls';

const schema = [
  { kind: 'number', key: 'gsyn', label: 'g', min: 0, max: 10, step: 0.5, default: 4, unit: 'mS' },
  { kind: 'select', key: 'geom', label: 'Spine', options: [{ value: 'a', label: 'A' }, { value: 'b', label: 'B' }], default: 'a' },
  { kind: 'toggle', key: 'field', label: 'Field', default: true },
  { kind: 'button', key: 'stimulate', label: 'Stimulate', showWhen: { key: 'geom', values: ['a'] } },
  { kind: 'number', key: 'neck', label: 'Neck', min: 0, max: 1, step: 0.1, default: 0.5, showWhen: { key: 'geom', values: ['b'] } },
] as const satisfies ParamSchema;

describe('ControlPanel (DOM)', () => {
  let worker: FakeWorker;
  let client: SimClient;
  let root: HTMLElement;
  let panel: ControlPanel;

  const el = <T extends HTMLElement>(sel: string) => root.querySelector<T>(sel)!;
  const fire = (target: HTMLElement, type: string) => target.dispatchEvent(new Event(type, { bubbles: true }));
  const sentAfterSetup = () => worker.sent.slice(1); // first message is the initial setBudget

  beforeEach(() => {
    worker = new FakeWorker();
    client = new SimClient(() => worker);
    root = document.createElement('div');
    document.body.append(root);
    panel = new ControlPanel(root, client, schema);
  });

  afterEach(() => {
    client.terminate();
    root.remove();
  });

  it('sends the default speed budget on mount and exposes params without speed', () => {
    expect(worker.sent[0]).toEqual({ type: 'setBudget', budgetMs: DEFAULT_BUDGET_MS });
    expect(panel.params()).toEqual({ gsyn: 4, geom: 'a', field: true, neck: 0.5 });
  });

  it('sends one setParams per change even though input and change both fire', () => {
    const slider = el<HTMLInputElement>('[data-key="gsyn"]');
    slider.value = '7.5';
    fire(slider, 'input');
    fire(slider, 'change');
    const box = el<HTMLInputElement>('[data-key="field"]');
    box.checked = false;
    fire(box, 'input');
    fire(box, 'change');
    const select = el<HTMLSelectElement>('[data-key="geom"]');
    select.value = 'b';
    fire(select, 'change');
    expect(sentAfterSetup()).toEqual([
      { type: 'setParams', params: { gsyn: 7.5 } },
      { type: 'setParams', params: { field: false } },
      { type: 'setParams', params: { geom: 'b' } },
    ]);
    expect(el('[data-out="gsyn"]').textContent).toBe('7.5 mS');
    expect(panel.params()).toEqual({ gsyn: 7.5, geom: 'b', field: false, neck: 0.5 });
  });

  it('setValues updates the controls and sends only the changed keys in one message', () => {
    panel.setValues({ gsyn: 2, geom: 'b', field: true, stimulate: 1, unknown: 3 });
    expect(sentAfterSetup()).toEqual([{ type: 'setParams', params: { gsyn: 2, geom: 'b' } }]);
    expect(el<HTMLInputElement>('[data-key="gsyn"]').value).toBe('2');
    expect(el<HTMLSelectElement>('[data-key="geom"]').value).toBe('b');
    expect(el('[data-out="gsyn"]').textContent).toBe('2.0 mS');
    panel.setValues({ gsyn: 2 });
    expect(sentAfterSetup()).toHaveLength(1);
    panel.setValues({ gsyn: 99, geom: 'nope' }); // clamped, and an unknown option falls back
    expect(sentAfterSetup()[1]).toEqual({ type: 'setParams', params: { gsyn: 10, geom: 'a' } });
  });

  it('routes speed to setBudget, not setParams', () => {
    const speed = el<HTMLInputElement>('[data-key="speed"]');
    speed.value = '0.5';
    fire(speed, 'input');
    expect(sentAfterSetup()).toEqual([{ type: 'setBudget', budgetMs: DEFAULT_BUDGET_MS / 2 }]);
    expect(el('[data-out="speed"]').textContent).toBe('0.50 ×');
  });

  it('routes play/pause, reset and action buttons', () => {
    el('[data-transport="play"]').click();
    expect(el('[data-transport="play"]').textContent).toBe('Pause');
    el('[data-transport="play"]').click();
    el('[data-transport="reset"]').click();
    el('[data-action="stimulate"]').click();
    expect(sentAfterSetup()).toEqual([
      { type: 'run', running: true },
      { type: 'run', running: false },
      { type: 'reset' },
      { type: 'action', name: 'stimulate' },
    ]);
  });

  it('shows Play again when the worker reports an error', () => {
    el('[data-transport="play"]').click();
    worker.emit({ type: 'error', message: 'boom' });
    expect(panel.isRunning).toBe(false);
    expect(el('[data-transport="play"]').textContent).toBe('Play');
  });

  it('updates the stats readout', () => {
    panel.updateStats({ simTimeMs: 12.5, stepsPerSec: 2500, simMsPerSec: 40, fps: 59.6 });
    expect(el('[data-stat="time"]').textContent).toBe('12.5 ms');
    expect(el('[data-stat="steps"]').textContent).toBe('2.5k');
    expect(el('[data-stat="fps"]').textContent).toBe('60 fps');
  });

  it('is removed with its listeners when the client terminates', () => {
    const slider = el<HTMLInputElement>('[data-key="gsyn"]');
    client.terminate();
    expect(root.querySelector('.ctl-panel')).toBeNull();
    const before = worker.sent.length;
    slider.value = '1';
    root.append(slider);
    fire(slider, 'input');
    worker.emit({ type: 'error', message: 'late' });
    expect(worker.sent.length).toBe(before);
  });

  it('shows controls only while their showWhen holds', () => {
    const shown = (sel: string) => !el(sel).closest<HTMLElement>('.ctl')!.hidden;
    expect(shown('[data-action="stimulate"]')).toBe(true);
    expect(shown('[data-key="neck"]')).toBe(false);
    const select = el<HTMLSelectElement>('[data-key="geom"]');
    select.value = 'b';
    fire(select, 'change');
    expect(shown('[data-action="stimulate"]')).toBe(false);
    expect(shown('[data-key="neck"]')).toBe(true);
    panel.setValues({ geom: 'a' });
    expect(shown('[data-key="neck"]')).toBe(false);
  });
});
