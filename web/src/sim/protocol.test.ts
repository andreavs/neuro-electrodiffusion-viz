import { describe, expect, it } from 'vitest';
import { transferablesOf } from './protocol';

describe('transferablesOf', () => {
  it('collects every field buffer of a frame', () => {
    const a = new Float32Array(4);
    const b = new Float32Array(8);
    const out = transferablesOf({
      type: 'frame',
      timeMs: 0,
      fields: [
        { name: 'a', shape: [4], data: a },
        { name: 'b', shape: [2, 4], data: b },
      ],
    });
    expect(out).toEqual([a.buffer, b.buffer]);
  });

  it('transfers nothing for control messages', () => {
    expect(transferablesOf({ type: 'ready', version: '0.1.0' })).toEqual([]);
  });
});
