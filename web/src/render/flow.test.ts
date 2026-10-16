import { describe, expect, it } from 'vitest';
import { viridis } from './colormap';
import { FlowTracers } from './flow';

describe('FlowTracers', () => {
  it('streams down the gradient', () => {
    const shape = [16, 8, 8] as const;
    const field = new Float32Array(16 * 8 * 8);
    for (let i = 0; i < field.length; i++) field[i] = i % 16; // grows along x
    const flow = new FlowTracers({ shape, extent: { x: 2, y: 1, z: 1 }, colormap: viridis, count: 200, trail: 4 });
    flow.setField(field);
    for (let s = 0; s < 10; s++) flow.step(1 / 60);
    const pos = flow.lines.geometry.getAttribute('position').array as Float32Array;
    const alpha = flow.lines.geometry.getAttribute('alpha').array as Float32Array;
    let down = 0;
    let up = 0;
    // The first segment of each trail runs from the newest point to the one before.
    for (let i = 0; i < 200; i++) {
      const v = i * 3 * 2;
      if (!(alpha[v]! > 0)) continue;
      const dx = pos[3 * v]! - pos[3 * (v + 1)]!;
      if (dx < 0) down++;
      else if (dx > 0) up++;
    }
    expect(down).toBeGreaterThan(50);
    expect(up).toBe(0);
    flow.dispose();
  });
});
