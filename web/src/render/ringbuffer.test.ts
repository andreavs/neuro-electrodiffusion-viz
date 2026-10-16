import { describe, expect, it } from 'vitest';
import { RingBuffer } from './ringbuffer';

describe('RingBuffer', () => {
  it('returns samples oldest first before wrapping', () => {
    const b = new RingBuffer(4);
    b.push(0, 10);
    b.push(1, 11);
    expect(b.length).toBe(2);
    expect([b.timeAt(0), b.valueAt(0), b.timeAt(1), b.valueAt(1)]).toEqual([0, 10, 1, 11]);
    expect(b.lastTime).toBe(1);
  });

  it('overwrites the oldest samples after wrapping', () => {
    const b = new RingBuffer(3);
    for (let i = 0; i < 7; i++) b.push(i, i * 2);
    expect(b.length).toBe(3);
    expect([0, 1, 2].map((i) => b.timeAt(i))).toEqual([4, 5, 6]);
    expect([0, 1, 2].map((i) => b.valueAt(i))).toEqual([8, 10, 12]);
  });

  it('finds the first sample in a window', () => {
    const b = new RingBuffer(5);
    for (let i = 0; i < 8; i++) b.push(i * 0.5, i); // keeps t = 1.5 .. 3.5
    expect(b.firstIndexAtOrAfter(0)).toBe(0);
    expect(b.firstIndexAtOrAfter(2.2)).toBe(2);
    expect(b.timeAt(b.firstIndexAtOrAfter(2.5))).toBe(2.5);
    expect(b.firstIndexAtOrAfter(10)).toBe(5);
  });

  it('bounds-checks and clears', () => {
    const b = new RingBuffer(2);
    expect(() => b.timeAt(0)).toThrow(RangeError);
    expect(Number.isNaN(b.lastTime)).toBe(true);
    b.push(1, 1);
    b.clear();
    expect(b.length).toBe(0);
    expect(() => new RingBuffer(0)).toThrow(RangeError);
  });
});
