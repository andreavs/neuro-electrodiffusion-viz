import type { SimClient } from '../sim/client';
import type { FrameMessage } from '../sim/protocol';
import type { StatsView } from './controls';

/** Frames per second over a sliding window of animation-frame timestamps. */
export class FpsMeter {
  private readonly times: number[] = [];

  constructor(private readonly windowMs = 1000) {}

  tick(now: number): number {
    this.times.push(now);
    return this.fpsAt(now);
  }

  /** Rate as of `now`; decays to 0 when ticks stop arriving. */
  fpsAt(now: number): number {
    while (this.times.length > 0 && now - this.times[0]! > this.windowMs) this.times.shift();
    const n = this.times.length;
    if (n < 2) return 0;
    const span = Math.max(now - this.times[0]!, this.times[n - 1]! - this.times[0]!);
    return span > 0 ? ((n - 1) * 1000) / span : 0;
  }
}

export interface RenderLoopOptions {
  /** Draw a new frame. Called at most once per animation frame, only when one arrived. */
  draw(frame: FrameMessage): void;
  /** Receives stats a few times per second (e.g. `ControlPanel`). */
  stats?: { updateStats(s: StatsView): void };
  statsIntervalMs?: number;
  raf?: (cb: FrameRequestCallback) => number;
  cancelRaf?: (id: number) => void;
}

/**
 * Drive rendering from requestAnimationFrame: take (and thereby ack) the
 * newest frame, draw it, and report stats. Returns a stop function; the loop
 * also stops when the client is terminated.
 */
export function startRenderLoop(client: SimClient, opts: RenderLoopOptions): () => void {
  const raf = opts.raf ?? requestAnimationFrame;
  const cancelRaf = opts.cancelRaf ?? cancelAnimationFrame;
  const interval = opts.statsIntervalMs ?? 250;
  const meter = new FpsMeter();
  let last: FrameMessage | null = null;
  let lastStats = -Infinity;
  let id = 0;
  let stopped = false;

  const loop = (now: number) => {
    if (stopped) return;
    id = raf(loop);
    const frame = client.takeFrame();
    if (frame) {
      last = frame;
      opts.draw(frame);
      // Count drawn frames, so a slow simulation shows up as a low rate.
      meter.tick(now);
    }
    if (opts.stats && last && now - lastStats >= interval) {
      lastStats = now;
      opts.stats.updateStats({
        simTimeMs: last.timeMs,
        stepsPerSec: last.runtime.stepsPerSec,
        simMsPerSec: last.runtime.simMsPerSec,
        fps: meter.fpsAt(now),
      });
    }
  };
  id = raf(loop);

  const stop = () => {
    if (stopped) return;
    stopped = true;
    cancelRaf(id);
    unregister();
  };
  const unregister = client.addDisposer(stop);
  return stop;
}
