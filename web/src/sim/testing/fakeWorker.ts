import type { WorkerLike } from '../client';
import type { FromWorker, ToWorker } from '../protocol';

/** Records what the client sends and lets tests emit worker messages. */
export class FakeWorker implements WorkerLike {
  sent: ToWorker[] = [];
  terminated = false;
  onmessage: ((ev: MessageEvent<FromWorker>) => void) | null = null;
  onerror: ((ev: ErrorEvent) => void) | null = null;
  onmessageerror: ((ev: MessageEvent) => void) | null = null;

  postMessage(msg: ToWorker): void {
    this.sent.push(msg);
  }
  terminate(): void {
    this.terminated = true;
  }
  emit(msg: FromWorker): void {
    this.onmessage?.({ data: msg } as MessageEvent<FromWorker>);
  }
  fail(message: string): void {
    this.onerror?.({ message, preventDefault() {} } as ErrorEvent);
  }
}
