/**
 * Optional synchronous spill/load fast path over `SharedArrayBuffer` +
 * `Atomics.wait`.
 *
 * The default {@link OpfsBackend} is async, which is sufficient because the JS
 * host spills only while the guest is quiesced. But a guest that must load a
 * Cold region from *within* a tight loop — without yielding to the event loop
 * — needs a synchronous call. `Atomics.wait` provides that, with two hard
 * constraints baked into this design:
 *
 *   1. `Atomics.wait` throws on the main thread in browsers, so the caller
 *      (the guest) must run inside a Web Worker. The serving OPFS worker is a
 *      *separate, dedicated* worker that loops on the channel.
 *   2. The serving side runs synchronous handlers. OPFS
 *      `FileSystemSyncAccessHandle`s must therefore be pre-opened (opening is
 *      async); see `docs/architecture.md`. Suited to a fixed working set.
 *
 * Page cross-origin isolation (COOP: same-origin, COEP: require-corp) is
 * required for `SharedArrayBuffer`; the Vite config sets these headers.
 *
 * Wire layout (one in-flight request at a time):
 *   Int32 header: [SIGNAL, OP, ID, GEN, LEN, STATUS]  (24 bytes)
 *   payload bytes follow at HEADER_BYTES.
 */
export const HEADER_I32 = 6;
export const HEADER_BYTES = HEADER_I32 * 4;

const SIGNAL = 0;
const OP = 1;
const ID = 2;
const GEN = 3;
const LEN = 4;
const STATUS = 5;

const SIG_IDLE = 0; // server waiting for a request
const SIG_REQ = 1; // client posted a request
const SIG_RES = 2; // server posted a response

export const OP_LOAD = 1;
export const OP_SPILL = 2;
export const OP_DELETE = 3;

const STATUS_OK = 0;
const STATUS_ERR = 1;

export interface SyncHandlers {
  load(id: number, gen: number): Uint8Array;
  spill(id: number, gen: number, bytes: Uint8Array): void;
  delete(id: number, gen: number): void;
}

export function makeChannel(maxPayloadBytes: number): SharedArrayBuffer {
  return new SharedArrayBuffer(HEADER_BYTES + maxPayloadBytes);
}

/** Client side — runs in the guest's worker. All calls block. */
export class SyncSpillClient {
  private readonly h: Int32Array;
  private readonly payload: Uint8Array;

  constructor(private readonly sab: SharedArrayBuffer) {
    this.h = new Int32Array(sab, 0, HEADER_I32);
    this.payload = new Uint8Array(sab, HEADER_BYTES);
  }

  load(id: number, gen: number): Uint8Array {
    this.request(OP_LOAD, id, gen, 0);
    const len = Atomics.load(this.h, LEN);
    const out = this.payload.slice(0, len);
    this.finish();
    return out;
  }

  spill(id: number, gen: number, bytes: Uint8Array): void {
    if (bytes.byteLength > this.payload.byteLength) {
      throw new RangeError("region exceeds sync channel payload capacity");
    }
    this.payload.set(bytes, 0);
    this.request(OP_SPILL, id, gen, bytes.byteLength);
    this.finish();
  }

  delete(id: number, gen: number): void {
    this.request(OP_DELETE, id, gen, 0);
    this.finish();
  }

  private request(op: number, id: number, gen: number, len: number): void {
    Atomics.store(this.h, OP, op);
    Atomics.store(this.h, ID, id);
    Atomics.store(this.h, GEN, gen);
    Atomics.store(this.h, LEN, len);
    Atomics.store(this.h, SIGNAL, SIG_REQ);
    Atomics.notify(this.h, SIGNAL);
    Atomics.wait(this.h, SIGNAL, SIG_REQ); // block until SIG_RES
    if (Atomics.load(this.h, STATUS) === STATUS_ERR) {
      this.finish();
      throw new Error(`sync spill op ${op} failed for region ${id} gen ${gen}`);
    }
  }

  private finish(): void {
    Atomics.store(this.h, SIGNAL, SIG_IDLE);
    Atomics.notify(this.h, SIGNAL);
  }
}

/**
 * Server side — runs in the dedicated OPFS worker. Blocks on each request,
 * runs the synchronous handler, posts the response. Loops forever; call from
 * a worker entry point. `onError` lets the worker surface handler failures.
 */
export function serveSyncChannel(sab: SharedArrayBuffer, handlers: SyncHandlers): never {
  const h = new Int32Array(sab, 0, HEADER_I32);
  const payload = new Uint8Array(sab, HEADER_BYTES);
  for (;;) {
    Atomics.wait(h, SIGNAL, SIG_IDLE); // block until a request arrives
    serveOnce(h, payload, handlers);
    Atomics.wait(h, SIGNAL, SIG_RES); // block until the client acks
  }
}

/** Process exactly one already-posted request. Exported for testing. */
export function serveOnce(h: Int32Array, payload: Uint8Array, handlers: SyncHandlers): void {
  const op = Atomics.load(h, OP);
  const id = Atomics.load(h, ID);
  const gen = Atomics.load(h, GEN);
  const len = Atomics.load(h, LEN);
  try {
    if (op === OP_LOAD) {
      const bytes = handlers.load(id, gen);
      payload.set(bytes, 0);
      Atomics.store(h, LEN, bytes.byteLength);
    } else if (op === OP_SPILL) {
      handlers.spill(id, gen, payload.slice(0, len));
    } else if (op === OP_DELETE) {
      handlers.delete(id, gen);
    }
    Atomics.store(h, STATUS, STATUS_OK);
  } catch {
    Atomics.store(h, STATUS, STATUS_ERR);
    Atomics.store(h, LEN, 0);
  }
  Atomics.store(h, SIGNAL, SIG_RES);
  Atomics.notify(h, SIGNAL);
}
