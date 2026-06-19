// Conforming sync-channel server for the cross-thread test. Plain JS so it
// can run directly in a node:worker_threads Worker without a TS loader. It
// mirrors the wire protocol in src/sync-channel.ts and echoes spilled bytes
// from an in-memory map (standing in for OPFS).
import { workerData } from "node:worker_threads";

const HEADER_I32 = 6;
const HEADER_BYTES = HEADER_I32 * 4;
const SIGNAL = 0, OP = 1, ID = 2, GEN = 3, LEN = 4, STATUS = 5;
const SIG_IDLE = 0, SIG_REQ = 1, SIG_RES = 2;
const OP_LOAD = 1, OP_SPILL = 2, OP_DELETE = 3;

const sab = workerData.sab;
const h = new Int32Array(sab, 0, HEADER_I32);
const payload = new Uint8Array(sab, HEADER_BYTES);
const store = new Map();
const key = (id, gen) => `${id}:${gen}`;

for (;;) {
  Atomics.wait(h, SIGNAL, SIG_IDLE);
  const op = Atomics.load(h, OP);
  const id = Atomics.load(h, ID);
  const gen = Atomics.load(h, GEN);
  const len = Atomics.load(h, LEN);
  try {
    if (op === OP_LOAD) {
      const bytes = store.get(key(id, gen));
      if (!bytes) throw new Error("missing");
      payload.set(bytes, 0);
      Atomics.store(h, LEN, bytes.byteLength);
    } else if (op === OP_SPILL) {
      store.set(key(id, gen), payload.slice(0, len));
    } else if (op === OP_DELETE) {
      store.delete(key(id, gen));
    }
    Atomics.store(h, STATUS, 0);
  } catch {
    Atomics.store(h, STATUS, 1);
    Atomics.store(h, LEN, 0);
  }
  Atomics.store(h, SIGNAL, SIG_RES);
  Atomics.notify(h, SIGNAL);
  Atomics.wait(h, SIGNAL, SIG_RES);
}
