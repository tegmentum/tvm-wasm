import { Worker } from "node:worker_threads";
import { fileURLToPath } from "node:url";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { SyncSpillClient, makeChannel } from "../src/sync-channel.js";

// Verifies the SharedArrayBuffer + Atomics handshake cross-thread: the real
// SyncSpillClient (main thread; Node permits Atomics.wait there) talks to a
// conforming server running in a worker. OPFS itself is browser-only, so the
// server echoes from memory — this proves the channel mechanism, not OPFS.
const serverPath = fileURLToPath(new URL("./fixtures/sync-server.mjs", import.meta.url));
let worker: Worker;
let client: SyncSpillClient;

beforeAll(() => {
  const sab = makeChannel(1 << 20); // 1 MiB payload
  worker = new Worker(serverPath, { workerData: { sab } });
  client = new SyncSpillClient(sab);
});

afterAll(async () => {
  await worker.terminate();
});

function pattern(len: number, seed: number): Uint8Array {
  const out = new Uint8Array(len);
  for (let i = 0; i < len; i++) out[i] = (i * 17 + seed) & 0xff;
  return out;
}

describe("SyncSpillClient over SharedArrayBuffer", () => {
  it("spills and synchronously loads bytes back", () => {
    const data = pattern(4096, 5);
    client.spill(42, 1, data);
    const got = client.load(42, 1);
    expect(got).toEqual(data);
  });

  it("supports multiple regions and generations", () => {
    client.spill(1, 7, pattern(256, 1));
    client.spill(1, 8, pattern(256, 2));
    expect(client.load(1, 7)).toEqual(pattern(256, 1));
    expect(client.load(1, 8)).toEqual(pattern(256, 2));
  });

  it("throws on loading a deleted region", () => {
    client.spill(9, 1, pattern(64, 3));
    client.delete(9, 1);
    expect(() => client.load(9, 1)).toThrowError();
  });
});
