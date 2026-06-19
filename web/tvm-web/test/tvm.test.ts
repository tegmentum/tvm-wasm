import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { beforeAll, describe, expect, it } from "vitest";
import { InMemoryBackend } from "../src/backend.js";
import { Tvm } from "../src/tvm.js";
import { RegionKind, Residency } from "../src/types.js";

// Node's V8 supports the multi-memory proposal, so the real generated guest
// runs here — only OPFS is browser-only, and InMemoryBackend stands in for it.
const wasmPath = fileURLToPath(new URL("../public/tvm-guest.wasm", import.meta.url));
let wasmBytes: Uint8Array;

beforeAll(() => {
  wasmBytes = readFileSync(wasmPath);
});

function pattern(len: number, seed: number): Uint8Array {
  const out = new Uint8Array(len);
  for (let i = 0; i < len; i++) out[i] = (i * 31 + seed) & 0xff;
  return out;
}

async function newTvm(budgetBytes?: number): Promise<{ tvm: Tvm; backend: InMemoryBackend }> {
  const backend = new InMemoryBackend();
  const opts = budgetBytes === undefined ? { backend } : { backend, budgetBytes };
  const tvm = await Tvm.instantiate(wasmBytes, opts);
  return { tvm, backend };
}

describe("Tvm over the real multi-memory guest", () => {
  it("exposes all 64 pools", async () => {
    const { tvm } = await newTvm();
    expect(tvm.poolCount()).toBe(64);
  });

  it("writes and reads region bytes through pool buffers", async () => {
    const { tvm } = await newTvm();
    const r = await tvm.createRegion(RegionKind.ObjectArena, 4096);
    const h = tvm.alloc(r, 1024);
    const data = pattern(1024, 7);
    tvm.write(h, data);
    expect(tvm.read(h, 1024)).toEqual(data);
  });

  it("round-trips a region through evict (OPFS) and promote", async () => {
    const { tvm, backend } = await newTvm();
    const r = await tvm.createRegion(RegionKind.ObjectArena, 8192);
    const h = tvm.alloc(r, 2048);
    const data = pattern(2048, 13);
    tvm.write(h, data);

    tvm.demote(r);
    await tvm.evict(r);
    expect(tvm.regionInfo(r).residency).toBe(Residency.Cold);
    expect(backend.size()).toBe(1);
    // Cold regions cannot be read (no transparent fault).
    expect(() => tvm.read(h, 2048)).toThrowError(/spilled/);

    await tvm.promote(r);
    expect(tvm.regionInfo(r).residency).toBe(Residency.Hot);
    expect(tvm.read(h, 2048)).toEqual(data);
  });

  it("preserves evicted data even after its old span memory is clobbered", async () => {
    // The core durability proof: once a region is evicted its pool span is
    // reclaimable, so its memory may be overwritten by reuse. The spilled
    // bytes live in the backing store and reload intact regardless.
    const { tvm } = await newTvm();
    const a = await tvm.createRegion(RegionKind.ObjectArena, 4096);
    const ha = tvm.alloc(a, 4096);
    const dataA = pattern(4096, 1);
    tvm.write(ha, dataA);
    const aSpan = tvm._directory().spanOf(a);

    tvm.demote(a);
    await tvm.evict(a);

    // Simulate the span being reused: clobber the old memory entirely.
    tvm._pools().writeRange(aSpan.memoryIndex, aSpan.baseOffset, pattern(4096, 2));

    // a's bytes are gone from RAM but live in the backing store.
    await tvm.promote(a);
    expect(tvm.read(ha, 4096)).toEqual(dataA);
  });

  it("auto-evicts warm regions to stay under the resident budget", async () => {
    // Budget = 12 KiB; three 8 KiB arenas can't all be resident.
    const { tvm } = await newTvm(12 * 1024);
    const ids: number[] = [];
    for (let i = 0; i < 3; i++) {
      const r = await tvm.createRegion(RegionKind.ObjectArena, 8 * 1024);
      const h = tvm.alloc(r, 8 * 1024);
      tvm.write(h, pattern(8 * 1024, i));
      tvm.demote(r); // make it eviction-eligible before the next allocation
      ids.push(r);
    }
    expect(tvm.residentBytes()).toBeLessThanOrEqual(12 * 1024);
    const cold = ids.filter((r) => tvm.regionInfo(r).residency === Residency.Cold);
    expect(cold.length).toBeGreaterThan(0);

    // Every region's data survives a promote, regardless of eviction order.
    for (let i = 0; i < ids.length; i++) {
      const r = ids[i]!;
      await tvm.promote(r);
      const info = tvm.regionInfo(r);
      const h = { regionId: r, generation: info.generation, offset: 0 };
      expect(tvm.read(h, 8 * 1024)).toEqual(pattern(8 * 1024, i));
    }
  });
});
