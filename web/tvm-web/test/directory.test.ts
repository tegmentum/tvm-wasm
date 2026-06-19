import { describe, expect, it } from "vitest";
import { Directory } from "../src/directory.js";
import { AllocatorKind, RegionKind, Residency, TvmError } from "../src/types.js";

function dir(poolCount: number, poolCapacity: number): Directory {
  return new Directory(
    Array.from({ length: poolCount }, (_, i) => ({ memoryIndex: i, capacity: poolCapacity })),
  );
}

describe("Directory placement (ported from GuestDirectory)", () => {
  it("places Hot regions in the low band round-robin", () => {
    // n=4 -> mid=2 -> hot band [0,2); four HotHeap regions cycle 0,1,0,1.
    const d = dir(4, 1024);
    const pools = [0, 1, 2, 3].map(() => {
      const id = d.createRegion(RegionKind.HotHeap, 128);
      return d.spanOf(id).poolIndex;
    });
    expect(pools).toEqual([0, 1, 0, 1]);
  });

  it("places Warm regions in the high band", () => {
    const d = dir(4, 1024);
    const pools = [0, 1, 2].map(() => {
      const id = d.createRegion(RegionKind.PageStore, 128);
      return d.spanOf(id).poolIndex;
    });
    expect(pools).toEqual([2, 3, 2]);
  });

  it("falls through to the other band when the preferred one is full", () => {
    const d = dir(2, 100); // mid=1 -> hot band is pool 0 only
    d.createRegion(RegionKind.HotHeap, 100);
    const r1 = d.createRegion(RegionKind.HotHeap, 100);
    expect(d.spanOf(r1).poolIndex).toBe(1);
  });

  it("resolves handles to the correct pool + absolute offset", () => {
    const d = dir(2, 4096);
    const r = d.createRegion(RegionKind.HotHeap, 1024);
    const h = d.alloc(r, 64);
    expect(d.resolve(h)).toMatchObject({ poolIndex: 0, absolute: 0 });
    const h2 = d.alloc(r, 32);
    expect(d.resolve(h2).absolute).toBe(64);
  });

  it("rejects stale handles", () => {
    const d = dir(1, 4096);
    const r = d.createRegion(RegionKind.HotHeap, 1024);
    const h = d.alloc(r, 32);
    expect(() => d.resolve({ ...h, generation: 99 })).toThrowError(TvmError);
  });

  it("fails allocation when no pool has room", () => {
    const d = dir(1, 100);
    d.createRegion(RegionKind.HotHeap, 100);
    expect(() => d.createRegion(RegionKind.HotHeap, 1)).toThrowError(/no pool has room/);
  });
});

describe("Directory residency + LRU (ported from RegionDirectory)", () => {
  it("demotes Hot -> Warm and surfaces an eviction candidate", () => {
    const d = dir(2, 4096);
    const r = d.createRegion(RegionKind.ObjectArena, 256); // spillable, starts Hot
    expect(d.evictionCandidate()).toBeNull();
    d.demote(r);
    expect(d.regionInfo(r).residency).toBe(Residency.Warm);
    expect(d.evictionCandidate()).toBe(r);
  });

  it("refuses to demote non-spillable regions", () => {
    const d = dir(2, 4096);
    const r = d.createRegion(RegionKind.HotHeap, 256); // not spillable
    expect(() => d.demote(r)).toThrowError(/not spillable/);
  });

  it("honors pin: pinned regions are never eviction candidates", () => {
    const d = dir(2, 4096);
    const r = d.createRegion(RegionKind.CodeCache, 256); // pinnable, not spillable
    d.pin(r);
    expect(d.regionInfo(r).pinned).toBe(true);
    expect(() => d.demote(r)).toThrowError(); // not spillable anyway
    expect(d.evictionCandidate()).toBeNull();
  });

  it("evicts oldest-first (LRU back)", () => {
    const d = dir(4, 4096);
    const a = d.createRegion(RegionKind.ObjectArena, 128);
    const b = d.createRegion(RegionKind.ObjectArena, 128);
    d.demote(a); // a pushed front
    d.demote(b); // b pushed front -> [b, a], back = a (oldest)
    expect(d.evictionCandidate()).toBe(a);
  });
});

describe("Directory span free-list (browser Cold-tier reclaim)", () => {
  it("reuses an evicted region's span for a later allocation", () => {
    const d = dir(1, 1000);
    const a = d.createRegion(RegionKind.ObjectArena, 400, AllocatorKind.Bump);
    const aSpan = d.spanOf(a);
    expect(aSpan.baseOffset).toBe(0);

    d.demote(a);
    const reclaimed = d.markEvicted(a);
    expect(reclaimed.baseOffset).toBe(0);
    expect(d.regionInfo(a).residency).toBe(Residency.Cold);

    // A new same-size region should reclaim a's freed span, not bump.
    const b = d.createRegion(RegionKind.ObjectArena, 400);
    expect(d.spanOf(b).baseOffset).toBe(0);
  });

  it("drops resident accounting for Cold regions", () => {
    const d = dir(2, 4096);
    const a = d.createRegion(RegionKind.ObjectArena, 256);
    const b = d.createRegion(RegionKind.ObjectArena, 256);
    expect(d.residentBytes()).toBe(512);
    d.demote(a);
    d.markEvicted(a);
    expect(d.residentBytes()).toBe(256);
    void b;
  });

  it("prepareReload hands a Cold region a fresh span and markResident makes it Hot", () => {
    const d = dir(2, 4096);
    const a = d.createRegion(RegionKind.ObjectArena, 256);
    d.demote(a);
    d.markEvicted(a);
    const span = d.prepareReload(a);
    expect(span.capacity).toBe(256);
    d.markResident(a);
    expect(d.regionInfo(a).residency).toBe(Residency.Hot);
  });
});
