/**
 * Region directory for the browser host.
 *
 * Merges three Rust sources into one JS-owned controller:
 *   - `tvm-guest-mm::GuestDirectory` — region creation, band-based pool
 *     placement, bump sub-allocation, handle resolve.
 *   - `tvm-core::RegionDirectory` — residency state machine, warm LRU,
 *     pin/unpin (crates/tvm-core/src/directory.rs:304-440).
 *   - NEW: a per-pool span free-list. wasm linear memory cannot shrink, so
 *     eviction does not free pages — its value is letting a Cold region's
 *     span be *reclaimed and reused* by a later allocation while the bytes
 *     sit safely in the backing store. That is the whole point of the
 *     browser Cold tier.
 *
 * This class is synchronous bookkeeping only. The async OPFS I/O is
 * orchestrated by `Tvm` (tvm.ts), which reads/writes pool bytes around the
 * `markEvicted` / `prepareReload` / `markResident` transitions.
 */
import { BumpAllocator } from "./allocator.js";
import { policyForKind } from "./policy.js";
import {
  AllocatorKind,
  type Handle,
  RegionKind,
  Residency,
  TvmError,
} from "./types.js";

export interface RegionMeta {
  id: number;
  generation: number;
  kind: RegionKind;
  capacity: number;
  used: number;
  residency: Residency;
  pinned: boolean;
  pinnable: boolean;
  spillable: boolean;
}

/** A contiguous reclaimed region of pool address space. */
interface Span {
  offset: number;
  size: number;
}

interface Pool {
  memoryIndex: number;
  capacity: number;
  /** Bump high-water mark; only ever grows. */
  used: number;
  /** Reclaimed spans below the high-water mark, available for reuse. */
  free: Span[];
}

interface RegionEntry {
  meta: RegionMeta;
  poolIndex: number;
  baseOffset: number;
  allocator: BumpAllocator;
}

/** Concrete placement of a region's bytes within a pool. */
export interface RegionSpan {
  poolIndex: number;
  memoryIndex: number;
  baseOffset: number;
  capacity: number;
}

export class Directory {
  private readonly pools: Pool[];
  private readonly slots: (RegionEntry | null)[] = [null];
  private nextId = 1;
  private cursorHot = 0;
  private cursorWarm = 0;
  /** Warm regions, eviction-eligible. Front = most-recently demoted; back = LRU. */
  private warmLru: number[] = [];

  constructor(poolCapacities: { memoryIndex: number; capacity: number }[]) {
    this.pools = poolCapacities.map((p) => ({
      memoryIndex: p.memoryIndex,
      capacity: p.capacity,
      used: 0,
      free: [],
    }));
  }

  poolCount(): number {
    return this.pools.length;
  }

  // ---- creation / allocation (GuestDirectory) ----------------------------

  createRegion(
    kind: RegionKind,
    capacity: number,
    _allocator: AllocatorKind = AllocatorKind.Bump,
  ): number {
    const id = this.nextId;
    this.nextId += 1;
    const policy = policyForKind(kind);
    const placement = this.allocateSpan(capacity, policy.initialResidency);
    if (placement === null) throw new TvmError("AllocationFailed", "no pool has room");

    const entry: RegionEntry = {
      meta: {
        id,
        generation: 1,
        kind,
        capacity,
        used: 0,
        residency: policy.initialResidency,
        pinned: false,
        pinnable: policy.pinnable,
        spillable: policy.spillable,
      },
      poolIndex: placement.poolIndex,
      baseOffset: placement.baseOffset,
      allocator: new BumpAllocator(capacity),
    };
    this.slots[id] = entry;
    if (policy.initialResidency === Residency.Warm) this.lruPushFront(id);
    return id;
  }

  alloc(regionId: number, size: number, align = 1): Handle {
    const entry = this.entry(regionId);
    const offset = entry.allocator.alloc(size, align);
    entry.meta.used = entry.allocator.usedBytes();
    return { regionId, generation: entry.meta.generation, offset };
  }

  /** Resolve a handle to `(poolIndex, absolute offset within pool)`. */
  resolve(handle: Handle): { poolIndex: number; memoryIndex: number; absolute: number } {
    const entry = this.entry(handle.regionId);
    if (entry.meta.generation !== handle.generation) {
      throw new TvmError("StaleHandle", `region ${handle.regionId} generation mismatch`);
    }
    if (entry.meta.residency === Residency.Cold) {
      throw new TvmError("NotResident", `region ${handle.regionId} is spilled; promote first`);
    }
    const absolute = entry.baseOffset + handle.offset;
    return {
      poolIndex: entry.poolIndex,
      memoryIndex: this.pools[entry.poolIndex]!.memoryIndex,
      absolute,
    };
  }

  regionInfo(regionId: number): Readonly<RegionMeta> {
    return this.entry(regionId).meta;
  }

  spanOf(regionId: number): RegionSpan {
    const e = this.entry(regionId);
    return {
      poolIndex: e.poolIndex,
      memoryIndex: this.pools[e.poolIndex]!.memoryIndex,
      baseOffset: e.baseOffset,
      capacity: e.meta.capacity,
    };
  }

  // ---- residency state machine (RegionDirectory) -------------------------

  pin(regionId: number): void {
    const entry = this.entry(regionId);
    if (!entry.meta.pinnable) throw new TvmError("PolicyViolation", "region is not pinnable");
    entry.meta.pinned = true;
    this.lruRemove(regionId);
  }

  unpin(regionId: number): void {
    const entry = this.entry(regionId);
    entry.meta.pinned = false;
    if (entry.meta.residency === Residency.Warm) this.lruPushFront(regionId);
  }

  /** Hot -> Warm: mark eviction-eligible (still resident). */
  demote(regionId: number): void {
    const entry = this.entry(regionId);
    if (entry.meta.pinned) throw new TvmError("Pinned");
    if (!entry.meta.spillable) throw new TvmError("PolicyViolation", "region is not spillable");
    if (entry.meta.residency === Residency.Hot) {
      entry.meta.residency = Residency.Warm;
      this.lruPushFront(regionId);
    }
  }

  /**
   * Transition a resident, spillable, unpinned region to Cold and reclaim
   * its pool span. Returns the span the caller must read bytes from BEFORE
   * the span is recycled. Caller is responsible for having already copied
   * the bytes to the backing store, or for spilling them after reading.
   */
  markEvicted(regionId: number): RegionSpan {
    const entry = this.entry(regionId);
    if (entry.meta.pinned) throw new TvmError("Pinned");
    if (!entry.meta.spillable) throw new TvmError("PolicyViolation", "region is not spillable");
    if (entry.meta.residency === Residency.Cold) {
      throw new TvmError("PolicyViolation", "region already cold");
    }
    const span = this.spanOf(regionId);
    this.freeSpan(entry.poolIndex, entry.baseOffset, entry.meta.capacity);
    entry.meta.residency = Residency.Cold;
    this.lruRemove(regionId);
    return span;
  }

  /**
   * Allocate a fresh span for a Cold region so its bytes can be reloaded.
   * Updates the region's placement and returns the new span. The region
   * stays Cold until `markResident` is called (after bytes are written).
   */
  prepareReload(regionId: number): RegionSpan {
    const entry = this.entry(regionId);
    if (entry.meta.residency !== Residency.Cold) {
      throw new TvmError("PolicyViolation", "region is not cold");
    }
    const placement = this.allocateSpan(entry.meta.capacity, Residency.Hot);
    if (placement === null) throw new TvmError("AllocationFailed", "no room to reload region");
    entry.poolIndex = placement.poolIndex;
    entry.baseOffset = placement.baseOffset;
    return this.spanOf(regionId);
  }

  markResident(regionId: number): void {
    const entry = this.entry(regionId);
    entry.meta.residency = Residency.Hot;
    this.lruRemove(regionId);
  }

  /** Oldest evictable warm region, or null. Mirrors `evict_warm_region`'s scan. */
  evictionCandidate(): number | null {
    for (let i = this.warmLru.length - 1; i >= 0; i--) {
      const id = this.warmLru[i]!;
      const entry = this.slots[id];
      if (!entry || entry.meta.pinned || entry.meta.residency !== Residency.Warm) {
        this.warmLru.splice(i, 1);
        continue;
      }
      return id;
    }
    return null;
  }

  /** Bytes occupying live pool spans (everything not Cold). */
  residentBytes(): number {
    let total = 0;
    for (const entry of this.slots) {
      if (entry && entry.meta.residency !== Residency.Cold) total += entry.meta.capacity;
    }
    return total;
  }

  liveRegionIds(): number[] {
    const ids: number[] = [];
    for (const entry of this.slots) if (entry) ids.push(entry.meta.id);
    return ids;
  }

  // ---- pool span allocation (free-list + bump) ---------------------------

  private allocateSpan(
    capacity: number,
    residency: Residency,
  ): { poolIndex: number; baseOffset: number } | null {
    const n = this.pools.length;
    if (n === 0) return null;
    const mid = Math.ceil(n / 2);
    const prefersHot = residency === Residency.Hot;
    const bands: [number, number, "hot" | "warm"][] = prefersHot
      ? [
          [0, mid, "hot"],
          [mid, n, "warm"],
        ]
      : [
          [mid, n, "warm"],
          [0, mid, "hot"],
        ];

    for (const [lo, hi, which] of bands) {
      if (lo === hi) continue;
      const span = hi - lo;
      const cursorStart = which === "hot" ? this.cursorHot : this.cursorWarm;
      for (let offset = 0; offset < span; offset++) {
        const idx = lo + ((cursorStart + offset) % span);
        const baseOffset = this.allocateFromPool(this.pools[idx]!, capacity);
        if (baseOffset !== null) {
          const nextCursor = (cursorStart + offset + 1) % span;
          if (which === "hot") this.cursorHot = nextCursor;
          else this.cursorWarm = nextCursor;
          return { poolIndex: idx, baseOffset };
        }
      }
    }
    return null;
  }

  /** First-fit from the free-list, else bump. Returns base offset or null. */
  private allocateFromPool(pool: Pool, capacity: number): number | null {
    for (let i = 0; i < pool.free.length; i++) {
      const span = pool.free[i]!;
      if (span.size >= capacity) {
        const offset = span.offset;
        if (span.size === capacity) {
          pool.free.splice(i, 1);
        } else {
          span.offset += capacity;
          span.size -= capacity;
        }
        return offset;
      }
    }
    if (pool.capacity - pool.used >= capacity) {
      const offset = pool.used;
      pool.used += capacity;
      return offset;
    }
    return null;
  }

  /** Return a span to the pool free-list, coalescing with adjacent spans. */
  private freeSpan(poolIndex: number, offset: number, size: number): void {
    const pool = this.pools[poolIndex]!;
    pool.free.push({ offset, size });
    pool.free.sort((a, b) => a.offset - b.offset);
    const merged: Span[] = [];
    for (const span of pool.free) {
      const last = merged[merged.length - 1];
      if (last && last.offset + last.size === span.offset) {
        last.size += span.size;
      } else {
        merged.push({ ...span });
      }
    }
    // Drop a trailing free span that meets the high-water mark back to bump.
    const tail = merged[merged.length - 1];
    if (tail && tail.offset + tail.size === pool.used) {
      pool.used = tail.offset;
      merged.pop();
    }
    pool.free = merged;
  }

  // ---- LRU helpers -------------------------------------------------------

  private lruPushFront(regionId: number): void {
    this.lruRemove(regionId);
    this.warmLru.unshift(regionId);
  }

  private lruRemove(regionId: number): void {
    const i = this.warmLru.indexOf(regionId);
    if (i >= 0) this.warmLru.splice(i, 1);
  }

  private entry(regionId: number): RegionEntry {
    const entry = this.slots[regionId];
    if (!entry) throw new TvmError("RegionNotFound", `region ${regionId} not found`);
    return entry;
  }
}
