/**
 * `Tvm` — the browser host facade. Instantiates a multi-memory guest module,
 * owns the region {@link Directory}, and orchestrates the OPFS-backed Cold
 * tier.
 *
 * Cooperative contract: native wasm loads cannot trap, so a spilled region
 * cannot be transparently faulted back. Callers must `evict` only regions
 * they are done touching, and `await promote(region)` before accessing a
 * region again. `read`/`write` throw `NotResident` if the region is Cold.
 */
import { type SpillBackend } from "./backend.js";
import { Directory, type RegionMeta } from "./directory.js";
import { PoolSet } from "./pools.js";
import { AllocatorKind, type Handle, RegionKind, Residency, TvmError } from "./types.js";

export interface TvmOptions {
  backend: SpillBackend;
  /**
   * Soft cap on resident bytes (sum of non-Cold region capacities). When a
   * new allocation would exceed it, the host evicts warm regions (LRU first)
   * to OPFS until it fits or no evictable region remains.
   */
  budgetBytes?: number;
}

export class Tvm {
  private constructor(
    private readonly instance: WebAssembly.Instance,
    private readonly pools: PoolSet,
    private readonly directory: Directory,
    private readonly backend: SpillBackend,
    private budgetBytes: number | undefined,
  ) {}

  static async instantiate(
    source: BufferSource | Response | Promise<Response>,
    opts: TvmOptions,
  ): Promise<Tvm> {
    const imports: WebAssembly.Imports = {};
    let instance: WebAssembly.Instance;
    if (source instanceof Response || source instanceof Promise) {
      ({ instance } = await WebAssembly.instantiateStreaming(source, imports));
    } else {
      ({ instance } = await WebAssembly.instantiate(source, imports));
    }
    const pools = new PoolSet(instance.exports as Record<string, unknown>);
    const directory = new Directory(pools.poolCapacities());
    return new Tvm(instance, pools, directory, opts.backend, opts.budgetBytes);
  }

  get exports(): WebAssembly.Exports {
    return this.instance.exports;
  }

  poolCount(): number {
    return this.pools.poolCount();
  }

  residentBytes(): number {
    return this.directory.residentBytes();
  }

  regionInfo(regionId: number): Readonly<RegionMeta> {
    return this.directory.regionInfo(regionId);
  }

  // ---- allocation --------------------------------------------------------

  async createRegion(
    kind: RegionKind,
    capacity: number,
    allocator: AllocatorKind = AllocatorKind.Bump,
  ): Promise<number> {
    await this.ensureBudgetFor(capacity);
    return this.directory.createRegion(kind, capacity, allocator);
  }

  alloc(regionId: number, size: number, align = 1): Handle {
    const handle = this.directory.alloc(regionId, size, align);
    const span = this.directory.spanOf(regionId);
    // Grow the pool so the freshly handed-out range is addressable.
    this.pools.ensureSize(span.memoryIndex, span.baseOffset + span.capacity);
    return handle;
  }

  // ---- data plane (direct pool buffer access) ----------------------------

  write(handle: Handle, bytes: Uint8Array): void {
    const { memoryIndex, absolute } = this.directory.resolve(handle);
    this.pools.writeRange(memoryIndex, absolute, bytes);
  }

  read(handle: Handle, length: number): Uint8Array {
    const { memoryIndex, absolute } = this.directory.resolve(handle);
    return this.pools.readRange(memoryIndex, absolute, length);
  }

  // ---- residency control -------------------------------------------------

  pin(regionId: number): void {
    this.directory.pin(regionId);
  }

  unpin(regionId: number): void {
    this.directory.unpin(regionId);
  }

  /** Mark a region eviction-eligible (Hot -> Warm). */
  demote(regionId: number): void {
    this.directory.demote(regionId);
  }

  /** Spill a region's bytes to OPFS and reclaim its pool span. */
  async evict(regionId: number): Promise<void> {
    const info = this.directory.regionInfo(regionId);
    const generation = info.generation;
    // markEvicted runs the policy checks and reclaims the span synchronously;
    // the bytes are still intact in memory until a later allocation reuses
    // the span, and readRange copies them out, so no await may interleave
    // between these two synchronous calls.
    const span = this.directory.markEvicted(regionId);
    const bytes = this.pools.readRange(span.memoryIndex, span.baseOffset, span.capacity);
    await this.backend.spill(regionId, generation, bytes);
  }

  /** Reload a Cold region from OPFS into a fresh span and mark it Hot. */
  async promote(regionId: number): Promise<void> {
    const info = this.directory.regionInfo(regionId);
    if (info.residency !== Residency.Cold) return;
    const bytes = await this.backend.load(regionId, info.generation);
    const span = this.directory.prepareReload(regionId);
    this.pools.writeRange(span.memoryIndex, span.baseOffset, bytes);
    this.directory.markResident(regionId);
  }

  /** Evict the oldest warm region, if any. Returns the evicted id or null. */
  async evictColdest(): Promise<number | null> {
    const candidate = this.directory.evictionCandidate();
    if (candidate === null) return null;
    await this.evict(candidate);
    return candidate;
  }

  setBudget(bytes: number | undefined): void {
    this.budgetBytes = bytes;
  }

  private async ensureBudgetFor(extra: number): Promise<void> {
    if (this.budgetBytes === undefined) return;
    while (this.directory.residentBytes() + extra > this.budgetBytes) {
      const evicted = await this.evictColdest();
      if (evicted === null) {
        // Nothing left to evict; allow the allocation (the browser will cap
        // us via memory.grow failure if we truly run out).
        return;
      }
    }
  }

  /** Internal accessor for tests. */
  _directory(): Directory {
    return this.directory;
  }

  /** Internal accessor for tests. */
  _pools(): PoolSet {
    return this.pools;
  }

  /** Reclaim worker/handles. */
  async close(): Promise<void> {
    if (this.backend.close) await this.backend.close();
  }
}

export { TvmError };
