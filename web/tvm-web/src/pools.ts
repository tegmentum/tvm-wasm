/**
 * Accessor over the guest module's exported pool memories.
 *
 * The generated module exports each pool as `mem0..memN` and the dispatch
 * helpers (`tvm_load_u32`, `tvm_store_u32`, `tvm_copy_to_default`, …) — see
 * crates/tvm-guest-mm/src/module.rs and dispatch.rs. The browser host reads
 * and writes pool bytes directly through `WebAssembly.Memory.buffer`, which
 * is both the fastest path for bulk spill/load and all we need for the data
 * plane. Per-element dispatch helpers remain available for callers that want
 * native multi-memory loads from within wasm.
 *
 * Pools are declared with 1 initial page and a 65536-page (4 GiB) max, so
 * they grow lazily — `ensureSize` grows a pool before a region is placed.
 */
const PAGE_SIZE = 65536;

export interface GuestExports {
  [name: string]: unknown;
}

export class PoolSet {
  private readonly memories: WebAssembly.Memory[] = [];

  constructor(private readonly exports: GuestExports) {
    let i = 0;
    for (;;) {
      const mem = exports[`mem${i}`];
      if (!(mem instanceof WebAssembly.Memory)) break;
      this.memories.push(mem);
      i++;
    }
    if (this.memories.length === 0) {
      throw new Error("guest module exports no pool memories (mem0..memN)");
    }
  }

  poolCount(): number {
    return this.memories.length;
  }

  /** Per-pool declared capacity in bytes (the 4 GiB wasm32 max by default). */
  poolCapacities(): { memoryIndex: number; capacity: number }[] {
    return this.memories.map((_, memoryIndex) => ({
      memoryIndex,
      capacity: PAGE_SIZE * 65536,
    }));
  }

  /** Grow pool `memoryIndex` so at least `byteLength` bytes are addressable. */
  ensureSize(memoryIndex: number, byteLength: number): void {
    const mem = this.mem(memoryIndex);
    const haveBytes = mem.buffer.byteLength;
    if (haveBytes >= byteLength) return;
    const needPages = Math.ceil((byteLength - haveBytes) / PAGE_SIZE);
    mem.grow(needPages);
  }

  /** Copy `length` bytes out of a pool. Re-reads `.buffer` (detaches on grow). */
  readRange(memoryIndex: number, offset: number, length: number): Uint8Array {
    const buf = this.mem(memoryIndex).buffer;
    if (offset + length > buf.byteLength) {
      throw new RangeError(`read ${offset}+${length} exceeds pool ${memoryIndex}`);
    }
    return new Uint8Array(buf, offset, length).slice();
  }

  /** Copy bytes into a pool at `offset`, growing if needed. */
  writeRange(memoryIndex: number, offset: number, bytes: Uint8Array): void {
    this.ensureSize(memoryIndex, offset + bytes.byteLength);
    new Uint8Array(this.mem(memoryIndex).buffer).set(bytes, offset);
  }

  /** Zero a pool range (used to prove a span is reusable after eviction). */
  zeroRange(memoryIndex: number, offset: number, length: number): void {
    this.ensureSize(memoryIndex, offset + length);
    new Uint8Array(this.mem(memoryIndex).buffer, offset, length).fill(0);
  }

  private mem(memoryIndex: number): WebAssembly.Memory {
    const mem = this.memories[memoryIndex];
    if (!mem) throw new RangeError(`no pool memory at index ${memoryIndex}`);
    return mem;
  }
}
