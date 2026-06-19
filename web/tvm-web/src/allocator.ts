/**
 * Bump allocator — port of `tvm-core::BumpAllocator`
 * (crates/tvm-core/src/allocator.rs). Used for region-internal
 * sub-allocation. `dealloc` is a no-op (bump never frees individual
 * allocations); the whole region is freed/spilled at the directory level.
 */
import { TvmError } from "./types.js";

const U32_MAX = 0xffff_ffff;

function alignUp(value: number, align: number): number {
  if (align <= 1) return value;
  // align is a power of two in practice, but mirror the Rust checked math.
  const rem = value % align;
  if (rem === 0) return value;
  const padded = value + (align - rem);
  if (padded > U32_MAX) throw new TvmError("AllocationFailed", "alignment overflow");
  return padded;
}

export class BumpAllocator {
  private used = 0;

  constructor(private readonly capacity: number) {}

  alloc(size: number, align: number): number {
    const aligned = alignUp(this.used, align);
    const end = aligned + size;
    if (end > this.capacity || end > U32_MAX) {
      throw new TvmError("AllocationFailed", "region out of capacity");
    }
    this.used = end;
    return aligned;
  }

  usedBytes(): number {
    return this.used;
  }

  reset(): void {
    this.used = 0;
  }
}
