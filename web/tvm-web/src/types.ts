/**
 * Core TVM types, ported 1:1 from the Rust crates so the browser host
 * mirrors the same semantics:
 *   - RegionKind / Residency: crates/tvm-core/src/region.rs, residency.rs
 *   - Handle pack/unpack:      crates/tvm-core/src/handle.rs
 *   - AllocatorKind:           crates/tvm-core/src/allocator.rs
 */

export enum RegionKind {
  HotHeap = "HotHeap",
  CodeCache = "CodeCache",
  DeviceState = "DeviceState",
  ObjectArena = "ObjectArena",
  BlobArena = "BlobArena",
  PageStore = "PageStore",
  Scratch = "Scratch",
}

/** Mirrors `tvm-core::Residency`. */
export enum Residency {
  /** Resident, live, recently touched. */
  Hot = "Hot",
  /** Resident but LRU-eligible — a candidate for eviction. */
  Warm = "Warm",
  /** Spilled to the backing store (OPFS); its pool span has been reclaimed. */
  Cold = "Cold",
}

export enum AllocatorKind {
  Bump = "Bump",
}

/**
 * A region handle. Matches the Rust `Handle` field layout
 * (`region_id: u16, generation: u16, offset: u32`). `region_id === 0`
 * is the null handle.
 */
export interface Handle {
  readonly regionId: number;
  readonly generation: number;
  readonly offset: number;
}

export const NULL_HANDLE: Handle = { regionId: 0, generation: 0, offset: 0 };

/** Pack a handle into a single bigint, matching Rust `Handle::pack`. */
export function packHandle(h: Handle): bigint {
  return (
    (BigInt(h.regionId) << 48n) |
    (BigInt(h.generation) << 32n) |
    BigInt(h.offset >>> 0)
  );
}

/** Inverse of {@link packHandle}; matches Rust `Handle::unpack`. */
export function unpackHandle(packed: bigint): Handle {
  return {
    regionId: Number((packed >> 48n) & 0xffffn),
    generation: Number((packed >> 32n) & 0xffffn),
    offset: Number(packed & 0xffff_ffffn),
  };
}

export function isNullHandle(h: Handle): boolean {
  return h.regionId === 0 && h.generation === 0 && h.offset === 0;
}

/** Error kinds mirroring the relevant `tvm-core::TvmError` variants. */
export class TvmError extends Error {
  constructor(
    public readonly kind: TvmErrorKind,
    message?: string,
  ) {
    super(message ?? kind);
    this.name = "TvmError";
  }
}

export type TvmErrorKind =
  | "AllocationFailed"
  | "RegionNotFound"
  | "StaleHandle"
  | "OutOfBounds"
  | "Pinned"
  | "PolicyViolation"
  | "NotResident"
  | "BackingStore";
