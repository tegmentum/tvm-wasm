/**
 * @tvm/web — browser host for TVM multi-memory guests with an OPFS-backed
 * Cold tier. See README.md and docs/architecture.md for the design.
 */
export { Tvm, type TvmOptions } from "./tvm.js";
export { Directory, type RegionMeta, type RegionSpan } from "./directory.js";
export { PoolSet } from "./pools.js";
export {
  type SpillBackend,
  InMemoryBackend,
  backingKey,
} from "./backend.js";
export { OpfsBackend, type OpfsBackendOptions } from "./opfs-backend.js";
export { policyForKind, type PlacementPolicy } from "./policy.js";
export { BumpAllocator } from "./allocator.js";
export {
  RegionKind,
  Residency,
  AllocatorKind,
  TvmError,
  type TvmErrorKind,
  type Handle,
  NULL_HANDLE,
  packHandle,
  unpackHandle,
  isNullHandle,
} from "./types.js";
