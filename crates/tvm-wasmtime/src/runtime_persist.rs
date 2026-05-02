//! Spill/load helpers for runtime-bound memory regions.
//!
//! `RegionDirectory` (in tvm-core) handles spill/load only for context-free
//! memory backends like `VecBackedRegion`. Wasmtime memories require a
//! `Store` to access their data, so they need a parallel API. These helpers
//! sit at the same level as `RegionDirectory::spill_region` but for
//! `RuntimeMemoryRegion<Cx>` implementations.
//!
//! Typical use:
//! ```ignore
//! // Spill: write the live memory to a backing store.
//! spill_runtime_region(&memory, &store, &mut backing, region_id, generation)?;
//!
//! // Later, load: reconstruct a fresh region from the same store.
//! let memory = load_runtime_region::<Store<()>, WasmtimeMemoryRegion>(
//!     &mut store, &mut backing, region_id, generation)?;
//! ```
//!
//! Both helpers are generic over the runtime context type so they work for
//! any future runtime that satisfies `RuntimeMemoryRegion<Cx>`, not just
//! wasmtime.

use tvm_core::{BackingStore, Result};

use crate::memory_factory::RuntimeMemoryRegion;

/// Spill a runtime-bound region's contents to the backing store. The region
/// itself is not consumed — the caller owns its lifecycle. After spilling,
/// dropping the region (or replacing it) frees the runtime's memory budget.
pub fn spill_runtime_region<Cx, M, B>(
    region: &M,
    cx: &Cx,
    backing: &mut B,
    region_id: u16,
    generation: u16,
) -> Result<()>
where
    M: RuntimeMemoryRegion<Cx>,
    B: BackingStore,
{
    let bytes = region.snapshot(cx)?;
    backing.spill(region_id, generation, &bytes)
}

/// Reconstruct a runtime-bound region from the backing store. Allocates a
/// fresh memory in the supplied context — the previous region (if any) must
/// have already been dropped, since runtimes typically can't reuse a memory
/// slot under the same store.
pub fn load_runtime_region<Cx, M, B>(
    cx: &mut Cx,
    backing: &mut B,
    region_id: u16,
    generation: u16,
) -> Result<M>
where
    M: RuntimeMemoryRegion<Cx>,
    B: BackingStore,
{
    let bytes = backing.load(region_id, generation)?;
    M::restore(cx, bytes)
}
