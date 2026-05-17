//! `tvm-guest-mm` — pure guest-side TVM with multi-memory pools.
//!
//! Unlike the host-side variant in `tvm-wasmtime`, this crate produces
//! **self-contained wasm modules** that declare all their TVM machinery
//! internally. No host imports, no special engine configuration beyond
//! enabling the multi-memory proposal. The same `.wasm` runs in any
//! engine that supports multi-memory: wasmtime, wasmer, browser engines
//! that have shipped the proposal, etc.
//!
//! ## Design summary
//!
//! - **N internal memories** ("pools") declared at module compile time.
//!   Each pool is its own 32-bit-indexed wasm memory.
//! - **Region directory** lives in pool 0 (which doubles as the
//!   metadata + workload-default memory). Subsequent pools (1..N) are
//!   pure data storage.
//! - **TVM regions are sub-allocations within pools.** Each pool runs
//!   its own bump allocator; cross-pool placement is round-robin or
//!   policy-driven.
//! - **Access dispatch happens in WAT.** Generated functions like
//!   `tvm_load_u32(pool_id, offset) -> u32` use a static match on
//!   `pool_id` to select the correct memory immediate. This is the only
//!   piece that can't be expressed in current Rust source — wasm's
//!   memory immediate is a static parameter.
//!
//! ## Usage shape (skeleton)
//!
//! ```text
//! Module declaration (generated):
//!   (memory 0 1)        ;; pool 0 — also default + metadata
//!   (memory 1 0 65536)  ;; pool 1, grows on demand to 4 GiB
//!   ...
//!   (memory 15 0 65536) ;; pool 15
//!
//!   ;; Dispatch functions:
//!   (func $tvm_load_u32 (param $pool i32) (param $off i32) (result i32) ...)
//!   (func $tvm_store_u32 (param $pool i32) (param $off i32) (param $v i32) ...)
//!
//!   ;; User workload imports:
//!   ;; The Rust-side library inside the module calls the dispatch
//!   ;; functions when reading/writing region bytes.
//! ```
//!
//! ## What this gives you
//!
//! - Working sets up to **N × 4 GiB** (default N=16 → 64 GiB).
//! - Same sandbox guarantees as M32: each pool is hardware-isolated by
//!   the engine's bounds checks; a buffer overflow in pool 3 cannot
//!   corrupt pool 7.
//! - Region/handle abstraction with generation-checked stale handle
//!   detection.
//! - Allocator choice per pool (Bump / Freelist / Slab — reusing the
//!   `tvm-core` types).
//!
//! ## What this doesn't have
//!
//! - **No spill to disk.** A pure-guest module cannot do I/O without
//!   host capabilities (WASI etc). Spill is appropriately layered on
//!   top of TVM by the embedder if needed; it's not part of TVM's core
//!   contract. See `docs/architecture.md` for the rationale.
//! - **No host-side observability.** Metrics live in pool 0 alongside
//!   the directory; the host just sees N wasm memories.
//! - **Fixed pool count at module compile time.** Can't grow the
//!   number of pools; can grow each pool's capacity up to its declared
//!   max (4 GiB).
//! - **Native multi-memory loads from Rust source aren't possible**
//!   today (toolchain gap). The dispatch path goes through generated
//!   WAT helper functions that ARE multi-memory-aware. Rust calls them
//!   via plain function calls.

pub use tvm_core::{
    AllocatorKind, BumpAllocator, FreelistAllocator, Handle, PlacementPolicy, RegionAllocator,
    RegionKind, Residency, Result, SlabAllocator, TvmError, TvmFacade, TvmSpill,
};

mod directory;
pub(crate) mod dispatch;
mod facade;
mod module;
mod multi;
pub mod wasi_spill;

pub use directory::{GuestDirectory, Pool};
pub use facade::{Dispatch, FnDispatch, GuestTvm};
pub use module::{tvm_guest_mm_module_template, ModuleParams};
pub use multi::{MultiGuestTvm, ShardId};
pub use wasi_spill::{emit_wasi_spill_helpers, tvm_guest_mm_module_with_wasi_spill};

/// Default number of pools per generated module. 64 × 4 GiB = 256 GiB
/// addressable working set per guest. Pages allocate lazily, so the
/// actual commitment is bounded by what the workload touches.
///
/// Bumped from 16 (64 GiB) once the BST dispatch made per-byte access
/// against many pools cheap. For workloads that need >256 GiB, raise
/// `n_pools` further at module-build time — the cap is module size,
/// not anything architectural.
pub const DEFAULT_POOL_COUNT: u32 = 64;
