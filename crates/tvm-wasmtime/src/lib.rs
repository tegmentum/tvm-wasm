//! `tvm-wasmtime` — wasmtime integration for `tvm-core`.
//!
//! ## Quick start
//!
//! ```ignore
//! use tvm_wasmtime::prelude::*;
//!
//! let (engine, mut store, linker) = TvmBuilder::new()
//!     .with_raw_imports()
//!     .build()?;
//!
//! let (region, handle) = store.data_mut()
//!     .alloc_in_new_region(RegionKind::HotHeap, 4096, 64)?;
//! store.data_mut().write_bytes(handle, b"hello")?;
//! ```
//!
//! ## Two access paths
//!
//! 1. **WIT (component model)** via [`add_to_linker`] / [`add_shared_to_linker`].
//!    Type-safe, multi-language portable. Used for setup and any call where
//!    the canonical-ABI cost is amortized.
//! 2. **Raw imports** via [`add_raw_imports`]. Plain `(i32 | i64) → i32 | i64`
//!    core-wasm calls; one host scratch copy per byte transfer. Use for hot
//!    inner loops. See `docs/fast-paths.md` for the full cost analysis.
//!
//! ## Which host type?
//!
//! Three host wrappers exist. Pick by your concurrency need, **not** by
//! whether you've already started writing code with one:
//!
//! | Type | Use when | Cost |
//! |---|---|---|
//! | [`TvmHost`] | Single store, single thread. The default. | Lowest — no locks. |
//! | [`SharedTvmHost`] | Multiple wasmtime stores, low contention. Mutations through any clone visible to all. | Single global mutex serializes calls. |
//! | [`ConcurrentTvmHost`] | Multiple stores, **frequent concurrent ops on different regions**. | Per-region locks; ops on distinct regions don't block each other. |
//!
//! Default to `TvmHost`. Migrate up only when threading actually demands
//! it. See `docs/architecture.md` "Multi-store sharing" for the full
//! comparison.

pub mod bindings;
pub mod builder;
pub mod concurrent_host;
pub mod engine_config;
pub mod host;
pub mod imported;
pub mod linker;
pub mod macros;
pub mod memory_factory;
pub mod per_actor;
pub mod prelude;
pub mod raw_linker;
pub mod raw_linker_wasmos;
pub mod region_view;
pub mod runtime_persist;
pub mod shared_host;

pub use builder::TvmBuilder;
pub use concurrent_host::ConcurrentTvmHost;
pub use engine_config::{imported_region_engine_config, pooling_imported_region_engine_config};
pub use host::TvmHost;
pub use imported::{
    build_imported_setup, build_imported_setup_with_data, create_imported_in_store, ImportedRegion,
};
pub use linker::{
    add_concurrent_to_linker, add_per_actor_to_linker, add_shared_to_linker, add_to_linker,
    add_to_linker_with,
};
pub use memory_factory::{RuntimeMemoryRegion, WasmtimeMemoryRegion, WASM_PAGE_SIZE};
pub use per_actor::{PerActorTvmHost, TvmBudget};
// ADR-0029 Phase 6.9.d Session 7: the wit-bindgen raw entry points
// are marked `#[deprecated]` at their definitions; #[allow(deprecated)]
// here so re-export doesn't emit an in-crate warning. The deprecation
// still fires at every consumer call site.
#[allow(deprecated)]
pub use raw_linker::{
    add_raw_imports, add_raw_imports_with_memory_name, add_raw_shared,
    add_raw_shared_with_memory_name,
};
pub use region_view::RegionView;
pub use runtime_persist::{load_runtime_region, spill_runtime_region};
pub use shared_host::SharedTvmHost;
