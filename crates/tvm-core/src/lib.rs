//! `tvm-core` — substrate for tiered virtual memory.
//!
//! A [`RegionDirectory`] owns a set of typed memory regions, each addressed
//! through generation-checked [`Handle`]s rather than raw pointers. Regions
//! can be promoted, demoted, spilled, loaded, and compacted; allocators are
//! pluggable per region; metrics and a small resolve cache support
//! observability and hot-path optimization.
//!
//! See `docs/architecture.md` in the repository root for the full picture
//! of what this crate provides and how it composes with `tvm-wasmtime`.
//!
//! ## Quick example
//!
//! ```
//! use tvm_core::{RegionDirectory, RegionKind, VecBackedRegion};
//!
//! let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
//! let region = dir
//!     .create_region(RegionKind::HotHeap, 1024, VecBackedRegion::new(1024))
//!     .unwrap();
//! let handle = dir.alloc(region, 32).unwrap();
//! dir.write(handle, b"32 bytes of data exactly!!!!!!!!").unwrap();
//! let mut buf = [0u8; 32];
//! dir.read(handle, &mut buf).unwrap();
//! ```
//!
//! ## std vs no_std
//!
//! With the default `std` feature on, the full crate surface is
//! available including `ConcurrentDirectory`, `BackingStore` file
//! impls, the WAT prelude, etc.
//!
//! With `default-features = false`, only the pure-data +
//! single-threaded subset compiles: types ([`Region`], [`Handle`],
//! [`Residency`], [`RegionKind`], [`PlacementPolicy`],
//! [`AllocatorKind`], [`TvmError`], [`EvictionPolicy`]), single-
//! threaded structures ([`RegionDirectory`], [`BumpAllocator`],
//! [`FreelistAllocator`], [`SlabAllocator`]), and the
//! [`MemoryRegion`] trait. Suitable for `wasm32-unknown-unknown`
//! guests that need to mirror host-side concepts without pulling
//! in libstd.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod allocator;
#[cfg(feature = "std")]
pub mod async_backing;
#[cfg(feature = "std")]
pub mod backing;
pub mod cache;
#[cfg(feature = "std")]
pub mod concurrent;
#[cfg(feature = "std")]
pub mod debug;
#[cfg(feature = "std")]
pub mod directory;
#[cfg(feature = "std")]
mod directory_slices;
pub mod error;
pub mod eviction;
#[cfg(feature = "std")]
pub mod external;
// `facade` is pure traits (TvmFacade, TvmSpill) — no std deps. It
// belongs in the no_std subset.
pub mod facade;
pub mod handle;
pub mod memory_region;
pub mod metrics;
pub mod policy;
#[cfg(feature = "std")]
pub mod prelude;
pub mod region;
pub mod residency;
#[cfg(feature = "std")]
pub mod shared;

pub use allocator::{
    AllocatorKind, BumpAllocator, FreelistAllocator, RegionAllocator, SlabAllocator,
};
#[cfg(feature = "std")]
pub use async_backing::{AsyncBackingStore, SyncAdapter};
#[cfg(feature = "std")]
pub use backing::{
    BackingStore, DynBackingStore, FileBackingStore, SingleFileBackingStore, VecBackedRegion,
};
pub use cache::{FastHit, ResolveCache, ResolveHit};
#[cfg(feature = "std")]
pub use concurrent::ConcurrentDirectory;
#[cfg(feature = "std")]
pub use debug::{
    dump_region_layout, fault_counts, validate_handle, validate_handles, HandleStatus,
};
// MemoryRegion + HandleRemap live in memory_region.rs (no_std);
// RegionDirectory + RegionEntry require std (BackingStore-using
// methods, file IO). All four were originally in directory.rs;
// the split happened in U2 of PLAN-tvm-convergence.md.
#[cfg(feature = "std")]
pub use directory::{RegionDirectory, RegionEntry};
#[cfg(feature = "std")]
pub use error::{set_last_error_context, take_last_error_context, ErrorContext};
pub use error::{Result, TvmError};
pub use eviction::{EvictionPolicy, EvictionReport, WithinTier};
pub use facade::{TvmFacade, TvmSpill};
pub use handle::Handle;
pub use memory_region::{HandleRemap, MemoryRegion};
pub use metrics::{MetricsSnapshot, RegionMetrics};
pub use policy::PlacementPolicy;
pub use region::{Region, RegionKind};
pub use residency::Residency;
#[cfg(feature = "std")]
pub use shared::SharedDirectory;
