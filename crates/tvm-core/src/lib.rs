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

pub mod allocator;
pub mod async_backing;
pub mod backing;
pub mod cache;
pub mod concurrent;
pub mod prelude;
pub mod debug;
pub mod directory;
mod directory_slices;
pub mod error;
pub mod external;
pub mod handle;
pub mod metrics;
pub mod policy;
pub mod region;
pub mod residency;
pub mod shared;

pub use allocator::{
    AllocatorKind, BumpAllocator, FreelistAllocator, RegionAllocator, SlabAllocator,
};
pub use async_backing::{AsyncBackingStore, SyncAdapter};
pub use backing::{
    BackingStore, DynBackingStore, FileBackingStore, SingleFileBackingStore, VecBackedRegion,
};
pub use directory::{HandleRemap, MemoryRegion, RegionDirectory, RegionEntry};
pub use error::{
    set_last_error_context, take_last_error_context, ErrorContext, Result, TvmError,
};
pub use handle::Handle;
pub use metrics::{MetricsSnapshot, RegionMetrics};
pub use region::{Region, RegionKind};
pub use residency::Residency;
pub use debug::{dump_region_layout, fault_counts, validate_handle, validate_handles, HandleStatus};
pub use cache::{FastHit, ResolveCache, ResolveHit};
pub use concurrent::ConcurrentDirectory;
pub use policy::PlacementPolicy;
pub use shared::SharedDirectory;
