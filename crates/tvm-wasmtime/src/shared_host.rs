//! `SharedTvmHost` — a thread-safe wrapper around `TvmHost` for multi-store /
//! multi-threaded use.
//!
//! ## Typical pattern
//!
//! ```ignore
//! let shared = SharedTvmHost::new();
//! let mut linker: Linker<SharedTvmHost> = Linker::new(&engine);
//! add_shared_to_linker(&mut linker)?;
//!
//! // Two wasmtime Stores, two threads, one shared region directory.
//! let h1 = std::thread::spawn({
//!     let shared = shared.clone();
//!     move || {
//!         let mut store = Store::new(&engine, shared);
//!         /* call into guest */
//!     }
//! });
//! let h2 = std::thread::spawn({
//!     let shared = shared.clone();
//!     move || {
//!         let mut store = Store::new(&engine, shared);
//!         /* call into guest */
//!     }
//! });
//! ```
//!
//! ## Lock granularity
//!
//! `SharedTvmHost` currently serializes all host-trait calls through a single
//! `Mutex<TvmHost>`. This is correct under any mix of concurrent calls but
//! means two regions can't be operated on in parallel from different stores.
//! For low-contention workloads (most TVM use cases — most calls touch
//! distinct stores' guests, not the same regions), the simplicity wins.
//!
//! Per-region locking is a deliberate future-work item; it will require
//! restructuring `RegionDirectory` itself, since several methods currently
//! take `&mut self` and reach into `self.regions` directly.

use std::sync::{Arc, Mutex, MutexGuard};

use tvm_core::{AllocatorKind, TvmError as CoreError};

use crate::bindings::tvm::memory::bytes::Host as BytesHost;
use crate::bindings::tvm::memory::diagnostics::Host as DiagnosticsHost;
use crate::bindings::tvm::memory::manager::Host as ManagerHost;
use crate::bindings::tvm::memory::types::{
    CompactResult, Handle, Host as TypesHost, RegionInfo, RegionKind, RegionMetrics, TvmError,
};
use crate::host::TvmHost;

// NOTE on per-thread cache: a thread-local `ResolveCache` would let
// lookups skip the inner mutex on the hot path. We considered it but
// chose not to ship it for the following reasons:
//
//   1. **Pointer-caching variant is unsafe**: a thread-local slot
//      caching the region's data pointer can dangle if another thread
//      spills/destroys the region. Fixing that needs Arc-wrapped region
//      memory or epoch-based reclamation (e.g. crossbeam-epoch). Both
//      are real refactors with their own perf cost.
//
//   2. **Validation-only variant saves ~10–15 ns**: caching just
//      `(generation, capacity)` lets us skip directory re-validation
//      on hit, but the mutex is still taken on every access. Marginal
//      gain that doesn't justify the per-thread machinery.
//
// The right answer for high-contention workloads is `ConcurrentTvmHost`,
// which uses per-region locking so different regions never serialize on
// each other. `SharedTvmHost` stays simple and correct for the
// low-contention case.

#[derive(Clone)]
pub struct SharedTvmHost {
    inner: Arc<Mutex<TvmHost>>,
}

impl Default for SharedTvmHost {
    fn default() -> Self {
        Self::new()
    }
}

impl SharedTvmHost {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(TvmHost::new())),
        }
    }

    pub fn from_host(host: TvmHost) -> Self {
        Self {
            inner: Arc::new(Mutex::new(host)),
        }
    }

    pub fn with_backing(path: impl AsRef<std::path::Path>) -> Result<Self, CoreError> {
        Ok(Self::from_host(TvmHost::with_backing(path)?))
    }

    pub fn with_allocator(self, allocator: AllocatorKind) -> Self {
        self.lock().default_allocator = allocator;
        self
    }

    pub fn lock(&self) -> MutexGuard<'_, TvmHost> {
        // Poisoning is treated as a programmer error: another thread paniced
        // while holding the lock. Recover by re-grabbing the inner data —
        // the directory's internal invariants are atomic per call.
        match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        }
    }
}

impl AsMut<SharedTvmHost> for SharedTvmHost {
    fn as_mut(&mut self) -> &mut SharedTvmHost {
        self
    }
}

impl TypesHost for SharedTvmHost {}

// `SharedTvmHost` is a strict locking wrapper: every Host method just
// `lock + delegate to inner TvmHost`. The macro below collapses what
// would otherwise be ~100 lines of mechanical forwarding into a
// declarative table.

macro_rules! forward_lock {
    (
        $trait:ident for $ty:ty {
            $(
                fn $name:ident ( $($arg:ident : $argty:ty),* $(,)? ) -> $ret:ty;
            )*
        }
    ) => {
        impl $trait for $ty {
            $(
                fn $name(&mut self, $($arg: $argty),*) -> $ret {
                    $trait::$name(&mut *self.lock(), $($arg),*)
                }
            )*
        }
    };
}

forward_lock! {
    ManagerHost for SharedTvmHost {
        fn create_region(kind: RegionKind, capacity: u32) -> Result<u16, TvmError>;
        fn destroy_region(region_id: u16) -> Result<(), TvmError>;
        fn alloc(region_id: u16, size: u32) -> Result<Handle, TvmError>;
        fn dealloc(ptr: Handle) -> Result<(), TvmError>;
        fn describe_region(region_id: u16) -> Result<RegionInfo, TvmError>;
        fn promote_region(region_id: u16) -> Result<(), TvmError>;
        fn demote_region(region_id: u16) -> Result<(), TvmError>;
        fn spill_region(region_id: u16) -> Result<(), TvmError>;
        fn load_region(region_id: u16) -> Result<(), TvmError>;
        fn pin(region_id: u16) -> Result<(), TvmError>;
        fn unpin(region_id: u16) -> Result<(), TvmError>;
        fn compact_region(region_id: u16) -> Result<CompactResult, TvmError>;
    }
}

forward_lock! {
    BytesHost for SharedTvmHost {
        fn read(ptr: Handle, len: u32) -> Result<Vec<u8>, TvmError>;
        fn write(ptr: Handle, data: Vec<u8>) -> Result<(), TvmError>;
        fn copy(src: Handle, dst: Handle, len: u32) -> Result<(), TvmError>;
        fn read_into(src: Handle, dst_region: u16, dst_offset: u32, len: u32) -> Result<(), TvmError>;
        fn write_from(src_region: u16, src_offset: u32, dst: Handle, len: u32) -> Result<(), TvmError>;
        fn copy_region(src_region: u16, src_offset: u32, dst_region: u16, dst_offset: u32, len: u32) -> Result<(), TvmError>;
    }
}

forward_lock! {
    DiagnosticsHost for SharedTvmHost {
        fn list_regions() -> Vec<RegionInfo>;
        fn fault_count(region_id: u16) -> u64;
        fn allocation_count(region_id: u16) -> u64;
        fn bytes_read_count(region_id: u16) -> u64;
        fn bytes_written_count(region_id: u16) -> u64;
        fn metrics_snapshot(region_id: u16) -> Result<RegionMetrics, TvmError>;
    }
}
