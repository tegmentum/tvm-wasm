//! Glob-importable common types for typical wasmtime + TVM setups.
//!
//! ```ignore
//! use tvm_wasmtime::prelude::*;
//!
//! let mut host = TvmHost::new();
//! let (region, handle) = host.alloc_in_new_region(RegionKind::HotHeap, 4096, 64)?;
//! host.write_bytes(handle, b"payload")?;
//! ```

pub use tvm_core::prelude::*;

pub use crate::builder::TvmBuilder;
pub use crate::concurrent_host::ConcurrentTvmHost;
pub use crate::host::TvmHost;
pub use crate::imported::{
    build_imported_setup, build_imported_setup_with_data, create_imported_in_store, ImportedRegion,
};
// Phase 6.9 D2 Session 15b: linker::add_*_to_linker RETIRED — see
// linker.rs docstring for the wasmos install-path replacements.
pub use crate::macros::tvm_mm_setup;
// Phase 6.9.d Session 7: deprecated at definition; see lib.rs for
// the rationale.
#[allow(deprecated)]
pub use crate::raw_linker::{add_raw_imports, add_raw_imports_with_memory_name};
pub use crate::shared_host::SharedTvmHost;
