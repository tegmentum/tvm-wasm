//! Common types for `tvm-core` consumers. Glob-import as
//! `use tvm_core::prelude::*;` to bring the everyday names into scope.

pub use crate::allocator::AllocatorKind;
pub use crate::backing::{BackingStore, FileBackingStore, VecBackedRegion};
pub use crate::error::{
    set_last_error_context, take_last_error_context, ErrorContext, Result, TvmError,
};
pub use crate::handle::Handle;
pub use crate::region::{Region, RegionKind};
pub use crate::residency::Residency;
pub use crate::policy::PlacementPolicy;
