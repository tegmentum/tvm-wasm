//! External-tier hook: a callback the directory invokes when a region
//! marked `Residency::External` is touched. This is the minimum viable
//! implementation of design-doc §4.4 tier "External" — the runtime
//! delegates loading to caller-supplied code rather than a built-in
//! `BackingStore`.
//!
//! Use cases:
//!   - Network-attached / object-storage region fetches.
//!   - Sharing regions across processes or machines.
//!   - Lazy materialization from a generator.
//!
//! Contract: the closure receives `(region_id, generation)` and returns
//! `Vec<u8>` of the region's full contents, or an error. The directory
//! installs the bytes via `M::restore`, transitions to `Residency::Hot`,
//! and proceeds with the access that triggered the fault.

use crate::error::Result;

pub type ExternalLoader = Box<dyn Fn(u16, u16) -> Result<Vec<u8>> + Send + Sync>;
