//! `wasmtime::component::bindgen!` output for the `tvm-guest` world.
//!
//! # Role
//!
//! `pub mod bindings` is intentionally public. The bindgen-generated
//! `Host` traits (`tvm::memory::manager::Host`,
//! `tvm::memory::bytes::Host`, `tvm::memory::diagnostics::Host`,
//! `tvm::memory::types::Host`) + record / enum / variant types
//! (`Handle`, `RegionKind`, `Residency`, `RegionInfo`,
//! `RegionMetrics`, `CompactResult`, `TvmError`) are still consumed by
//! two categories of code:
//!
//!  1. **Internal Host trait impls** — [`crate::TvmHost`],
//!     [`crate::shared_host::SharedTvmHost`],
//!     [`crate::concurrent_host::ConcurrentTvmHost`], and
//!     [`crate::per_actor::PerActorTvmHost`] each `impl` the four
//!     Host traits, providing the business logic that the wasmos
//!     [`crate::wasmos_bindings`] handlers delegate through
//!     (their #[host_iface(sync)] bodies call
//!     `BgManagerHost::create_region(&mut host, ...)` etc).
//!  2. **External consumers with custom hosts** — sqlink-host's
//!     `host/src/wasmos_tvm.rs` (Session 15a) `use
//!     tvm_wasmtime::bindings::tvm::memory::{bytes,diagnostics,manager}::Host`
//!     to delegate through the same trait surface from a
//!     cross-major-version-pinned consumer. Any downstream that
//!     writes its own tvm host follows the same pattern.
//!
//! # Historical note
//!
//! The `add_*_to_linker` family that this bindgen! also emitted was
//! wrapped by tvm-wasmtime's `linker.rs` and re-exported at the crate
//! root. The 5 `add_*_to_linker` wrappers were RETIRED at ADR-0029
//! Phase 6.9 D2 Session 15b once every downstream consumer
//! (girder-wasmtime, sqlink-host, ducklink) migrated to the wasmos
//! install path in [`crate::wasmos_bindings`]. `linker.rs`'s module
//! docstring carries the retired-entry → wasmos-install-path
//! migration table for future reference. This bindings module STAYS
//! because it feeds the two roles above (Host trait impl surface),
//! independent of the retired wrapper family.

wasmtime::component::bindgen!({
    path: "../../wit",
    world: "tvm-guest",
});
