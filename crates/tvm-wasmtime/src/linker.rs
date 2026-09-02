//! Wit-bindgen linker installers — RETIRED at ADR-0029 Phase 6.9
//! D2 Session 15b (2026-09-01).
//!
//! The 5 `add_*_to_linker` fns (`add_to_linker`,
//! `add_shared_to_linker`, `add_concurrent_to_linker`,
//! `add_per_actor_to_linker`, `add_to_linker_with`) that lived
//! here have been removed. Consumers migrate to the wasmos install
//! path in [`crate::wasmos_bindings`]:
//!
//! | Retired entry                   | Wasmos replacement                                    |
//! |---------------------------------|-------------------------------------------------------|
//! | `add_to_linker<T>`              | `install_tvm_imports_per_actor::<T>`                  |
//! | `add_shared_to_linker`          | `install_tvm_imports_shared`                          |
//! | `add_concurrent_to_linker`      | `install_tvm_imports_concurrent`                      |
//! | `add_per_actor_to_linker`       | `install_tvm_imports_per_actor_budget`                |
//! | `add_to_linker_with<T, H>`      | `TvmHostSource::PerActor` + `TvmHostExtractor` trait  |
//!
//! Each wasmos entry returns a
//! [`wasmos_runtime_api::HostImports`] composite the caller
//! installs into a `wasmtime::component::Linker<S>` via
//! `wasmos-runtime-wasmtime-v48::async_bridge::install_host_imports`.
//!
//! Downstream migration history:
//! - girder-wasmtime line 871: Session 14 (girder `af5def0`).
//! - sqlink-host (3 production + 6 tests): Session 15a (sqlink `474d4582`).
//! - tvm-wasm internal reference tests (component_host, shared_host,
//!   end_to_end_guest): Session 15b.
//!
//! This file stays as an anchor for the retirement history. The
//! `bindings.rs` module STAYS — its Host trait definitions and
//! record types are still consumed by the trait impls on
//! `TvmHost` / `SharedTvmHost` / `ConcurrentTvmHost` /
//! `PerActorTvmHost` (which the wasmos install path delegates
//! through) and by consumers with custom hosts.
