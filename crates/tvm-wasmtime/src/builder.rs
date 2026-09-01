//! Fluent builder for typical TVM-on-wasmtime setups. Reduces the
//! 5–10-line "create engine, create host, register imports, instantiate
//! linker" boilerplate to a single chained call.
//!
//! ```ignore
//! use tvm_wasmtime::prelude::*;
//!
//! // The everyday setup: a host with default options + raw imports.
//! let (mut store, linker) = TvmBuilder::new()
//!     .with_raw_imports()
//!     .build()?;
//! ```

use std::path::PathBuf;

use tvm_core::{AllocatorKind, DynBackingStore};
use wasmtime::{Engine, Linker, Store};

use crate::TvmHost;

/// Fluent builder for `(Store<TvmHost>, Linker<TvmHost>)` pairs.
pub struct TvmBuilder {
    backing_path: Option<PathBuf>,
    custom_backing: Option<DynBackingStore>,
    default_allocator: AllocatorKind,
    multi_memory: bool,
    register_raw: bool,
    register_wit: bool,
}

impl Default for TvmBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl TvmBuilder {
    pub fn new() -> Self {
        Self {
            backing_path: None,
            custom_backing: None,
            default_allocator: AllocatorKind::Bump,
            multi_memory: false,
            register_raw: false,
            register_wit: false,
        }
    }

    /// Configure a file-backed spill store. Mutually exclusive with
    /// `with_custom_backing` — last call wins.
    pub fn with_backing(mut self, path: impl Into<PathBuf>) -> Self {
        self.backing_path = Some(path.into());
        self.custom_backing = None;
        self
    }

    /// Configure an arbitrary user-supplied backing store (S3, in-memory
    /// stub for tests, network-attached, etc.). The trait surface is
    /// just `spill` + `load`. Last call wins between this and
    /// `with_backing`.
    pub fn with_custom_backing<B: tvm_core::BackingStore + Send + 'static>(
        mut self,
        backing: B,
    ) -> Self {
        self.custom_backing = Some(Box::new(backing));
        self.backing_path = None;
        self
    }

    /// Pick the default allocator for new regions.
    pub fn with_allocator(mut self, allocator: AllocatorKind) -> Self {
        self.default_allocator = allocator;
        self
    }

    /// Enable multi-memory support on the engine. Required for
    /// `ImportedRegion` workloads.
    pub fn with_multi_memory(mut self) -> Self {
        self.multi_memory = true;
        self
    }

    /// Register the `tvm.*` raw fast-path imports in the linker.
    pub fn with_raw_imports(mut self) -> Self {
        self.register_raw = true;
        self
    }

    /// Register the WIT host trait impls in the linker. Requires the
    /// component-model feature on wasmtime; uses the `tvm-guest` world.
    pub fn with_wit_imports(mut self) -> Self {
        self.register_wit = true;
        self
    }

    /// Finalize: returns the configured `(Engine, Store, Linker)` tuple.
    /// The store data is a `TvmHost` initialized per the builder.
    ///
    /// Note: the `Linker` returned is the **core-wasm** linker. For the
    /// WIT/component-model path use [`build_component`] which returns a
    /// `wasmtime::component::Linker` instead.
    pub fn build(self) -> wasmtime::Result<(Engine, Store<TvmHost>, Linker<TvmHost>)> {
        let (engine, store, linker, _) = self.build_internal(false)?;
        Ok((engine, store, linker))
    }

    /// Same as [`build`] but returns a component-model `Linker` and (if
    /// `with_wit_imports` was called) registers the WIT host trait impls.
    /// Use this when the guest is a wasm component, not a core module.
    pub fn build_component(
        self,
    ) -> wasmtime::Result<(Engine, Store<TvmHost>, wasmtime::component::Linker<TvmHost>)> {
        let mut config = if self.multi_memory {
            crate::engine_config::imported_region_engine_config()
        } else {
            wasmtime::Config::new()
        };
        config.wasm_component_model(true);
        let engine = Engine::new(&config)?;
        let mut host = if let Some(b) = self.custom_backing {
            TvmHost::with_custom_backing(b)
        } else if let Some(p) = self.backing_path {
            TvmHost::with_backing(p)?
        } else {
            TvmHost::new()
        };
        host.default_allocator = self.default_allocator;
        let store = Store::new(&engine, host);
        let mut linker = wasmtime::component::Linker::<TvmHost>::new(&engine);
        if self.register_wit {
            // Phase 6.9 D2 Session 5: TvmBuilder's wit-bindgen
            // finisher is the sibling of the wasmos finisher
            // (`for_wasmos()`); the deprecation on add_to_linker
            // doesn't change what this method returns. Consumers
            // wanting to migrate off this whole builder branch
            // should use TvmBuilder::for_wasmos() (Phase 6.9.d
            // Session 1).
            #[allow(deprecated)]
            crate::linker::add_to_linker(&mut linker)?;
        }
        Ok((engine, store, linker))
    }

    /// ADR-0029 Phase 6.9.d — wasmos-flavored builder finisher.
    /// Returns a [`wasmos_runtime_api::CoreImports`] composite that
    /// carries the TVM raw fast-path imports as [`wasmos_runtime_api::
    /// CoreImportFn`] handlers, wired against a shared TVM host.
    ///
    /// The returned `CoreImports` contains **no wasmtime types** — it
    /// is the wasmtime-independent handoff surface for tvm-wasm.
    /// Consumers pass it to their chosen wasmos adapter (or a raw
    /// wasmtime engine via the wasmos adapter's core-imports install
    /// path when that lands) without ever naming `Engine`, `Store`,
    /// or `Linker` themselves.
    ///
    /// The `.with_raw_imports()` toggle from the wasmtime-typed
    /// [`Self::build`] flow does not apply here — this method
    /// always registers the raw imports (that's the point of the
    /// wasmos-flavored builder path). Backing store + allocator +
    /// multi-memory settings from the fluent builder are IGNORED
    /// because the wasmos path currently owns the SharedTvmHost
    /// construction internally; a `with_wasmos_backing` /
    /// `with_wasmos_allocator` follow-up threads those through when
    /// a consumer needs the customisation.
    ///
    /// The returned handle is intentionally NOT a `(Store, Linker)`
    /// pair like [`Self::build`] — the wasmos path pushes engine
    /// ownership onto the caller's adapter (or a future
    /// [`wasmos_runtime_api::Runtime`] instance), so returning
    /// wasmtime-owned state here would defeat the abstraction.
    pub fn for_wasmos(self) -> wasmos_runtime_api::CoreImports {
        let host = crate::shared_host::SharedTvmHost::new();
        crate::raw_linker_wasmos::add_raw_imports(
            wasmos_runtime_api::CoreImports::new(),
            host,
        )
    }

    /// ADR-0029 Phase 6.9 D2 Session 11 — wasmos-flavored
    /// component-model builder finisher. Peer of [`Self::for_wasmos`]
    /// for the component-model side of the surface. Returns a
    /// [`wasmos_runtime_api::HostImports`] composite carrying all
    /// three `tvm:memory@0.1.0` interfaces (`manager` / `bytes` /
    /// `diagnostics`) as `#[host_iface(sync)]` handlers wired
    /// against a shared TVM host — the wasmos equivalent of what
    /// [`Self::build_component`] + [`crate::linker::add_to_linker`]
    /// produced together in the wit-bindgen path.
    ///
    /// The tuple's [`crate::shared_host::SharedTvmHost`] is
    /// returned alongside the imports so the caller can
    /// hold a second clone-handle for host-side operations
    /// (spill, load, direct region inspection) while the
    /// installed `HostImports` handle stays owned by the
    /// wasmos [`wasmos_runtime_api::ExecutionContext`].
    ///
    /// The `.with_wit_imports()` toggle from the wasmtime-typed
    /// [`Self::build_component`] flow does not apply here — this
    /// method always installs all three interfaces (matching the
    /// wit-bindgen "install everything the WIT world declares"
    /// behavior). Backing store + allocator + multi-memory
    /// settings are IGNORED for the same reason [`Self::for_wasmos`]
    /// ignores them; a `with_wasmos_backing` follow-up threads
    /// them through when a consumer needs customisation.
    ///
    /// Typical composition with [`Self::for_wasmos`] for a
    /// consumer that wants both raw + WIT paths:
    ///
    /// ```rust,ignore
    /// let core_imports = TvmBuilder::new().for_wasmos();
    /// let (host, host_imports) = TvmBuilder::new().for_wasmos_component();
    /// let ctx = ExecutionContext::new()
    ///     .with_core_imports(core_imports)
    ///     .with_host_imports(host_imports);
    /// ```
    ///
    /// Both call sites use fresh `TvmBuilder::new()` because the
    /// builder is consumed by these finishers; a follow-up
    /// `for_wasmos_full()` returning both composites off a single
    /// builder pass is a natural next addition.
    pub fn for_wasmos_component(
        self,
    ) -> (
        crate::shared_host::SharedTvmHost,
        wasmos_runtime_api::HostImports,
    ) {
        let host = crate::shared_host::SharedTvmHost::new();
        let imports = crate::wasmos_bindings::install_tvm_imports_shared(
            wasmos_runtime_api::HostImports::new(),
            host.clone(),
        );
        (host, imports)
    }

    /// D2 Session 11 — one-shot finisher that returns everything a
    /// consumer needs for the wasmos install path: the raw-tvm
    /// `CoreImports` + the component-model `HostImports`, both
    /// sharing the same [`SharedTvmHost`] so a guest that uses
    /// both surfaces sees consistent state.
    ///
    /// This is the ergonomics gap called out in the Phase 6.9
    /// tvm-wasm recon's "D2 core work complete — remaining is
    /// decision-gated" section. Consumers who want the full wasmos
    /// composite in one call use this; consumers who need only one
    /// side use [`Self::for_wasmos`] or [`Self::for_wasmos_component`]
    /// independently.
    pub fn for_wasmos_full(
        self,
    ) -> (
        crate::shared_host::SharedTvmHost,
        wasmos_runtime_api::CoreImports,
        wasmos_runtime_api::HostImports,
    ) {
        let host = crate::shared_host::SharedTvmHost::new();
        let core = crate::raw_linker_wasmos::add_raw_imports(
            wasmos_runtime_api::CoreImports::new(),
            host.clone(),
        );
        let host_imports = crate::wasmos_bindings::install_tvm_imports_shared(
            wasmos_runtime_api::HostImports::new(),
            host.clone(),
        );
        (host, core, host_imports)
    }

    fn build_internal(
        self,
        component_only: bool,
    ) -> wasmtime::Result<(Engine, Store<TvmHost>, Linker<TvmHost>, bool)> {
        let mut config = if self.multi_memory {
            crate::engine_config::imported_region_engine_config()
        } else {
            wasmtime::Config::new()
        };
        if component_only {
            config.wasm_component_model(true);
        }
        let engine = Engine::new(&config)?;
        let mut host = if let Some(b) = self.custom_backing {
            TvmHost::with_custom_backing(b)
        } else if let Some(p) = self.backing_path {
            TvmHost::with_backing(p)?
        } else {
            TvmHost::new()
        };
        host.default_allocator = self.default_allocator;
        let store = Store::new(&engine, host);
        let mut linker = Linker::<TvmHost>::new(&engine);
        if self.register_raw {
            // Deprecated (Phase 6.9.d Session 7) — TvmBuilder's
            // wit-bindgen-linker finisher is the sibling of the wasmos
            // finisher (`for_wasmos()`); the deprecation on
            // `add_raw_imports` doesn't change what this method returns
            // (still `(Engine, Store, Linker)`). Consumers wanting to
            // migrate off this whole builder branch should use
            // `TvmBuilder::for_wasmos()` from Phase 6.9.d Session 1.
            #[allow(deprecated)]
            crate::raw_linker::add_raw_imports(&mut linker)?;
        }
        Ok((engine, store, linker, self.register_wit))
    }
}

#[cfg(test)]
mod wasmos_builder_tests {
    use super::*;

    #[test]
    fn for_wasmos_component_returns_host_and_imports_matched_by_shared_arc() {
        let (host, imports) = TvmBuilder::new().for_wasmos_component();
        // The returned host is a fresh SharedTvmHost; the imports
        // hold their own Arc-clone(s) captured at install time.
        // Length checks the composite carries all three interfaces
        // (manager / bytes / diagnostics) — matches the
        // install_tvm_imports_shared behavior.
        assert_eq!(imports.len(), 3);
        assert!(imports.get("tvm:memory/manager@0.1.0").is_some());
        assert!(imports.get("tvm:memory/bytes@0.1.0").is_some());
        assert!(imports.get("tvm:memory/diagnostics@0.1.0").is_some());
        // The host is a live SharedTvmHost — cloning still yields a
        // shared handle onto the same TvmHost.
        let _clone = host.clone();
    }

    #[test]
    fn for_wasmos_full_returns_shared_host_bound_across_core_and_host_imports() {
        let (host, core, host_imports) = TvmBuilder::new().for_wasmos_full();
        // Raw path: 26 tvm.* handlers.
        assert_eq!(core.len(), 26);
        // Component-model path: 3 wit-bindgen-tvm interfaces.
        assert_eq!(host_imports.len(), 3);
        // The host is the returned shared handle; consumers use it
        // for host-side operations (spill, load, direct inspection)
        // while both composites drive guest-facing dispatch.
        let mut g = host.lock();
        // Sanity poke — the shared host works.
        assert_eq!(g.default_allocator, tvm_core::AllocatorKind::Bump);
        g.default_allocator = tvm_core::AllocatorKind::Freelist;
        drop(g);
        // A fresh lock sees the mutation — proves this is one shared
        // TvmHost, not per-composite clones.
        assert_eq!(host.lock().default_allocator, tvm_core::AllocatorKind::Freelist);
    }

    #[test]
    fn for_wasmos_component_carries_bytes_and_diagnostics_interface_names_correctly() {
        // Regression guard for the interface-name discipline (ducklink
        // Phase 6.2.h.8 caught: version tag + kebab-case matter).
        // If someone silently drops @0.1.0 or camelCases an arm, this
        // fails at test time not integration time.
        let (_, imports) = TvmBuilder::new().for_wasmos_component();
        for iface in [
            "tvm:memory/manager@0.1.0",
            "tvm:memory/bytes@0.1.0",
            "tvm:memory/diagnostics@0.1.0",
        ] {
            assert!(
                imports.get(iface).is_some(),
                "expected interface {iface} in composite"
            );
        }
    }
}
