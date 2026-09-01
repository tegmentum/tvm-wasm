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
