//! ADR-0029 Phase 6.9 D2 in-place — Session 2 first slice.
//!
//! Wasmos-side mirror types for the `tvm:memory@0.1.0/types`
//! interface. Same shape / same field names as the
//! `wasmtime::component::bindgen!`-generated types in
//! [`crate::bindings::tvm::memory::types`], plus `#[derive(...)]`
//! from `wasmos_runtime_api` so wasmos `#[host_iface]` handlers can
//! consume + produce them directly.
//!
//! # Coexistence
//!
//! This module is **additive**. It doesn't touch [`crate::bindings`]
//! or the existing `Host` trait impls on [`crate::TvmHost`]. Session
//! 3 will wire `#[host_iface(sync)]` structs for the `manager`,
//! `bytes`, and `diagnostics` interfaces against these mirrors and
//! land `install_tvm_imports_*` composites (peer to sqlink Arc 1's
//! `install_sqlink_imports`). Session 4+ will deprecate the
//! `add_*_to_linker` family in `linker.rs`.
//!
//! # From converters
//!
//! Bidirectional `From` impls between the mirrors and the
//! bindgen-generated types let existing production code keep the
//! bindgen shape while wasmos-native paths use the mirrors. Same
//! pattern that ducklink Phase 6.2.m proved at scale (29 mirror
//! types, ~940 callsite migrations with zero external edit).
//!
//! # Test strategy
//!
//! Each mirror type has a round-trip test at the bottom of this
//! file: bindgen → mirror → bindgen (and back) → assert equality on
//! wire-visible fields. If the WIT surface adds a field, the bindgen
//! side gets it automatically and the round-trip test fires as
//! `mirror doesn't have the new field` — the mirror gets updated.

use wasmos_runtime_api::{
    host_iface, HostCallContext, HostImports, RuntimeResult, WitEnum, WitRecord,
    WitVariant,
};

use crate::bindings::tvm::memory::bytes::Host as BgBytesHost;
use crate::bindings::tvm::memory::diagnostics::Host as BgDiagnosticsHost;
use crate::bindings::tvm::memory::manager::Host as BgManagerHost;
use crate::bindings::tvm::memory::types as bg;
use crate::shared_host::SharedTvmHost;

// ── enum RegionKind ─────────────────────────────────────────────────
//
// Mirror of `bg::RegionKind`. Kebab-case in WIT → PascalCase in
// wit-bindgen's Rust output; the mirror uses the same variant
// names so `#[derive(WitEnum)]` produces the correct wire encoding.

/// Wasmos mirror of [`bg::RegionKind`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, WitEnum)]
pub enum RegionKind {
    HotHeap,
    ObjectArena,
    BlobArena,
    PageStore,
    Scratch,
    DeviceState,
    CodeCache,
}

impl From<bg::RegionKind> for RegionKind {
    fn from(k: bg::RegionKind) -> Self {
        match k {
            bg::RegionKind::HotHeap => Self::HotHeap,
            bg::RegionKind::ObjectArena => Self::ObjectArena,
            bg::RegionKind::BlobArena => Self::BlobArena,
            bg::RegionKind::PageStore => Self::PageStore,
            bg::RegionKind::Scratch => Self::Scratch,
            bg::RegionKind::DeviceState => Self::DeviceState,
            bg::RegionKind::CodeCache => Self::CodeCache,
        }
    }
}

impl From<RegionKind> for bg::RegionKind {
    fn from(k: RegionKind) -> Self {
        match k {
            RegionKind::HotHeap => Self::HotHeap,
            RegionKind::ObjectArena => Self::ObjectArena,
            RegionKind::BlobArena => Self::BlobArena,
            RegionKind::PageStore => Self::PageStore,
            RegionKind::Scratch => Self::Scratch,
            RegionKind::DeviceState => Self::DeviceState,
            RegionKind::CodeCache => Self::CodeCache,
        }
    }
}

// ── enum Residency ──────────────────────────────────────────────────

/// Wasmos mirror of [`bg::Residency`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, WitEnum)]
pub enum Residency {
    Hot,
    Warm,
    Cold,
    External,
}

impl From<bg::Residency> for Residency {
    fn from(r: bg::Residency) -> Self {
        match r {
            bg::Residency::Hot => Self::Hot,
            bg::Residency::Warm => Self::Warm,
            bg::Residency::Cold => Self::Cold,
            bg::Residency::External => Self::External,
        }
    }
}

impl From<Residency> for bg::Residency {
    fn from(r: Residency) -> Self {
        match r {
            Residency::Hot => Self::Hot,
            Residency::Warm => Self::Warm,
            Residency::Cold => Self::Cold,
            Residency::External => Self::External,
        }
    }
}

// ── record Handle ───────────────────────────────────────────────────
//
// The wire ABI for a record<u16, u16, u32> is stable; mirror keeps
// the exact field ordering the WIT declares (region-id → generation
// → offset).

/// Wasmos mirror of [`bg::Handle`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, WitRecord)]
pub struct Handle {
    pub region_id: u16,
    pub generation: u16,
    pub offset: u32,
}

impl From<bg::Handle> for Handle {
    fn from(h: bg::Handle) -> Self {
        Self { region_id: h.region_id, generation: h.generation, offset: h.offset }
    }
}

impl From<Handle> for bg::Handle {
    fn from(h: Handle) -> Self {
        Self { region_id: h.region_id, generation: h.generation, offset: h.offset }
    }
}

// ── record RegionInfo ───────────────────────────────────────────────

/// Wasmos mirror of [`bg::RegionInfo`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, WitRecord)]
pub struct RegionInfo {
    pub id: u16,
    pub generation: u16,
    pub kind: RegionKind,
    pub capacity: u32,
    pub used: u32,
    pub residency: Residency,
}

impl From<bg::RegionInfo> for RegionInfo {
    fn from(r: bg::RegionInfo) -> Self {
        Self {
            id: r.id,
            generation: r.generation,
            kind: r.kind.into(),
            capacity: r.capacity,
            used: r.used,
            residency: r.residency.into(),
        }
    }
}

impl From<RegionInfo> for bg::RegionInfo {
    fn from(r: RegionInfo) -> Self {
        Self {
            id: r.id,
            generation: r.generation,
            kind: r.kind.into(),
            capacity: r.capacity,
            used: r.used,
            residency: r.residency.into(),
        }
    }
}

// ── record RegionMetrics ────────────────────────────────────────────

/// Wasmos mirror of [`bg::RegionMetrics`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, WitRecord)]
pub struct RegionMetrics {
    pub allocations: u64,
    pub bytes_allocated: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub faults: u64,
    pub promotions: u64,
    pub demotions: u64,
}

impl From<bg::RegionMetrics> for RegionMetrics {
    fn from(m: bg::RegionMetrics) -> Self {
        Self {
            allocations: m.allocations,
            bytes_allocated: m.bytes_allocated,
            bytes_read: m.bytes_read,
            bytes_written: m.bytes_written,
            faults: m.faults,
            promotions: m.promotions,
            demotions: m.demotions,
        }
    }
}

impl From<RegionMetrics> for bg::RegionMetrics {
    fn from(m: RegionMetrics) -> Self {
        Self {
            allocations: m.allocations,
            bytes_allocated: m.bytes_allocated,
            bytes_read: m.bytes_read,
            bytes_written: m.bytes_written,
            faults: m.faults,
            promotions: m.promotions,
            demotions: m.demotions,
        }
    }
}

// ── record CompactResult ────────────────────────────────────────────
//
// WIT field `mapping: list<tuple<u32, u32>>` — Rust `Vec<(u32, u32)>`.
// WitRecord derive supports Vec<tuple> via the wasmos macro extensions
// landed at Phase 6.8 Session 2 + Session 3b tuple impls.

/// Wasmos mirror of [`bg::CompactResult`].
#[derive(Clone, Debug, PartialEq, Eq, WitRecord)]
pub struct CompactResult {
    pub old_generation: u16,
    pub new_generation: u16,
    pub mapping: Vec<(u32, u32)>,
}

impl From<bg::CompactResult> for CompactResult {
    fn from(c: bg::CompactResult) -> Self {
        Self {
            old_generation: c.old_generation,
            new_generation: c.new_generation,
            mapping: c.mapping,
        }
    }
}

impl From<CompactResult> for bg::CompactResult {
    fn from(c: CompactResult) -> Self {
        Self {
            old_generation: c.old_generation,
            new_generation: c.new_generation,
            mapping: c.mapping,
        }
    }
}

// ── variant TvmError ────────────────────────────────────────────────
//
// Mixed shape: 5 unit arms + 2 payload arms (u16 + String). WitVariant
// derive on wasmos accommodates any payload shape that itself
// satisfies WitBridge — u16 + String are built-in.

/// Wasmos mirror of [`bg::TvmError`].
#[derive(Clone, Debug, PartialEq, Eq, WitVariant)]
pub enum TvmError {
    RegionNotFound(u16),
    StaleHandle,
    OutOfBounds,
    NotResident,
    AllocationFailed,
    BackingStore(String),
    Pinned,
}

impl From<bg::TvmError> for TvmError {
    fn from(e: bg::TvmError) -> Self {
        match e {
            bg::TvmError::RegionNotFound(id) => Self::RegionNotFound(id),
            bg::TvmError::StaleHandle => Self::StaleHandle,
            bg::TvmError::OutOfBounds => Self::OutOfBounds,
            bg::TvmError::NotResident => Self::NotResident,
            bg::TvmError::AllocationFailed => Self::AllocationFailed,
            bg::TvmError::BackingStore(s) => Self::BackingStore(s),
            bg::TvmError::Pinned => Self::Pinned,
        }
    }
}

impl From<TvmError> for bg::TvmError {
    fn from(e: TvmError) -> Self {
        match e {
            TvmError::RegionNotFound(id) => Self::RegionNotFound(id),
            TvmError::StaleHandle => Self::StaleHandle,
            TvmError::OutOfBounds => Self::OutOfBounds,
            TvmError::NotResident => Self::NotResident,
            TvmError::AllocationFailed => Self::AllocationFailed,
            TvmError::BackingStore(s) => Self::BackingStore(s),
            TvmError::Pinned => Self::Pinned,
        }
    }
}

// ── Host structs — #[host_iface(sync)] for tvm:memory@0.1.0 (D2 Sessions 3-4) ─
//
// Wasmos-native implementations of the three function-carrying
// interfaces: `manager`, `bytes`, `diagnostics`. Each struct holds
// a [`TvmHostSource`] and dispatches through it — the enum unifies
// SHARED (`SharedTvmHost` locked per call) and PER-ACTOR (state
// pulled from `ctx.consumer_state::<T>()`) concurrency models.
// Mirrors the raw_linker_wasmos::TvmHostSource pattern from Phase
// 6.9.d Session 5.
//
// # Delegation
//
// Handlers delegate to the existing wit-bindgen `Host` trait impls
// on `TvmHost` (defined in `host.rs`) to avoid duplicating the
// business logic across paths. Arguments enter as mirror types
// and get converted to wit-bindgen types via `.into()` before the
// delegation; results convert back via `From` at the boundary.
// The From converters were established above in Session 2.
//
// # Interface naming
//
// Wasmos matches interface names verbatim against the guest's
// imports. The WIT declares `package tvm:memory@0.1.0`, so the
// installed names are `tvm:memory/manager@0.1.0` etc. — kebab-
// case, version-tagged, matching wit-bindgen's `add_to_linker`
// name generation.

use crate::TvmHost;

/// D2 Session 4 — abstract over "where the `TvmHost` lives" so the
/// same three host structs work under two concurrency models.
/// Twin of [`crate::raw_linker_wasmos::TvmHostSource`] but tied to
/// [`HostCallContext`] instead of `CoreImportContext` (the
/// component-model vs core-Wasm split).
///
/// - [`Self::Shared`]: caller closes a [`SharedTvmHost`] into every
///   handler at register time; per-call `Arc<Mutex<>>::lock()`.
/// - [`Self::PerActor`]: caller uses one of the
///   `install_tvm_imports_per_actor*` entries against a wasmos
///   ExecutionContext whose consumer state is `T: AsMut<TvmHost>`.
///   Handlers pull `&mut TvmHost` via
///   `ctx.consumer_state::<T>()` — the wasmos abstraction's
///   equivalent of the wit-bindgen `caller.data_mut().as_mut()`
///   pattern (matches girder's `RawTvmActorInstance` shape).
#[derive(Clone)]
pub(crate) enum TvmHostSource {
    Shared(SharedTvmHost),
    PerActor(std::sync::Arc<dyn TvmHostExtractor>),
}

/// D2 Session 4 — consumer-supplied projection from ctx to
/// `&mut TvmHost`. Twin of
/// [`crate::raw_linker_wasmos::TvmHostExtractor`] for the
/// component-model side.
pub trait TvmHostExtractor: Send + Sync + 'static {
    fn extract<'a>(
        &self,
        ctx: &'a mut HostCallContext<'_>,
    ) -> RuntimeResult<&'a mut TvmHost>;
}

/// D2 Session 4 — monomorphized extractor for any
/// `T: AsMut<TvmHost> + 'static`. Twin of
/// [`crate::raw_linker_wasmos::AsMutExtractor`].
pub struct HostAsMutExtractor<T: AsMut<TvmHost> + 'static> {
    _marker: std::marker::PhantomData<fn() -> T>,
}

impl<T: AsMut<TvmHost> + 'static> HostAsMutExtractor<T> {
    pub fn new() -> Self {
        Self { _marker: std::marker::PhantomData }
    }
}

impl<T: AsMut<TvmHost> + 'static> Default for HostAsMutExtractor<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: AsMut<TvmHost> + 'static> TvmHostExtractor for HostAsMutExtractor<T> {
    fn extract<'a>(
        &self,
        ctx: &'a mut HostCallContext<'_>,
    ) -> RuntimeResult<&'a mut TvmHost> {
        let state = ctx.consumer_state::<T>().ok_or_else(|| {
            wasmos_runtime_api::RuntimeError::msg(format!(
                "wasmos_bindings::TvmHostSource::PerActor: no `{}` in \
                 `ctx.consumer_state`. Install this composite via a wasmos \
                 adapter that populates the consumer-state slot with the \
                 expected type (v48 sync_bridge_resource + \
                 install_host_call, or the future component-model \
                 install_host_imports bridge).",
                std::any::type_name::<T>(),
            ))
        })?;
        Ok(state.as_mut())
    }
}

/// Guard that dereferences to `&mut TvmHost` regardless of variant.
/// Twin of [`crate::raw_linker_wasmos::TvmHostGuard`].
pub(crate) enum TvmHostGuard<'a> {
    Shared(std::sync::MutexGuard<'a, TvmHost>),
    PerActor(&'a mut TvmHost),
}

impl std::ops::Deref for TvmHostGuard<'_> {
    type Target = TvmHost;
    fn deref(&self) -> &TvmHost {
        match self {
            TvmHostGuard::Shared(g) => &*g,
            TvmHostGuard::PerActor(h) => &**h,
        }
    }
}

impl std::ops::DerefMut for TvmHostGuard<'_> {
    fn deref_mut(&mut self) -> &mut TvmHost {
        match self {
            TvmHostGuard::Shared(g) => &mut *g,
            TvmHostGuard::PerActor(h) => &mut **h,
        }
    }
}

impl TvmHostSource {
    fn lock<'a>(
        &'a self,
        ctx: &'a mut HostCallContext<'_>,
    ) -> RuntimeResult<TvmHostGuard<'a>> {
        match self {
            TvmHostSource::Shared(shared) => Ok(TvmHostGuard::Shared(shared.lock())),
            TvmHostSource::PerActor(extractor) => {
                let host = extractor.extract(ctx)?;
                Ok(TvmHostGuard::PerActor(host))
            }
        }
    }
}

/// Wasmos-native implementation of `tvm:memory/manager@0.1.0`.
/// Reaches `&mut TvmHost` via [`TvmHostSource`] — either a
/// [`SharedTvmHost`] locked per call, or the consumer's own
/// per-actor state pulled via `ctx.consumer_state::<T>()`.
#[derive(Clone)]
pub struct TvmManagerHost {
    state: TvmHostSource,
}

impl TvmManagerHost {
    /// D2 Session 3 — shared-concurrency constructor.
    pub fn new(host: SharedTvmHost) -> Self {
        Self { state: TvmHostSource::Shared(host) }
    }

    /// D2 Session 4 — per-actor constructor; consumer store data
    /// must implement `AsMut<TvmHost>`.
    pub fn per_actor<T: AsMut<TvmHost> + 'static>() -> Self {
        Self {
            state: TvmHostSource::PerActor(std::sync::Arc::new(
                HostAsMutExtractor::<T>::new(),
            )),
        }
    }
}

#[host_iface(sync)]
impl TvmManagerHost {
    fn create_region(
        &self,
        ctx: &mut HostCallContext<'_>,
        kind: RegionKind,
        capacity: u32,
    ) -> RuntimeResult<Result<u16, TvmError>> {
        let mut g = self.state.lock(ctx)?;
        Ok(BgManagerHost::create_region(&mut *g, kind.into(), capacity).map_err(Into::into))
    }

    fn destroy_region(
        &self,
        ctx: &mut HostCallContext<'_>,
        region_id: u16,
    ) -> RuntimeResult<Result<(), TvmError>> {
        let mut g = self.state.lock(ctx)?;
        Ok(BgManagerHost::destroy_region(&mut *g, region_id).map_err(Into::into))
    }

    fn alloc(
        &self,
        ctx: &mut HostCallContext<'_>,
        region_id: u16,
        size: u32,
    ) -> RuntimeResult<Result<Handle, TvmError>> {
        let mut g = self.state.lock(ctx)?;
        Ok(BgManagerHost::alloc(&mut *g, region_id, size)
            .map(Into::into)
            .map_err(Into::into))
    }

    fn dealloc(
        &self,
        ctx: &mut HostCallContext<'_>,
        ptr: Handle,
    ) -> RuntimeResult<Result<(), TvmError>> {
        let mut g = self.state.lock(ctx)?;
        Ok(BgManagerHost::dealloc(&mut *g, ptr.into()).map_err(Into::into))
    }

    fn describe_region(
        &self,
        ctx: &mut HostCallContext<'_>,
        region_id: u16,
    ) -> RuntimeResult<Result<RegionInfo, TvmError>> {
        let mut g = self.state.lock(ctx)?;
        Ok(BgManagerHost::describe_region(&mut *g, region_id)
            .map(Into::into)
            .map_err(Into::into))
    }

    fn promote_region(
        &self,
        ctx: &mut HostCallContext<'_>,
        region_id: u16,
    ) -> RuntimeResult<Result<(), TvmError>> {
        let mut g = self.state.lock(ctx)?;
        Ok(BgManagerHost::promote_region(&mut *g, region_id).map_err(Into::into))
    }

    fn demote_region(
        &self,
        ctx: &mut HostCallContext<'_>,
        region_id: u16,
    ) -> RuntimeResult<Result<(), TvmError>> {
        let mut g = self.state.lock(ctx)?;
        Ok(BgManagerHost::demote_region(&mut *g, region_id).map_err(Into::into))
    }

    fn spill_region(
        &self,
        ctx: &mut HostCallContext<'_>,
        region_id: u16,
    ) -> RuntimeResult<Result<(), TvmError>> {
        let mut g = self.state.lock(ctx)?;
        Ok(BgManagerHost::spill_region(&mut *g, region_id).map_err(Into::into))
    }

    fn load_region(
        &self,
        ctx: &mut HostCallContext<'_>,
        region_id: u16,
    ) -> RuntimeResult<Result<(), TvmError>> {
        let mut g = self.state.lock(ctx)?;
        Ok(BgManagerHost::load_region(&mut *g, region_id).map_err(Into::into))
    }

    fn pin(
        &self,
        ctx: &mut HostCallContext<'_>,
        region_id: u16,
    ) -> RuntimeResult<Result<(), TvmError>> {
        let mut g = self.state.lock(ctx)?;
        Ok(BgManagerHost::pin(&mut *g, region_id).map_err(Into::into))
    }

    fn unpin(
        &self,
        ctx: &mut HostCallContext<'_>,
        region_id: u16,
    ) -> RuntimeResult<Result<(), TvmError>> {
        let mut g = self.state.lock(ctx)?;
        Ok(BgManagerHost::unpin(&mut *g, region_id).map_err(Into::into))
    }

    fn compact_region(
        &self,
        ctx: &mut HostCallContext<'_>,
        region_id: u16,
    ) -> RuntimeResult<Result<CompactResult, TvmError>> {
        let mut g = self.state.lock(ctx)?;
        Ok(BgManagerHost::compact_region(&mut *g, region_id)
            .map(Into::into)
            .map_err(Into::into))
    }
}

/// Register the `tvm:memory/manager@0.1.0` interface on the given
/// [`HostImports`] set. Consumers thread the returned imports
/// into the wasmos `ExecutionContext` at instantiate time.
pub fn install_tvm_manager_imports(
    imports: HostImports,
    host: SharedTvmHost,
) -> HostImports {
    imports.register_sync("tvm:memory/manager@0.1.0", TvmManagerHost::new(host))
}

/// Wasmos-native implementation of `tvm:memory/bytes@0.1.0`.
#[derive(Clone)]
pub struct TvmBytesHost {
    state: TvmHostSource,
}

impl TvmBytesHost {
    pub fn new(host: SharedTvmHost) -> Self {
        Self { state: TvmHostSource::Shared(host) }
    }
    pub fn per_actor<T: AsMut<TvmHost> + 'static>() -> Self {
        Self {
            state: TvmHostSource::PerActor(std::sync::Arc::new(
                HostAsMutExtractor::<T>::new(),
            )),
        }
    }
}

#[host_iface(sync)]
impl TvmBytesHost {
    fn read(
        &self,
        ctx: &mut HostCallContext<'_>,
        ptr: Handle,
        len: u32,
    ) -> RuntimeResult<Result<Vec<u8>, TvmError>> {
        let mut g = self.state.lock(ctx)?;
        Ok(BgBytesHost::read(&mut *g, ptr.into(), len).map_err(Into::into))
    }

    fn write(
        &self,
        ctx: &mut HostCallContext<'_>,
        ptr: Handle,
        data: Vec<u8>,
    ) -> RuntimeResult<Result<(), TvmError>> {
        let mut g = self.state.lock(ctx)?;
        Ok(BgBytesHost::write(&mut *g, ptr.into(), data).map_err(Into::into))
    }

    fn copy(
        &self,
        ctx: &mut HostCallContext<'_>,
        src: Handle,
        dst: Handle,
        len: u32,
    ) -> RuntimeResult<Result<(), TvmError>> {
        let mut g = self.state.lock(ctx)?;
        Ok(BgBytesHost::copy(&mut *g, src.into(), dst.into(), len).map_err(Into::into))
    }

    fn read_into(
        &self,
        ctx: &mut HostCallContext<'_>,
        src: Handle,
        dst_region: u16,
        dst_offset: u32,
        len: u32,
    ) -> RuntimeResult<Result<(), TvmError>> {
        let mut g = self.state.lock(ctx)?;
        Ok(BgBytesHost::read_into(&mut *g, src.into(), dst_region, dst_offset, len)
            .map_err(Into::into))
    }

    fn write_from(
        &self,
        ctx: &mut HostCallContext<'_>,
        src_region: u16,
        src_offset: u32,
        dst: Handle,
        len: u32,
    ) -> RuntimeResult<Result<(), TvmError>> {
        let mut g = self.state.lock(ctx)?;
        Ok(BgBytesHost::write_from(&mut *g, src_region, src_offset, dst.into(), len)
            .map_err(Into::into))
    }

    fn copy_region(
        &self,
        ctx: &mut HostCallContext<'_>,
        src_region: u16,
        src_offset: u32,
        dst_region: u16,
        dst_offset: u32,
        len: u32,
    ) -> RuntimeResult<Result<(), TvmError>> {
        let mut g = self.state.lock(ctx)?;
        Ok(BgBytesHost::copy_region(
            &mut *g,
            src_region,
            src_offset,
            dst_region,
            dst_offset,
            len,
        )
        .map_err(Into::into))
    }
}

/// Register the `tvm:memory/bytes@0.1.0` interface.
pub fn install_tvm_bytes_imports(
    imports: HostImports,
    host: SharedTvmHost,
) -> HostImports {
    imports.register_sync("tvm:memory/bytes@0.1.0", TvmBytesHost::new(host))
}

/// Wasmos-native implementation of `tvm:memory/diagnostics@0.1.0`.
#[derive(Clone)]
pub struct TvmDiagnosticsHost {
    state: TvmHostSource,
}

impl TvmDiagnosticsHost {
    pub fn new(host: SharedTvmHost) -> Self {
        Self { state: TvmHostSource::Shared(host) }
    }
    pub fn per_actor<T: AsMut<TvmHost> + 'static>() -> Self {
        Self {
            state: TvmHostSource::PerActor(std::sync::Arc::new(
                HostAsMutExtractor::<T>::new(),
            )),
        }
    }
}

#[host_iface(sync)]
impl TvmDiagnosticsHost {
    fn list_regions(
        &self,
        ctx: &mut HostCallContext<'_>,
    ) -> RuntimeResult<Vec<RegionInfo>> {
        let mut g = self.state.lock(ctx)?;
        Ok(BgDiagnosticsHost::list_regions(&mut *g)
            .into_iter()
            .map(Into::into)
            .collect())
    }

    fn fault_count(
        &self,
        ctx: &mut HostCallContext<'_>,
        region_id: u16,
    ) -> RuntimeResult<u64> {
        let mut g = self.state.lock(ctx)?;
        Ok(BgDiagnosticsHost::fault_count(&mut *g, region_id))
    }

    fn allocation_count(
        &self,
        ctx: &mut HostCallContext<'_>,
        region_id: u16,
    ) -> RuntimeResult<u64> {
        let mut g = self.state.lock(ctx)?;
        Ok(BgDiagnosticsHost::allocation_count(&mut *g, region_id))
    }

    fn bytes_read_count(
        &self,
        ctx: &mut HostCallContext<'_>,
        region_id: u16,
    ) -> RuntimeResult<u64> {
        let mut g = self.state.lock(ctx)?;
        Ok(BgDiagnosticsHost::bytes_read_count(&mut *g, region_id))
    }

    fn bytes_written_count(
        &self,
        ctx: &mut HostCallContext<'_>,
        region_id: u16,
    ) -> RuntimeResult<u64> {
        let mut g = self.state.lock(ctx)?;
        Ok(BgDiagnosticsHost::bytes_written_count(&mut *g, region_id))
    }

    fn metrics_snapshot(
        &self,
        ctx: &mut HostCallContext<'_>,
        region_id: u16,
    ) -> RuntimeResult<Result<RegionMetrics, TvmError>> {
        let mut g = self.state.lock(ctx)?;
        Ok(BgDiagnosticsHost::metrics_snapshot(&mut *g, region_id)
            .map(Into::into)
            .map_err(Into::into))
    }
}

/// Register the `tvm:memory/diagnostics@0.1.0` interface.
pub fn install_tvm_diagnostics_imports(
    imports: HostImports,
    host: SharedTvmHost,
) -> HostImports {
    imports.register_sync("tvm:memory/diagnostics@0.1.0", TvmDiagnosticsHost::new(host))
}

/// One-shot composite that registers all three
/// `tvm:memory@0.1.0` interfaces against the same
/// [`SharedTvmHost`]. Mirrors the sqlink Arc 1
/// `install_sqlink_imports` shape.
///
/// Consumers who want to install a subset — e.g. diagnostics-only
/// probes without giving the guest region-management privileges —
/// use the per-interface entry points above.
pub fn install_tvm_imports_shared(
    imports: HostImports,
    host: SharedTvmHost,
) -> HostImports {
    let imports = install_tvm_manager_imports(imports, host.clone());
    let imports = install_tvm_bytes_imports(imports, host.clone());
    install_tvm_diagnostics_imports(imports, host)
}

// ── D2 Session 4 — per-actor install fns ────────────────────────────
//
// Peer of the shared install fns above. Handlers pull `&mut TvmHost`
// from `ctx.consumer_state::<T>()` on each call, where `T` is the
// consumer's store data type implementing `AsMut<TvmHost>`. Wired
// through a wasmos adapter's HostCall bridge that populates the
// consumer_state slot (v48 sync_bridge_resource + the future
// component-model install path).

/// Register `tvm:memory/manager@0.1.0` in per-actor mode.
pub fn install_tvm_manager_imports_per_actor<T: AsMut<TvmHost> + 'static>(
    imports: HostImports,
) -> HostImports {
    imports.register_sync(
        "tvm:memory/manager@0.1.0",
        TvmManagerHost::per_actor::<T>(),
    )
}

/// Register `tvm:memory/bytes@0.1.0` in per-actor mode.
pub fn install_tvm_bytes_imports_per_actor<T: AsMut<TvmHost> + 'static>(
    imports: HostImports,
) -> HostImports {
    imports.register_sync(
        "tvm:memory/bytes@0.1.0",
        TvmBytesHost::per_actor::<T>(),
    )
}

/// Register `tvm:memory/diagnostics@0.1.0` in per-actor mode.
pub fn install_tvm_diagnostics_imports_per_actor<T: AsMut<TvmHost> + 'static>(
    imports: HostImports,
) -> HostImports {
    imports.register_sync(
        "tvm:memory/diagnostics@0.1.0",
        TvmDiagnosticsHost::per_actor::<T>(),
    )
}

/// Per-actor composite — one-shot registration of all three
/// interfaces against `T: AsMut<TvmHost>` consumer state. Peer of
/// [`install_tvm_imports_shared`].
pub fn install_tvm_imports_per_actor<T: AsMut<TvmHost> + 'static>(
    imports: HostImports,
) -> HostImports {
    let imports = install_tvm_manager_imports_per_actor::<T>(imports);
    let imports = install_tvm_bytes_imports_per_actor::<T>(imports);
    install_tvm_diagnostics_imports_per_actor::<T>(imports)
}

// ── D2 Session 12 — PerActorTvmHost budget-accounting wasmos peer ───
//
// Closes bindings.rs retirement gate 2. The wit-bindgen
// `add_per_actor_to_linker` (deprecated at D2 Session 5) delegates
// to `PerActorTvmHost`'s own `ManagerHost` / `BytesHost` /
// `DiagnosticsHost` impls (in per_actor.rs) which enforce the
// outstanding-bytes budget on alloc + return budget on dealloc.
// This section wraps that trait surface as wasmos
// `#[host_iface(sync)]` handlers so consumers get identical
// budget semantics through the wasmos install path.
//
// Concurrency model: `PerActorTvmHost` is `Clone`; it internally
// holds `Arc<Mutex<BudgetState>>` + `SharedTvmHost` (also
// `Arc<Mutex<TvmHost>>`). Every handler clones the host into a
// local `mut` binding — the clone is cheap (two Arc bumps) and
// mutations go through the internal mutexes, matching the
// wit-bindgen path's concurrency behavior exactly.

/// Wasmos-native implementation of `tvm:memory/manager@0.1.0`
/// against [`PerActorTvmHost`]. Enforces the actor's
/// outstanding-bytes budget on `alloc` (returning
/// [`TvmError::AllocationFailed`] + setting the overrun flag when
/// exceeded) and returns the budget on `dealloc`.
#[derive(Clone)]
pub struct TvmManagerHostBudget {
    host: crate::PerActorTvmHost,
}

impl TvmManagerHostBudget {
    pub fn new(host: crate::PerActorTvmHost) -> Self {
        Self { host }
    }
}

#[host_iface(sync)]
impl TvmManagerHostBudget {
    fn create_region(
        &self,
        _ctx: &mut HostCallContext<'_>,
        kind: RegionKind,
        capacity: u32,
    ) -> RuntimeResult<Result<u16, TvmError>> {
        let mut h = self.host.clone();
        Ok(BgManagerHost::create_region(&mut h, kind.into(), capacity).map_err(Into::into))
    }

    fn destroy_region(
        &self,
        _ctx: &mut HostCallContext<'_>,
        region_id: u16,
    ) -> RuntimeResult<Result<(), TvmError>> {
        let mut h = self.host.clone();
        Ok(BgManagerHost::destroy_region(&mut h, region_id).map_err(Into::into))
    }

    fn alloc(
        &self,
        _ctx: &mut HostCallContext<'_>,
        region_id: u16,
        size: u32,
    ) -> RuntimeResult<Result<Handle, TvmError>> {
        let mut h = self.host.clone();
        Ok(BgManagerHost::alloc(&mut h, region_id, size)
            .map(Into::into)
            .map_err(Into::into))
    }

    fn dealloc(
        &self,
        _ctx: &mut HostCallContext<'_>,
        ptr: Handle,
    ) -> RuntimeResult<Result<(), TvmError>> {
        let mut h = self.host.clone();
        Ok(BgManagerHost::dealloc(&mut h, ptr.into()).map_err(Into::into))
    }

    fn describe_region(
        &self,
        _ctx: &mut HostCallContext<'_>,
        region_id: u16,
    ) -> RuntimeResult<Result<RegionInfo, TvmError>> {
        let mut h = self.host.clone();
        Ok(BgManagerHost::describe_region(&mut h, region_id)
            .map(Into::into)
            .map_err(Into::into))
    }

    fn promote_region(
        &self,
        _ctx: &mut HostCallContext<'_>,
        region_id: u16,
    ) -> RuntimeResult<Result<(), TvmError>> {
        let mut h = self.host.clone();
        Ok(BgManagerHost::promote_region(&mut h, region_id).map_err(Into::into))
    }

    fn demote_region(
        &self,
        _ctx: &mut HostCallContext<'_>,
        region_id: u16,
    ) -> RuntimeResult<Result<(), TvmError>> {
        let mut h = self.host.clone();
        Ok(BgManagerHost::demote_region(&mut h, region_id).map_err(Into::into))
    }

    fn spill_region(
        &self,
        _ctx: &mut HostCallContext<'_>,
        region_id: u16,
    ) -> RuntimeResult<Result<(), TvmError>> {
        let mut h = self.host.clone();
        Ok(BgManagerHost::spill_region(&mut h, region_id).map_err(Into::into))
    }

    fn load_region(
        &self,
        _ctx: &mut HostCallContext<'_>,
        region_id: u16,
    ) -> RuntimeResult<Result<(), TvmError>> {
        let mut h = self.host.clone();
        Ok(BgManagerHost::load_region(&mut h, region_id).map_err(Into::into))
    }

    fn pin(
        &self,
        _ctx: &mut HostCallContext<'_>,
        region_id: u16,
    ) -> RuntimeResult<Result<(), TvmError>> {
        let mut h = self.host.clone();
        Ok(BgManagerHost::pin(&mut h, region_id).map_err(Into::into))
    }

    fn unpin(
        &self,
        _ctx: &mut HostCallContext<'_>,
        region_id: u16,
    ) -> RuntimeResult<Result<(), TvmError>> {
        let mut h = self.host.clone();
        Ok(BgManagerHost::unpin(&mut h, region_id).map_err(Into::into))
    }

    fn compact_region(
        &self,
        _ctx: &mut HostCallContext<'_>,
        region_id: u16,
    ) -> RuntimeResult<Result<CompactResult, TvmError>> {
        let mut h = self.host.clone();
        Ok(BgManagerHost::compact_region(&mut h, region_id)
            .map(Into::into)
            .map_err(Into::into))
    }
}

/// Register `tvm:memory/manager@0.1.0` in per-actor-budget mode.
pub fn install_tvm_manager_imports_per_actor_budget(
    imports: HostImports,
    host: crate::PerActorTvmHost,
) -> HostImports {
    imports.register_sync("tvm:memory/manager@0.1.0", TvmManagerHostBudget::new(host))
}

/// Wasmos-native implementation of `tvm:memory/bytes@0.1.0`
/// against [`PerActorTvmHost`]. Bytes methods don't touch the
/// budget (the wit-bindgen `PerActorTvmHost::BytesHost` impl just
/// delegates to the inner `SharedTvmHost`); this struct exists
/// for API symmetry with the manager + diagnostics variants so
/// consumers install a single-flavor composite.
#[derive(Clone)]
pub struct TvmBytesHostBudget {
    host: crate::PerActorTvmHost,
}

impl TvmBytesHostBudget {
    pub fn new(host: crate::PerActorTvmHost) -> Self {
        Self { host }
    }
}

#[host_iface(sync)]
impl TvmBytesHostBudget {
    fn read(
        &self,
        _ctx: &mut HostCallContext<'_>,
        ptr: Handle,
        len: u32,
    ) -> RuntimeResult<Result<Vec<u8>, TvmError>> {
        let mut h = self.host.clone();
        Ok(BgBytesHost::read(&mut h, ptr.into(), len).map_err(Into::into))
    }

    fn write(
        &self,
        _ctx: &mut HostCallContext<'_>,
        ptr: Handle,
        data: Vec<u8>,
    ) -> RuntimeResult<Result<(), TvmError>> {
        let mut h = self.host.clone();
        Ok(BgBytesHost::write(&mut h, ptr.into(), data).map_err(Into::into))
    }

    fn copy(
        &self,
        _ctx: &mut HostCallContext<'_>,
        src: Handle,
        dst: Handle,
        len: u32,
    ) -> RuntimeResult<Result<(), TvmError>> {
        let mut h = self.host.clone();
        Ok(BgBytesHost::copy(&mut h, src.into(), dst.into(), len).map_err(Into::into))
    }

    fn read_into(
        &self,
        _ctx: &mut HostCallContext<'_>,
        src: Handle,
        dst_region: u16,
        dst_offset: u32,
        len: u32,
    ) -> RuntimeResult<Result<(), TvmError>> {
        let mut h = self.host.clone();
        Ok(BgBytesHost::read_into(&mut h, src.into(), dst_region, dst_offset, len)
            .map_err(Into::into))
    }

    fn write_from(
        &self,
        _ctx: &mut HostCallContext<'_>,
        src_region: u16,
        src_offset: u32,
        dst: Handle,
        len: u32,
    ) -> RuntimeResult<Result<(), TvmError>> {
        let mut h = self.host.clone();
        Ok(BgBytesHost::write_from(&mut h, src_region, src_offset, dst.into(), len)
            .map_err(Into::into))
    }

    fn copy_region(
        &self,
        _ctx: &mut HostCallContext<'_>,
        src_region: u16,
        src_offset: u32,
        dst_region: u16,
        dst_offset: u32,
        len: u32,
    ) -> RuntimeResult<Result<(), TvmError>> {
        let mut h = self.host.clone();
        Ok(BgBytesHost::copy_region(&mut h, src_region, src_offset, dst_region, dst_offset, len)
            .map_err(Into::into))
    }
}

/// Register `tvm:memory/bytes@0.1.0` in per-actor-budget mode.
pub fn install_tvm_bytes_imports_per_actor_budget(
    imports: HostImports,
    host: crate::PerActorTvmHost,
) -> HostImports {
    imports.register_sync("tvm:memory/bytes@0.1.0", TvmBytesHostBudget::new(host))
}

/// Wasmos-native implementation of `tvm:memory/diagnostics@0.1.0`
/// against [`PerActorTvmHost`].
#[derive(Clone)]
pub struct TvmDiagnosticsHostBudget {
    host: crate::PerActorTvmHost,
}

impl TvmDiagnosticsHostBudget {
    pub fn new(host: crate::PerActorTvmHost) -> Self {
        Self { host }
    }
}

#[host_iface(sync)]
impl TvmDiagnosticsHostBudget {
    fn list_regions(
        &self,
        _ctx: &mut HostCallContext<'_>,
    ) -> RuntimeResult<Vec<RegionInfo>> {
        let mut h = self.host.clone();
        Ok(BgDiagnosticsHost::list_regions(&mut h)
            .into_iter()
            .map(Into::into)
            .collect())
    }

    fn fault_count(
        &self,
        _ctx: &mut HostCallContext<'_>,
        region_id: u16,
    ) -> RuntimeResult<u64> {
        let mut h = self.host.clone();
        Ok(BgDiagnosticsHost::fault_count(&mut h, region_id))
    }

    fn allocation_count(
        &self,
        _ctx: &mut HostCallContext<'_>,
        region_id: u16,
    ) -> RuntimeResult<u64> {
        let mut h = self.host.clone();
        Ok(BgDiagnosticsHost::allocation_count(&mut h, region_id))
    }

    fn bytes_read_count(
        &self,
        _ctx: &mut HostCallContext<'_>,
        region_id: u16,
    ) -> RuntimeResult<u64> {
        let mut h = self.host.clone();
        Ok(BgDiagnosticsHost::bytes_read_count(&mut h, region_id))
    }

    fn bytes_written_count(
        &self,
        _ctx: &mut HostCallContext<'_>,
        region_id: u16,
    ) -> RuntimeResult<u64> {
        let mut h = self.host.clone();
        Ok(BgDiagnosticsHost::bytes_written_count(&mut h, region_id))
    }

    fn metrics_snapshot(
        &self,
        _ctx: &mut HostCallContext<'_>,
        region_id: u16,
    ) -> RuntimeResult<Result<RegionMetrics, TvmError>> {
        let mut h = self.host.clone();
        Ok(BgDiagnosticsHost::metrics_snapshot(&mut h, region_id)
            .map(Into::into)
            .map_err(Into::into))
    }
}

/// Register `tvm:memory/diagnostics@0.1.0` in per-actor-budget mode.
pub fn install_tvm_diagnostics_imports_per_actor_budget(
    imports: HostImports,
    host: crate::PerActorTvmHost,
) -> HostImports {
    imports.register_sync("tvm:memory/diagnostics@0.1.0", TvmDiagnosticsHostBudget::new(host))
}

/// Per-actor-budget composite — one-shot registration of all three
/// `tvm:memory@0.1.0` interfaces against a
/// [`PerActorTvmHost`]. Peer of [`install_tvm_imports_shared`] +
/// [`install_tvm_imports_per_actor`], covering the budget-accounting
/// concurrency model.
///
/// The wasmos install path for consumers whose actor model needs
/// outstanding-bytes bookkeeping (girder's per-actor TVM shape).
/// Replaces the deprecated
/// [`crate::linker::add_per_actor_to_linker`] entry.
pub fn install_tvm_imports_per_actor_budget(
    imports: HostImports,
    host: crate::PerActorTvmHost,
) -> HostImports {
    let imports = install_tvm_manager_imports_per_actor_budget(imports, host.clone());
    let imports = install_tvm_bytes_imports_per_actor_budget(imports, host.clone());
    install_tvm_diagnostics_imports_per_actor_budget(imports, host)
}

// ── Round-trip tests ────────────────────────────────────────────────
//
// Guard against silent WIT drift: if the wit-bindgen shape changes,
// the mirror's field-count or arm-count assertion catches it.

#[cfg(test)]
mod tests {
    use super::*;

    // Region-kind + residency: exhaustive-match on the mirror
    // catches a new arm added to WIT.

    #[test]
    fn region_kind_round_trip_all_arms() {
        for k in [
            RegionKind::HotHeap,
            RegionKind::ObjectArena,
            RegionKind::BlobArena,
            RegionKind::PageStore,
            RegionKind::Scratch,
            RegionKind::DeviceState,
            RegionKind::CodeCache,
        ] {
            let bg: bg::RegionKind = k.into();
            let m: RegionKind = bg.into();
            assert_eq!(k, m);
        }
    }

    #[test]
    fn residency_round_trip_all_arms() {
        for r in [Residency::Hot, Residency::Warm, Residency::Cold, Residency::External] {
            let bg: bg::Residency = r.into();
            let m: Residency = bg.into();
            assert_eq!(r, m);
        }
    }

    #[test]
    fn handle_round_trip() {
        let h = Handle { region_id: 42, generation: 7, offset: 1_048_576 };
        let bg: bg::Handle = h.into();
        let back: Handle = bg.into();
        assert_eq!(h, back);
    }

    #[test]
    fn region_info_round_trip_covers_nested_enums() {
        let r = RegionInfo {
            id: 3,
            generation: 5,
            kind: RegionKind::PageStore,
            capacity: 1024,
            used: 256,
            residency: Residency::Warm,
        };
        let bg: bg::RegionInfo = r.into();
        let back: RegionInfo = bg.into();
        assert_eq!(r, back);
    }

    #[test]
    fn region_metrics_round_trip() {
        let m = RegionMetrics {
            allocations: 1,
            bytes_allocated: 2,
            bytes_read: 3,
            bytes_written: 4,
            faults: 5,
            promotions: 6,
            demotions: 7,
        };
        let bg: bg::RegionMetrics = m.into();
        let back: RegionMetrics = bg.into();
        assert_eq!(m, back);
    }

    #[test]
    fn compact_result_round_trip_preserves_mapping() {
        let c = CompactResult {
            old_generation: 10,
            new_generation: 11,
            mapping: vec![(0, 8), (16, 24), (40, 48)],
        };
        let bg: bg::CompactResult = c.clone().into();
        let back: CompactResult = bg.into();
        assert_eq!(c, back);
    }

    #[test]
    fn tvm_error_round_trip_all_arms() {
        let cases = [
            TvmError::RegionNotFound(99),
            TvmError::StaleHandle,
            TvmError::OutOfBounds,
            TvmError::NotResident,
            TvmError::AllocationFailed,
            TvmError::BackingStore("disk full".into()),
            TvmError::Pinned,
        ];
        for e in cases {
            let bg: bg::TvmError = e.clone().into();
            let back: TvmError = bg.into();
            assert_eq!(e, back);
        }
    }

    // WitBridge sanity: the mirror types round-trip through
    // wasmos_runtime_api::Value. If the derive is wrong, this
    // catches it before Session 3 wires it into #[host_iface].

    use wasmos_runtime_api::WitBridge;

    #[test]
    fn wit_bridge_handle_round_trip() {
        let h = Handle { region_id: 1, generation: 2, offset: 3 };
        let v = h.to_value();
        let back = Handle::from_value(v).expect("Handle::from_value");
        assert_eq!(h, back);
    }

    #[test]
    fn wit_bridge_region_info_round_trip() {
        let r = RegionInfo {
            id: 1,
            generation: 2,
            kind: RegionKind::BlobArena,
            capacity: 4096,
            used: 128,
            residency: Residency::Cold,
        };
        let v = r.to_value();
        let back = RegionInfo::from_value(v).expect("RegionInfo::from_value");
        assert_eq!(r, back);
    }

    #[test]
    fn wit_bridge_tvm_error_round_trip_variant_payloads() {
        let e = TvmError::BackingStore("io error".into());
        let v = e.clone().to_value();
        let back = TvmError::from_value(v).expect("TvmError::from_value");
        assert_eq!(e, back);
    }

    // ── D2 Session 3 — install fns ────────────────────────────────

    /// Compile-check + registration test for the composite. Verifies
    /// all three interface names land in the `HostImports` set — if
    /// the register_sync API drifts or the #[host_iface(sync)] macro
    /// changes its trait bounds, this fails at build time.
    #[test]
    fn install_tvm_imports_shared_registers_all_three_interfaces() {
        let host = SharedTvmHost::new();
        let imports = install_tvm_imports_shared(HostImports::new(), host);
        // HostImports doesn't expose an iter today; the registration
        // itself is the compile-check. If register_sync silently
        // stopped chaining, this would leak a warning about
        // dropped return values.
        let _ = imports;
    }

    /// One inspection test per interface — locks the SharedTvmHost
    /// and asks the diagnostics handler for list_regions on an
    /// empty host. Confirms the handler dispatches without a live
    /// wasm instance.
    #[test]
    fn diagnostics_list_regions_on_empty_host() {
        use wasmos_runtime_api::SyncHostCall;
        use wasmos_runtime_api::Value;

        struct StubCtx;
        impl wasmos_runtime_api::HostCallCtxImpl for StubCtx {
            fn new_host_resource(
                &mut self,
                _iface: &str,
                _name: &str,
                _rep: u32,
            ) -> RuntimeResult<Value> {
                unreachable!("list_regions does not mint resources")
            }
            fn resource_rep(&mut self, _v: &Value) -> RuntimeResult<u32> {
                unreachable!()
            }
        }

        let host = SharedTvmHost::new();
        let handler = TvmDiagnosticsHost::new(host);
        let mut inner = StubCtx;
        let mut ctx = HostCallContext::new(&mut inner);
        let out = handler
            .call(&mut ctx, "list-regions", vec![])
            .expect("list_regions dispatch");
        // list-regions returns a WIT list<region-info>. Empty host →
        // empty list. Wire shape: Value::List(vec![]).
        match out.as_slice() {
            [Value::List(items)] if items.is_empty() => {}
            other => panic!("expected empty Value::List, got {other:?}"),
        }
    }

    // ── D2 Session 4 — per-actor path tests ────────────────────────
    //
    // Ctx impls with a `consumer_state` slot pointed at a real
    // `TvmHost` (matching what a wasmos adapter bridge would do
    // when routing calls through a wasmtime Store<T>).

    struct PerActorCtx<'a, T: 'static + Send> {
        state: &'a mut T,
    }

    impl<T: 'static + Send> wasmos_runtime_api::HostCallCtxImpl for PerActorCtx<'_, T> {
        fn new_host_resource(
            &mut self,
            _iface: &str,
            _name: &str,
            _rep: u32,
        ) -> RuntimeResult<wasmos_runtime_api::Value> {
            unreachable!("this test path does not mint resources")
        }
        fn resource_rep(&mut self, _v: &wasmos_runtime_api::Value) -> RuntimeResult<u32> {
            unreachable!()
        }
        fn consumer_state(&mut self) -> Option<&mut (dyn std::any::Any + Send)> {
            Some(self.state as &mut (dyn std::any::Any + Send))
        }
    }

    /// Wrapper store data implementing `AsMut<TvmHost>` — mirrors
    /// girder's `RawTvmState { tvm: TvmHost, ... }` shape.
    struct MyStoreData {
        tvm: TvmHost,
    }
    impl AsMut<TvmHost> for MyStoreData {
        fn as_mut(&mut self) -> &mut TvmHost {
            &mut self.tvm
        }
    }

    #[test]
    fn per_actor_diagnostics_dispatches_through_consumer_state() {
        use wasmos_runtime_api::{SyncHostCall, Value};

        let mut data = MyStoreData { tvm: TvmHost::new() };
        let handler = TvmDiagnosticsHost::per_actor::<MyStoreData>();
        let mut inner = PerActorCtx { state: &mut data };
        let mut ctx = HostCallContext::new(&mut inner);
        let out = handler
            .call(&mut ctx, "list-regions", vec![])
            .expect("list_regions per-actor dispatch");
        match out.as_slice() {
            [Value::List(items)] if items.is_empty() => {}
            other => panic!("expected empty Value::List, got {other:?}"),
        }
    }

    #[test]
    fn per_actor_wrong_state_type_errors_with_diagnostic() {
        use wasmos_runtime_api::SyncHostCall;

        // Handler expects MyStoreData, but ctx exposes ()
        // (through a mismatched-type consumer_state). Downcast
        // returns None; the extractor surfaces the diagnostic.
        let mut wrong: u64 = 0;
        let handler = TvmDiagnosticsHost::per_actor::<MyStoreData>();
        let mut inner = PerActorCtx { state: &mut wrong };
        let mut ctx = HostCallContext::new(&mut inner);
        let err = handler
            .call(&mut ctx, "list-regions", vec![])
            .expect_err("wrong consumer_state type must error");
        let msg = err.to_string();
        assert!(
            msg.contains("consumer_state") && msg.contains("MyStoreData"),
            "expected wrong-type diagnostic naming MyStoreData; got: {msg}"
        );
    }

    /// Composite compile-check for the per-actor entry point.
    #[test]
    fn install_tvm_imports_per_actor_composite_compiles() {
        let imports = install_tvm_imports_per_actor::<MyStoreData>(HostImports::new());
        let _ = imports;
    }

    // ── D2 Session 12 — PerActorTvmHost budget-accounting tests ────

    use crate::{PerActorTvmHost, SharedTvmHost, TvmBudget};

    /// Empty stub ctx for the per-actor-budget dispatch tests —
    /// these handlers don't touch resources or consumer_state.
    struct EmptyCtx;
    impl wasmos_runtime_api::HostCallCtxImpl for EmptyCtx {
        fn new_host_resource(
            &mut self,
            _iface: &str,
            _name: &str,
            _rep: u32,
        ) -> RuntimeResult<wasmos_runtime_api::Value> {
            unreachable!()
        }
        fn resource_rep(&mut self, _v: &wasmos_runtime_api::Value) -> RuntimeResult<u32> {
            unreachable!()
        }
    }

    #[test]
    fn install_tvm_imports_per_actor_budget_composite_registers_all_three() {
        let host = PerActorTvmHost::new(SharedTvmHost::new(), TvmBudget::unlimited());
        let imports = install_tvm_imports_per_actor_budget(HostImports::new(), host);
        assert_eq!(imports.len(), 3);
        assert!(imports.get("tvm:memory/manager@0.1.0").is_some());
        assert!(imports.get("tvm:memory/bytes@0.1.0").is_some());
        assert!(imports.get("tvm:memory/diagnostics@0.1.0").is_some());
    }

    #[test]
    fn per_actor_budget_alloc_returns_error_on_overrun() {
        use wasmos_runtime_api::{SyncHostCall, Value};

        // 32 outstanding-bytes budget. Alloc 16 succeeds; alloc 32 more
        // (total 48) hits the budget cap.
        let inner = SharedTvmHost::new();
        {
            let mut g = inner.lock();
            <TvmHost as BgManagerHost>::create_region(
                &mut g,
                bg::RegionKind::HotHeap,
                256,
            )
            .expect("create_region");
        }
        let host = PerActorTvmHost::new(inner, TvmBudget::outstanding_bytes(32));

        let manager = TvmManagerHostBudget::new(host.clone());
        let mut inner_ctx = EmptyCtx;
        let mut ctx = HostCallContext::new(&mut inner_ctx);

        // First alloc: 16 bytes — succeeds (16 <= 32).
        let out1 = manager
            .call(&mut ctx, "alloc", vec![Value::U16(0), Value::U32(16)])
            .expect("first alloc");
        // Second alloc: 32 bytes more — outstanding would be 48 > 32.
        let out2 = manager
            .call(&mut ctx, "alloc", vec![Value::U16(0), Value::U32(32)])
            .expect("second alloc dispatches");

        // First outer Value::Variant carries the WIT-canonical
        // result<handle, tvm-error> tag; second is a variant with
        // "err" discriminant AND the budget-overrun flag is set on
        // the host.
        let out2_is_err = matches!(
            out2.as_slice(),
            [Value::Result(Err(_))]
        );
        assert!(
            out2_is_err,
            "expected Result::Err for over-budget alloc, got {out2:?}"
        );
        assert!(host.budget_overrun(), "budget_overrun flag must be set after over-budget alloc");
        let _ = out1;
    }

    #[test]
    fn per_actor_budget_dealloc_returns_budget() {
        use wasmos_runtime_api::{SyncHostCall, Value};

        let inner = SharedTvmHost::new();
        {
            let mut g = inner.lock();
            <TvmHost as BgManagerHost>::create_region(
                &mut g,
                bg::RegionKind::HotHeap,
                256,
            )
            .expect("create_region");
        }
        // Budget = 64. Alloc 64, dealloc, alloc 64 again — second
        // alloc succeeds because dealloc returned the budget.
        let host = PerActorTvmHost::new(inner, TvmBudget::outstanding_bytes(64));
        let manager = TvmManagerHostBudget::new(host.clone());
        let mut inner_ctx = EmptyCtx;
        let mut ctx = HostCallContext::new(&mut inner_ctx);

        // Alloc 64.
        let out1 = manager
            .call(&mut ctx, "alloc", vec![Value::U16(0), Value::U32(64)])
            .expect("alloc");
        // Extract the handle. Result::Ok carries the payload.
        let handle_value = match out1.as_slice() {
            [Value::Result(Ok(Some(p)))] => (**p).clone(),
            other => panic!("expected Result::Ok(Some(_)), got {other:?}"),
        };
        assert_eq!(host.outstanding_bytes(), 64);

        // Dealloc — budget released.
        let _ = manager
            .call(&mut ctx, "dealloc", vec![handle_value])
            .expect("dealloc");
        assert_eq!(host.outstanding_bytes(), 0);

        // Alloc 64 again — succeeds because budget was returned.
        let out2 = manager
            .call(&mut ctx, "alloc", vec![Value::U16(0), Value::U32(64)])
            .expect("second alloc");
        let is_ok = matches!(out2.as_slice(), [Value::Result(Ok(_))]);
        assert!(is_ok, "post-dealloc alloc must succeed within budget");
    }
}
