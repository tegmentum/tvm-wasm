//! Imported regions — TVM regions whose **backing storage is a wasmtime
//! linear memory** rather than a host-side Vec. The guest imports the
//! memory and accesses it natively (`i32.load`); TVM's role is region
//! lifecycle management (alloc, dealloc, pin, spill, compact, generation
//! validation), not byte movement.
//!
//! ## What this gives you
//!
//! - **Native access cost** — every read/write is an engine-emitted
//!   wasm load/store, no host call. Same speed as M32.
//! - **TVM lifecycle for free** — the same `pin` / `spill` / `compact`
//!   primitives work, since the directory still owns the allocator and
//!   metadata. Only the bytes move differently.
//! - **>4 GiB via region composition** — each imported region is its own
//!   32-bit memory; the guest switches between them via the memory
//!   immediate on each load. No per-instruction address-width tax.
//!
//! ## When to use vs. host regions
//!
//! - **Imported regions**: hot path data accessed by guest code. The
//!   guest must declare `(import "tvm" "r<id>" (memory ...))` for each.
//! - **Host regions**: data that crosses host/guest boundaries
//!   (serialization, IO buffers, regions touched primarily from the
//!   host side). The raw fast path through `tvm.read`/`tvm.write`
//!   stays available.
//!
//! Both flavors coexist in a single `TvmHost`. Region IDs are unique
//! across both.

use std::sync::atomic::{AtomicU64, Ordering};

use tvm_core::{
    AllocatorKind, Handle, PlacementPolicy, Region, RegionAllocator, RegionKind, Residency, Result,
    TvmError,
};
use wasmtime::{AsContext, AsContextMut, Memory, MemoryType, StoreContextMut};

/// Configured `(engine, store, linker, payload)` returned by the
/// imported-region setup helpers. `T` is the per-region payload the helper
/// hands back — [`Handle`]s when data was written, or raw region ids.
pub type ImportedSetup<T> = wasmtime::Result<(
    wasmtime::Engine,
    wasmtime::Store<crate::TvmHost>,
    wasmtime::Linker<crate::TvmHost>,
    Vec<T>,
)>;

/// One imported-memory region: meta + allocator + the underlying wasmtime
/// memory. Mirrors `RegionEntry<M>` but for the imported case.
pub struct ImportedRegion {
    pub meta: Region,
    pub memory: Memory,
    pub allocator: RegionAllocator,
    pub allocations: AtomicU64,
    pub bytes_allocated: AtomicU64,
}

impl ImportedRegion {
    /// Create a new imported region with a freshly allocated wasmtime
    /// memory of the requested capacity (rounded up to wasm pages).
    pub fn new<T>(
        store: &mut StoreContextMut<'_, T>,
        id: u16,
        kind: RegionKind,
        capacity: u32,
        allocator: AllocatorKind,
        policy: PlacementPolicy,
    ) -> Result<Self> {
        const PAGE: u64 = 65_536;
        let pages = (capacity as u64).div_ceil(PAGE).max(1) as u32;
        // Bind max == min: the region's capacity is fixed at creation
        // time and never grows. Telling wasmtime this lets the JIT fold
        // bounds checks against a constant, and combined with the
        // `memory_may_move(false)` engine config it reserves exactly
        // `pages * 64 KiB` of virtual address space (no wasted 4 GiB
        // reservation per memory) and hoists the memory base pointer out
        // of hot loops.
        let memory = Memory::new(store.as_context_mut(), MemoryType::new(pages, Some(pages)))
            .map_err(|e| TvmError::BackingStore(e.to_string()))?;
        Ok(Self {
            meta: Region {
                id,
                generation: 1,
                kind,
                capacity,
                used: 0,
                residency: policy.initial_residency,
                pinned: false,
                pinnable: policy.pinnable,
                spillable: policy.spillable,
            },
            memory,
            allocator: RegionAllocator::new(allocator, capacity),
            allocations: AtomicU64::new(0),
            bytes_allocated: AtomicU64::new(0),
        })
    }

    /// Allocate `size` bytes within this region. Returns a handle whose
    /// offset is valid for direct `i32.load` access against the imported
    /// memory.
    pub fn alloc(&mut self, size: u32) -> Result<Handle> {
        let offset = self.allocator.alloc(size, 1)?;
        self.meta.used = self.allocator.used();
        self.allocations.fetch_add(1, Ordering::Relaxed);
        self.bytes_allocated
            .fetch_add(size as u64, Ordering::Relaxed);
        Ok(Handle {
            region_id: self.meta.id,
            generation: self.meta.generation,
            offset,
        })
    }

    /// Free a previously-allocated handle. Validates generation; bump
    /// allocator returns success without freeing (matching host-region
    /// semantics).
    pub fn dealloc(&mut self, handle: Handle) -> Result<()> {
        if handle.generation != self.meta.generation {
            return Err(TvmError::StaleHandle);
        }
        self.allocator.dealloc(handle.offset)?;
        self.meta.used = self.allocator.used();
        Ok(())
    }

    /// Read bytes from the imported memory through the host-mediated
    /// path. Useful for bridge code (serialization, debugging); guests
    /// typically use native loads instead.
    pub fn read<T: 'static>(
        &self,
        store: &impl AsContext<Data = T>,
        handle: Handle,
        buf: &mut [u8],
    ) -> Result<()> {
        if handle.generation != self.meta.generation {
            return Err(TvmError::StaleHandle);
        }
        let end = handle
            .offset
            .checked_add(buf.len() as u32)
            .ok_or(TvmError::OutOfBounds)?;
        if end > self.meta.capacity {
            return Err(TvmError::OutOfBounds);
        }
        self.memory
            .read(store.as_context(), handle.offset as usize, buf)
            .map_err(|_| TvmError::OutOfBounds)
    }

    pub fn write<T: 'static>(
        &self,
        store: &mut impl AsContextMut<Data = T>,
        handle: Handle,
        data: &[u8],
    ) -> Result<()> {
        if handle.generation != self.meta.generation {
            return Err(TvmError::StaleHandle);
        }
        let end = handle
            .offset
            .checked_add(data.len() as u32)
            .ok_or(TvmError::OutOfBounds)?;
        if end > self.meta.capacity {
            return Err(TvmError::OutOfBounds);
        }
        self.memory
            .write(store.as_context_mut(), handle.offset as usize, data)
            .map_err(|_| TvmError::OutOfBounds)
    }

    /// Pin: forbids spill/demote.
    pub fn pin(&mut self) -> Result<()> {
        if !self.meta.pinnable {
            return Err(TvmError::PolicyViolation);
        }
        self.meta.pinned = true;
        Ok(())
    }

    pub fn unpin(&mut self) {
        self.meta.pinned = false;
    }

    pub fn is_resident(&self) -> bool {
        matches!(self.meta.residency, Residency::Hot | Residency::Warm)
    }

    /// Bump generation — used by compaction. Old handles fail validation
    /// after this.
    pub fn bump_generation(&mut self) -> u16 {
        let mut next = self.meta.generation.wrapping_add(1);
        if next == 0 {
            next = 1;
        }
        self.meta.generation = next;
        next
    }

    pub fn memory(&self) -> Memory {
        self.memory
    }

    pub fn import_name(&self) -> String {
        format!("r{}", self.meta.id)
    }
}

/// One-call setup with pre-loaded payloads: build the engine/store/linker,
/// create N imported regions, allocate `payloads[i].len()` bytes in each,
/// write the payload, and return everything wired up.
///
/// Returns `(engine, store, linker, handles)` where `handles[i]` is a
/// validated handle to the bytes you just wrote. The caller instantiates
/// against the linker and calls into the guest.
///
/// Equivalent to the long-form pattern in
/// `bench-framework/runner/src/main.rs::run_tvm_unified_sequential` but in
/// one call.
pub fn build_imported_setup_with_data(
    payloads: &[&[u8]],
    kind: tvm_core::RegionKind,
    extra_capacity: u32,
) -> ImportedSetup<tvm_core::Handle> {
    let config = crate::engine_config::imported_region_engine_config();
    let engine = wasmtime::Engine::new(&config)?;
    let host = crate::TvmHost::new();
    let mut store = wasmtime::Store::new(&engine, host);

    let mut handles = Vec::with_capacity(payloads.len());
    for payload in payloads {
        let cap = (payload.len() as u32).saturating_add(extra_capacity);
        let region_id = create_imported_in_store(&mut store, kind, cap)?;
        let handle = store
            .data_mut()
            .imported_alloc(region_id, payload.len() as u32)?;
        let memory = store
            .data()
            .imported_region(region_id)
            .expect("just created")
            .memory();
        memory.write(&mut store, handle.offset as usize, payload)?;
        handles.push(handle);
    }

    // Wire imports into a linker.
    let imports: Vec<(String, wasmtime::Memory)> = store
        .data()
        .imported
        .iter()
        .map(|r| (r.import_name(), r.memory()))
        .collect();
    let mut linker = wasmtime::Linker::new(&engine);
    for (name, memory) in imports {
        linker.define(&mut store, "tvm", &name, memory)?;
    }
    Ok((engine, store, linker, handles))
}

/// One-call setup: build a multi-memory-enabled engine, create N imported
/// regions of equal capacity, register them in a linker, return the
/// configured `(engine, store, linker, region_ids)` tuple. The caller
/// instantiates against the linker.
///
/// This collapses the ~30 lines of borrow-checker dance in the bench
/// runner into one call. Use when the workload has a fixed number of
/// imported regions all of the same size and kind.
pub fn build_imported_setup(
    n_regions: u32,
    region_capacity: u32,
    kind: tvm_core::RegionKind,
) -> ImportedSetup<u16> {
    let config = crate::engine_config::imported_region_engine_config();
    let engine = wasmtime::Engine::new(&config)?;
    let host = crate::TvmHost::new();
    let mut store = wasmtime::Store::new(&engine, host);
    let mut ids = Vec::with_capacity(n_regions as usize);
    for _ in 0..n_regions {
        let id = create_imported_in_store(&mut store, kind, region_capacity)?;
        ids.push(id);
    }
    // Snapshot the (name, memory) pairs without holding a long borrow.
    let imports: Vec<(String, wasmtime::Memory)> = store
        .data()
        .imported
        .iter()
        .map(|r| (r.import_name(), r.memory()))
        .collect();
    let mut linker = wasmtime::Linker::new(&engine);
    for (name, memory) in imports {
        linker.define(&mut store, "tvm", &name, memory)?;
    }
    Ok((engine, store, linker, ids))
}

/// Convenience: create an imported region and register it in the host's
/// vector, dancing around wasmtime's borrow checker. The dance is
/// necessary because both the host (`store.data_mut()`) and the wasmtime
/// store context (`store.as_context_mut()`) borrow `store` exclusively;
/// we have to acquire each in disjoint scopes.
///
/// This is the canonical entry point. `TvmHost` does NOT have a
/// `create_imported_region` method that takes the store-context — adding
/// one would require wrappers or `unsafe` to satisfy the borrow checker.
pub fn create_imported_in_store(
    store: &mut wasmtime::Store<crate::TvmHost>,
    kind: tvm_core::RegionKind,
    capacity: u32,
) -> tvm_core::Result<u16> {
    use tvm_core::PlacementPolicy;
    use wasmtime::AsContextMut;
    let id = {
        let host = store.data_mut();
        let id = host.next_imported_id;
        host.next_imported_id = host
            .next_imported_id
            .checked_add(1)
            .ok_or(tvm_core::TvmError::AllocationFailed)?;
        id
    };
    let allocator = store.data().default_allocator;
    let region = {
        let mut ctx = store.as_context_mut();
        ImportedRegion::new(
            &mut ctx,
            id,
            kind,
            capacity,
            allocator,
            PlacementPolicy::for_kind(kind),
        )?
    };
    store.data_mut().imported.push(region);
    Ok(id)
}

// ── D2 Session 6 — WasmosImportedRegion (SharedMemory-backed) ───────
//
// Wasmos-native peer of [`ImportedRegion`]. Backs the region with a
// [`wasmos_runtime_api::SharedMemory`] handle instead of a
// wasmtime `Memory`, so the same allocator + metadata contract
// works on any wasmos-backed adapter (v48 / edge / WAMR).
//
// # Wiring contract
//
// The consumer creates the shared memory via
// [`wasmos_runtime_api::Runtime::create_shared_memory`] and passes
// the handle to [`WasmosImportedRegion::new`]. The consumer then
// wires the SAME handle into a [`wasmos_runtime_api::
// SharedMemoryImports`] composite so the guest imports it. The
// host + guest share the underlying allocation.
//
// # What's here vs future work
//
// This session lands the region struct + core operations (alloc,
// dealloc, read, write, pin, unpin, bump_generation, snapshot).
// The `build_wasmos_imported_setup*` helpers that mirror
// [`build_imported_setup`] / [`build_imported_setup_with_data`]
// are Session 7 — they need to coordinate an `Arc<dyn Runtime>` +
// `ExecutionContext` build path that this session doesn't touch.

/// One wasmos-backed imported memory region. Peer of
/// [`ImportedRegion`]; same allocator / metadata / handle
/// contract, but the memory is a [`SharedMemory`] handle
/// consumers create through the wasmos Runtime.
pub struct WasmosImportedRegion {
    pub meta: Region,
    pub memory: wasmos_runtime_api::SharedMemory,
    pub allocator: RegionAllocator,
    pub allocations: AtomicU64,
    pub bytes_allocated: AtomicU64,
}

impl WasmosImportedRegion {
    /// Wrap a caller-provided [`SharedMemory`] as an imported
    /// region. The `capacity` argument governs allocator bounds
    /// but does NOT influence the SharedMemory itself (which is
    /// sized at creation via [`wasmos_runtime_api::Runtime::
    /// create_shared_memory`]).
    pub fn new(
        memory: wasmos_runtime_api::SharedMemory,
        id: u16,
        kind: RegionKind,
        capacity: u32,
        allocator: AllocatorKind,
        policy: PlacementPolicy,
    ) -> Self {
        Self {
            meta: Region {
                id,
                generation: 1,
                kind,
                capacity,
                used: 0,
                residency: policy.initial_residency,
                pinned: false,
                pinnable: policy.pinnable,
                spillable: policy.spillable,
            },
            memory,
            allocator: RegionAllocator::new(allocator, capacity),
            allocations: AtomicU64::new(0),
            bytes_allocated: AtomicU64::new(0),
        }
    }

    /// Allocate `size` bytes within this region.
    pub fn alloc(&mut self, size: u32) -> Result<Handle> {
        let offset = self.allocator.alloc(size, 1)?;
        self.meta.used = self.allocator.used();
        self.allocations.fetch_add(1, Ordering::Relaxed);
        self.bytes_allocated.fetch_add(size as u64, Ordering::Relaxed);
        Ok(Handle {
            region_id: self.meta.id,
            generation: self.meta.generation,
            offset,
        })
    }

    /// Free a previously-allocated handle. Validates generation;
    /// bump allocator returns success without freeing (matching
    /// host-region semantics).
    pub fn dealloc(&mut self, handle: Handle) -> Result<()> {
        if handle.generation != self.meta.generation {
            return Err(TvmError::StaleHandle);
        }
        self.allocator.dealloc(handle.offset)?;
        self.meta.used = self.allocator.used();
        Ok(())
    }

    /// Read bytes from the shared memory through the host-mediated
    /// path. No ctx — SharedMemory carries its own state.
    pub fn read(&self, handle: Handle, buf: &mut [u8]) -> Result<()> {
        if handle.generation != self.meta.generation {
            return Err(TvmError::StaleHandle);
        }
        let end = handle
            .offset
            .checked_add(buf.len() as u32)
            .ok_or(TvmError::OutOfBounds)?;
        if end > self.meta.capacity {
            return Err(TvmError::OutOfBounds);
        }
        let bytes = self
            .memory
            .read(handle.offset as usize, buf.len())
            .map_err(|_| TvmError::OutOfBounds)?;
        if bytes.len() != buf.len() {
            return Err(TvmError::OutOfBounds);
        }
        buf.copy_from_slice(&bytes);
        Ok(())
    }

    /// Write bytes into the shared memory.
    pub fn write(&self, handle: Handle, data: &[u8]) -> Result<()> {
        if handle.generation != self.meta.generation {
            return Err(TvmError::StaleHandle);
        }
        let end = handle
            .offset
            .checked_add(data.len() as u32)
            .ok_or(TvmError::OutOfBounds)?;
        if end > self.meta.capacity {
            return Err(TvmError::OutOfBounds);
        }
        self.memory
            .write(handle.offset as usize, data)
            .map_err(|_| TvmError::OutOfBounds)
    }

    /// Pin: forbids spill/demote.
    pub fn pin(&mut self) -> Result<()> {
        if !self.meta.pinnable {
            return Err(TvmError::PolicyViolation);
        }
        self.meta.pinned = true;
        Ok(())
    }

    pub fn unpin(&mut self) {
        self.meta.pinned = false;
    }

    pub fn is_resident(&self) -> bool {
        matches!(self.meta.residency, Residency::Hot | Residency::Warm)
    }

    /// Bump generation — used by compaction. Old handles fail
    /// validation after this.
    pub fn bump_generation(&mut self) -> u16 {
        let mut next = self.meta.generation.wrapping_add(1);
        if next == 0 {
            next = 1;
        }
        self.meta.generation = next;
        next
    }

    /// Return a fresh clone of the underlying shared-memory
    /// handle. Cheap; shares the same allocation.
    pub fn shared_memory(&self) -> wasmos_runtime_api::SharedMemory {
        self.memory.clone_handle()
    }

    /// The guest-facing import name (`r<id>`), matching
    /// [`ImportedRegion::import_name`] — consumers pass this to
    /// [`wasmos_runtime_api::SharedMemoryImports::register`]
    /// alongside the shared-memory handle.
    pub fn import_name(&self) -> String {
        format!("r{}", self.meta.id)
    }
}

// ── D2 Session 7 — build_wasmos_imported_setup helper ───────────────
//
// Peer of [`build_imported_setup`] on the wasmos side. Takes an
// `Arc<dyn Runtime>` the caller supplies (tvm-wasmtime doesn't
// pick which adapter — that stays a caller concern per ADR-0029)
// and returns a fully wired setup: N shared memories allocated
// via `Runtime::create_shared_memory`, wrapped in
// `WasmosImportedRegion` instances, and paired with a
// `SharedMemoryImports` composite naming each region as the
// wit-bindgen `("tvm", "r<id>")` shape.
//
// The returned tuple mirrors [`build_imported_setup`] but with
// wasmos abstractions instead of wasmtime:
//   `(ExecutionContext, TvmHost, Vec<u16>)` instead of
//   `(wasmtime::Engine, wasmtime::Store<TvmHost>,
//    wasmtime::Linker<TvmHost>, Vec<u16>)`.
//
// The caller instantiates a module against the returned
// ExecutionContext via `runtime.instantiate_module(&compiled, ctx)`.

/// Setup output for the wasmos-native imported-region path.
pub struct WasmosImportedSetup {
    /// Fully-populated ExecutionContext with the shared-memory
    /// imports naming each region as `("tvm", "r<id>")`. The
    /// caller adds its own `host_imports` (via
    /// [`ExecutionContext::with_host_imports`]) before threading
    /// through `Runtime::instantiate_module`.
    pub context: wasmos_runtime_api::ExecutionContext,
    /// A [`crate::TvmHost`] with the newly-created imported regions
    /// registered in its `wasmos_imported` vec. Callers use the
    /// returned `region_ids` to index this host's regions for
    /// alloc / read / write.
    pub host: crate::TvmHost,
    /// The region ids the caller uses to reference each region on
    /// the host side (matches the `"r<id>"` guest-import names).
    pub region_ids: Vec<u16>,
}

/// Async peer of [`build_imported_setup`]. See module docstring
/// for the shape.
///
/// # Arguments
///
/// - `runtime`: consumer-provided wasmos adapter that supports
///   the `wasmos:shared-memory` capability. Adapters that don't
///   surface [`wasmos_runtime_api::RuntimeError::CapabilityUnavailable`].
/// - `n_regions`: how many imported regions to allocate.
/// - `region_capacity`: bytes per region. Rounded up to the next
///   wasm page (64 KiB) by the adapter.
/// - `kind`: TVM region kind (informs allocator + placement policy).
///
/// Every intermediate error surfaces as
/// [`wasmos_runtime_api::RuntimeError`]; the caller wraps into its
/// own error type at the boundary.
pub async fn build_wasmos_imported_setup(
    runtime: &dyn wasmos_runtime_api::Runtime,
    n_regions: u32,
    region_capacity: u32,
    kind: RegionKind,
) -> wasmos_runtime_api::RuntimeResult<WasmosImportedSetup> {
    use wasmos_runtime_api::{ExecutionContext, SharedMemoryDescriptor, SharedMemoryImports};

    const PAGE: u32 = 64 * 1024;

    let mut host = crate::TvmHost::new();
    let mut region_ids = Vec::with_capacity(n_regions as usize);
    let mut imports = SharedMemoryImports::new();

    for _ in 0..n_regions {
        // Reserve a fresh id on the host — matches the
        // [`create_imported_in_store`] convention above so
        // wit-bindgen + wasmos regions share id space.
        let id = host.next_imported_id;
        host.next_imported_id = host.next_imported_id.checked_add(1).ok_or_else(|| {
            wasmos_runtime_api::RuntimeError::msg(
                "build_wasmos_imported_setup: imported-region id space exhausted",
            )
        })?;

        // Ask the adapter for a shared memory sized to fit
        // `region_capacity` — round up to whole wasm pages.
        let min_pages = region_capacity.div_ceil(PAGE).max(1);
        let descriptor = SharedMemoryDescriptor::new(min_pages, min_pages);
        let handle = runtime.create_shared_memory(descriptor).await?;

        // Wrap in a WasmosImportedRegion. Allocator picks up
        // from the host's default (matches wit-bindgen path).
        let region = WasmosImportedRegion::new(
            handle.clone_handle(),
            id,
            kind,
            region_capacity,
            host.default_allocator,
            PlacementPolicy::for_kind(kind),
        );
        host.wasmos_imported.push(region);
        imports = imports.register("tvm", format!("r{id}"), handle);
        region_ids.push(id);
    }

    let context = ExecutionContext::new().with_shared_memory_imports(imports);
    Ok(WasmosImportedSetup { context, host, region_ids })
}

/// One-call setup with pre-loaded payloads. Async peer of
/// [`build_imported_setup_with_data`]. Alloc + write each payload
/// through the host-side [`WasmosImportedRegion::alloc`] +
/// [`WasmosImportedRegion::write`] path — the same memory is
/// visible to the guest through the returned
/// [`WasmosImportedSetup::context`].
pub async fn build_wasmos_imported_setup_with_data(
    runtime: &dyn wasmos_runtime_api::Runtime,
    payloads: &[&[u8]],
    kind: RegionKind,
    extra_capacity: u32,
) -> wasmos_runtime_api::RuntimeResult<(WasmosImportedSetup, Vec<Handle>)> {
    let mut capacities = Vec::with_capacity(payloads.len());
    for p in payloads {
        capacities.push((p.len() as u32).saturating_add(extra_capacity));
    }
    // Take the max capacity as the region size — build_imported_setup
    // uses per-region capacity but the shared-memory descriptor
    // benefits from uniform pages when they end up in the same
    // adapter pool.
    let region_capacity = capacities.iter().copied().max().unwrap_or(0);

    let mut setup =
        build_wasmos_imported_setup(runtime, payloads.len() as u32, region_capacity, kind).await?;

    let mut handles = Vec::with_capacity(payloads.len());
    for (i, payload) in payloads.iter().enumerate() {
        let region = &mut setup.host.wasmos_imported[i];
        let h = region
            .alloc(payload.len() as u32)
            .map_err(|e| wasmos_runtime_api::RuntimeError::msg(format!("alloc: {e:?}")))?;
        region
            .write(h, payload)
            .map_err(|e| wasmos_runtime_api::RuntimeError::msg(format!("write: {e:?}")))?;
        handles.push(h);
    }
    Ok((setup, handles))
}

#[cfg(test)]
mod wasmos_imported_tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use wasmos_runtime_api::{SharedMemory, SharedMemoryImpl};

    const PAGE: usize = 64 * 1024;

    struct FakeShared {
        buf: Mutex<Vec<u8>>,
    }
    impl SharedMemoryImpl for FakeShared {
        fn size_pages(&self) -> u64 {
            (self.buf.lock().unwrap().len() / PAGE) as u64
        }
        fn data_size_bytes(&self) -> usize {
            self.buf.lock().unwrap().len()
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn read(&self, off: usize, len: usize) -> wasmos_runtime_api::RuntimeResult<Vec<u8>> {
            let b = self.buf.lock().unwrap();
            let end = off.saturating_add(len);
            if end > b.len() {
                return Err(wasmos_runtime_api::RuntimeError::msg("oob"));
            }
            Ok(b[off..end].to_vec())
        }
        fn write(&self, off: usize, bytes: &[u8]) -> wasmos_runtime_api::RuntimeResult<()> {
            let mut b = self.buf.lock().unwrap();
            let end = off.saturating_add(bytes.len());
            if end > b.len() {
                return Err(wasmos_runtime_api::RuntimeError::msg("oob"));
            }
            b[off..end].copy_from_slice(bytes);
            Ok(())
        }
    }

    fn shared(bytes: usize) -> SharedMemory {
        SharedMemory::from_impl(Arc::new(FakeShared {
            buf: Mutex::new(vec![0u8; bytes]),
        }))
    }

    fn region(capacity: u32) -> WasmosImportedRegion {
        WasmosImportedRegion::new(
            shared(PAGE),
            0,
            RegionKind::HotHeap,
            capacity,
            AllocatorKind::Bump,
            PlacementPolicy::for_kind(RegionKind::HotHeap),
        )
    }

    #[test]
    fn wasmos_imported_alloc_yields_handle_with_current_generation() {
        let mut r = region(1024);
        let h = r.alloc(64).expect("alloc");
        assert_eq!(h.region_id, 0);
        assert_eq!(h.generation, 1);
        assert_eq!(h.offset, 0);
    }

    #[test]
    fn wasmos_imported_read_write_round_trip() {
        let mut r = region(1024);
        let h = r.alloc(8).unwrap();
        r.write(h, &[1, 2, 3, 4, 5, 6, 7, 8]).expect("write");
        let mut buf = [0u8; 8];
        r.read(h, &mut buf).expect("read");
        assert_eq!(buf, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn wasmos_imported_stale_generation_errors() {
        let mut r = region(1024);
        let old = r.alloc(4).unwrap();
        r.bump_generation();
        let err = r.write(old, &[9; 4]).unwrap_err();
        assert!(matches!(err, TvmError::StaleHandle));
    }

    #[test]
    fn wasmos_imported_out_of_bounds_read_errors() {
        let r = region(64);
        // Handle beyond capacity even though the shared memory is
        // sized to a full page.
        let h = Handle { region_id: 0, generation: 1, offset: 100 };
        let mut buf = [0u8; 4];
        let err = r.read(h, &mut buf).unwrap_err();
        assert!(matches!(err, TvmError::OutOfBounds));
    }

    #[test]
    fn wasmos_imported_import_name_matches_r_id() {
        let mut r = region(128);
        r.meta.id = 7;
        assert_eq!(r.import_name(), "r7");
    }

    #[test]
    fn wasmos_imported_shared_memory_clone_shares_backing() {
        let r = region(128);
        let a = r.shared_memory();
        let b = r.shared_memory();
        assert_eq!(a.data_size_bytes(), b.data_size_bytes());
    }

    // ── D2 Session 7 — build_wasmos_imported_setup tests ───────

    use async_trait::async_trait;
    use wasmos_runtime_api::{
        CompileOptions, CompiledComponent, CompiledModule, ComponentSource, ExecutionContext,
        Instance, ModuleInstance, Runtime, RuntimeMetadata, RuntimeResult, SharedMemoryDescriptor,
    };

    /// Stub Runtime that only implements create_shared_memory +
    /// metadata — enough to exercise build_wasmos_imported_setup
    /// without a real adapter. Other methods are unreachable
    /// under this test path.
    struct StubRuntime {
        meta: RuntimeMetadata,
    }

    impl StubRuntime {
        fn new() -> Self {
            Self {
                meta: RuntimeMetadata {
                    api_version: wasmos_runtime_api::RuntimeApiVersion::new(1),
                    implementation: "stub".into(),
                    implementation_version: "0.0.0".into(),
                    channel: wasmos_runtime_api::RuntimeChannel::Edge,
                    capabilities: wasmos_runtime_api::RuntimeCapabilities::default(),
                    adapter: "tvm-wasmtime-test-stub".into(),
                },
            }
        }
    }

    #[async_trait]
    impl Runtime for StubRuntime {
        fn metadata(&self) -> &RuntimeMetadata {
            &self.meta
        }
        async fn compile_component(
            &self,
            _source: ComponentSource,
            _options: CompileOptions,
        ) -> RuntimeResult<CompiledComponent> {
            unreachable!("stub does not compile components")
        }
        async fn instantiate(
            &self,
            _component: &CompiledComponent,
            _context: ExecutionContext,
        ) -> RuntimeResult<Instance> {
            unreachable!()
        }
        async fn compile_module(
            &self,
            _source: ComponentSource,
            _options: CompileOptions,
        ) -> RuntimeResult<CompiledModule> {
            unreachable!()
        }
        async fn instantiate_module(
            &self,
            _module: &CompiledModule,
            _context: ExecutionContext,
        ) -> RuntimeResult<ModuleInstance> {
            unreachable!()
        }
        async fn create_shared_memory(
            &self,
            descriptor: SharedMemoryDescriptor,
        ) -> RuntimeResult<wasmos_runtime_api::SharedMemory> {
            // Return a Vec-backed FakeShared sized to descriptor.min_pages.
            Ok(wasmos_runtime_api::SharedMemory::from_impl(Arc::new(FakeShared {
                buf: Mutex::new(vec![0u8; descriptor.min_pages as usize * PAGE]),
            })))
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn build_wasmos_imported_setup_allocates_n_regions_and_populates_imports() {
        let rt = StubRuntime::new();
        let setup = super::build_wasmos_imported_setup(&rt, 3, 4096, RegionKind::HotHeap)
            .await
            .expect("build");
        assert_eq!(setup.region_ids, vec![0, 1, 2]);
        assert_eq!(setup.host.wasmos_imported.len(), 3);
        // ExecutionContext carries the three shared-memory imports
        // under names ("tvm", "r0") / ("tvm", "r1") / ("tvm", "r2").
        assert!(setup.context.shared_memory_imports.get("tvm", "r0").is_some());
        assert!(setup.context.shared_memory_imports.get("tvm", "r1").is_some());
        assert!(setup.context.shared_memory_imports.get("tvm", "r2").is_some());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn build_wasmos_imported_setup_with_data_preloads_bytes() {
        let rt = StubRuntime::new();
        let (setup, handles) = super::build_wasmos_imported_setup_with_data(
            &rt,
            &[b"hello", b"world!"],
            RegionKind::HotHeap,
            /* extra_capacity */ 16,
        )
        .await
        .expect("build");

        assert_eq!(setup.region_ids, vec![0, 1]);
        assert_eq!(handles.len(), 2);

        // Read back through the host-side WasmosImportedRegion; the
        // ExecutionContext's SharedMemoryImports carries clones of
        // the SAME shared memories, so the guest (if instantiated)
        // would see the same bytes.
        let mut buf0 = [0u8; 5];
        setup.host.wasmos_imported[0].read(handles[0], &mut buf0).unwrap();
        assert_eq!(&buf0, b"hello");
        let mut buf1 = [0u8; 6];
        setup.host.wasmos_imported[1].read(handles[1], &mut buf1).unwrap();
        assert_eq!(&buf1, b"world!");
    }
}
