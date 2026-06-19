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
