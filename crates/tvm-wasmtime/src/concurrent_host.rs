//! `ConcurrentTvmHost` — multi-store host backed by per-region locking.
//!
//! Compared to [`SharedTvmHost`](crate::SharedTvmHost), which serializes
//! every host call through a single mutex, `ConcurrentTvmHost` lets
//! operations on **different regions** run concurrently. Operations on the
//! same region still serialize through that region's lock, which is the
//! correct safety boundary.
//!
//! Use this when:
//! - You have multiple wasmtime stores on different threads.
//! - The workload mostly hits **distinct** regions per call (which is the
//!   common case — e.g. each guest gets its own arena).
//!
//! Stay with `SharedTvmHost` for simpler single-store setups; the simpler
//! lock and cache are slightly cheaper under no contention.

use std::sync::{Arc, Mutex};

use tvm_core::{
    AllocatorKind, ConcurrentDirectory, FileBackingStore, Handle as CoreHandle,
    RegionKind as CoreRegionKind, Residency as CoreResidency, ResolveCache, TvmError as CoreError,
    VecBackedRegion,
};

use crate::bindings::tvm::memory::bytes::Host as BytesHost;
use crate::bindings::tvm::memory::diagnostics::Host as DiagnosticsHost;
use crate::bindings::tvm::memory::manager::Host as ManagerHost;
use crate::bindings::tvm::memory::types::{
    CompactResult, Handle, Host as TypesHost, RegionInfo, RegionKind, RegionMetrics, Residency,
    TvmError,
};

#[derive(Clone)]
pub struct ConcurrentTvmHost {
    inner: Arc<Inner>,
}

struct Inner {
    directory: ConcurrentDirectory<VecBackedRegion>,
    backing: Mutex<Option<FileBackingStore>>,
    cache: Mutex<ResolveCache>,
    default_allocator: Mutex<AllocatorKind>,
}

impl Default for ConcurrentTvmHost {
    fn default() -> Self {
        Self::new()
    }
}

impl ConcurrentTvmHost {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Inner {
                directory: ConcurrentDirectory::new(),
                backing: Mutex::new(None),
                cache: Mutex::new(ResolveCache::new()),
                default_allocator: Mutex::new(AllocatorKind::Bump),
            }),
        }
    }

    pub fn with_backing(path: impl AsRef<std::path::Path>) -> Result<Self, CoreError> {
        let host = Self::new();
        let backing = FileBackingStore::new(path.as_ref().to_path_buf())?;
        *host.inner.backing.lock().unwrap() = Some(backing);
        Ok(host)
    }

    pub fn with_allocator(self, allocator: AllocatorKind) -> Self {
        *self.inner.default_allocator.lock().unwrap() = allocator;
        self
    }

    pub fn directory(&self) -> &ConcurrentDirectory<VecBackedRegion> {
        &self.inner.directory
    }
}

impl AsMut<ConcurrentTvmHost> for ConcurrentTvmHost {
    fn as_mut(&mut self) -> &mut ConcurrentTvmHost {
        self
    }
}

fn err(e: CoreError) -> TvmError {
    match e {
        CoreError::RegionNotFound(id) => TvmError::RegionNotFound(id),
        CoreError::StaleHandle => TvmError::StaleHandle,
        CoreError::OutOfBounds => TvmError::OutOfBounds,
        CoreError::NotResident => TvmError::NotResident,
        CoreError::AllocationFailed => TvmError::AllocationFailed,
        CoreError::BackingStore(s) => TvmError::BackingStore(s),
        CoreError::Pinned => TvmError::Pinned,
        CoreError::UnsupportedAllocator => {
            TvmError::BackingStore("unsupported by allocator".into())
        }
        CoreError::PolicyViolation => TvmError::BackingStore("forbidden by region policy".into()),
    }
}

fn to_core_kind(k: RegionKind) -> CoreRegionKind {
    match k {
        RegionKind::HotHeap => CoreRegionKind::HotHeap,
        RegionKind::ObjectArena => CoreRegionKind::ObjectArena,
        RegionKind::BlobArena => CoreRegionKind::BlobArena,
        RegionKind::PageStore => CoreRegionKind::PageStore,
        RegionKind::Scratch => CoreRegionKind::Scratch,
        RegionKind::DeviceState => CoreRegionKind::DeviceState,
        RegionKind::CodeCache => CoreRegionKind::CodeCache,
    }
}

fn from_core_kind(k: CoreRegionKind) -> RegionKind {
    match k {
        CoreRegionKind::HotHeap => RegionKind::HotHeap,
        CoreRegionKind::ObjectArena => RegionKind::ObjectArena,
        CoreRegionKind::BlobArena => RegionKind::BlobArena,
        CoreRegionKind::PageStore => RegionKind::PageStore,
        CoreRegionKind::Scratch => RegionKind::Scratch,
        CoreRegionKind::DeviceState => RegionKind::DeviceState,
        CoreRegionKind::CodeCache => RegionKind::CodeCache,
    }
}

fn from_core_residency(r: CoreResidency) -> Residency {
    match r {
        CoreResidency::Hot => Residency::Hot,
        CoreResidency::Warm => Residency::Warm,
        CoreResidency::Cold => Residency::Cold,
        CoreResidency::External => Residency::External,
    }
}

fn to_core_handle(h: Handle) -> CoreHandle {
    CoreHandle {
        region_id: h.region_id,
        generation: h.generation,
        offset: h.offset,
    }
}

fn to_wit_handle(h: CoreHandle) -> Handle {
    Handle {
        region_id: h.region_id,
        generation: h.generation,
        offset: h.offset,
    }
}

impl TypesHost for ConcurrentTvmHost {}

impl ManagerHost for ConcurrentTvmHost {
    fn create_region(&mut self, kind: RegionKind, capacity: u32) -> Result<u16, TvmError> {
        let allocator = *self.inner.default_allocator.lock().unwrap();
        self.inner
            .directory
            .create_region_with(
                to_core_kind(kind),
                capacity,
                allocator,
                VecBackedRegion::new(capacity),
            )
            .map_err(err)
    }

    fn destroy_region(&mut self, region_id: u16) -> Result<(), TvmError> {
        self.inner.cache.lock().unwrap().invalidate(region_id);
        self.inner.directory.destroy_region(region_id).map_err(err)
    }

    fn alloc(&mut self, region_id: u16, size: u32) -> Result<Handle, TvmError> {
        self.inner
            .directory
            .alloc(region_id, size)
            .map(to_wit_handle)
            .map_err(err)
    }

    fn dealloc(&mut self, ptr: Handle) -> Result<(), TvmError> {
        self.inner
            .directory
            .dealloc(to_core_handle(ptr))
            .map_err(err)
    }

    fn describe_region(&mut self, region_id: u16) -> Result<RegionInfo, TvmError> {
        let info = self.inner.directory.region_info(region_id).map_err(err)?;
        Ok(RegionInfo {
            id: info.id,
            generation: info.generation,
            kind: from_core_kind(info.kind),
            capacity: info.capacity,
            used: info.used,
            residency: from_core_residency(info.residency),
        })
    }

    fn promote_region(&mut self, region_id: u16) -> Result<(), TvmError> {
        // Concurrent promote/demote are not implemented; fall back to load.
        self.inner.cache.lock().unwrap().invalidate(region_id);
        let mut backing_guard = self.inner.backing.lock().unwrap();
        let backing = backing_guard
            .as_mut()
            .ok_or_else(|| TvmError::BackingStore("no backing store configured".into()))?;
        self.inner
            .directory
            .load_region(region_id, backing)
            .map_err(err)
    }

    fn demote_region(&mut self, region_id: u16) -> Result<(), TvmError> {
        self.inner.cache.lock().unwrap().invalidate(region_id);
        let mut backing_guard = self.inner.backing.lock().unwrap();
        let backing = backing_guard
            .as_mut()
            .ok_or_else(|| TvmError::BackingStore("no backing store configured".into()))?;
        self.inner
            .directory
            .spill_region(region_id, backing)
            .map_err(err)
    }

    fn spill_region(&mut self, region_id: u16) -> Result<(), TvmError> {
        self.inner.cache.lock().unwrap().invalidate(region_id);
        let mut backing_guard = self.inner.backing.lock().unwrap();
        let backing = backing_guard
            .as_mut()
            .ok_or_else(|| TvmError::BackingStore("no backing store configured".into()))?;
        self.inner
            .directory
            .spill_region(region_id, backing)
            .map_err(err)
    }

    fn load_region(&mut self, region_id: u16) -> Result<(), TvmError> {
        self.inner.cache.lock().unwrap().invalidate(region_id);
        let mut backing_guard = self.inner.backing.lock().unwrap();
        let backing = backing_guard
            .as_mut()
            .ok_or_else(|| TvmError::BackingStore("no backing store configured".into()))?;
        self.inner
            .directory
            .load_region(region_id, backing)
            .map_err(err)
    }

    fn pin(&mut self, region_id: u16) -> Result<(), TvmError> {
        self.inner.directory.pin(region_id).map_err(err)
    }

    fn unpin(&mut self, region_id: u16) -> Result<(), TvmError> {
        self.inner.directory.unpin(region_id).map_err(err)
    }

    fn compact_region(&mut self, region_id: u16) -> Result<CompactResult, TvmError> {
        self.inner.cache.lock().unwrap().invalidate(region_id);
        let remap = self
            .inner
            .directory
            .compact_region(region_id)
            .map_err(err)?;
        let mut mapping: Vec<(u32, u32)> = remap.mapping.into_iter().collect();
        mapping.sort_by_key(|p| p.0);
        Ok(CompactResult {
            old_generation: remap.old_generation,
            new_generation: remap.new_generation,
            mapping,
        })
    }
}

impl BytesHost for ConcurrentTvmHost {
    fn read(&mut self, ptr: Handle, len: u32) -> Result<Vec<u8>, TvmError> {
        let mut buf = vec![0u8; len as usize];
        self.inner
            .directory
            .read(to_core_handle(ptr), &mut buf)
            .map_err(err)?;
        Ok(buf)
    }

    fn write(&mut self, ptr: Handle, data: Vec<u8>) -> Result<(), TvmError> {
        self.inner
            .directory
            .write(to_core_handle(ptr), &data)
            .map_err(err)
    }

    fn copy(&mut self, src: Handle, dst: Handle, len: u32) -> Result<(), TvmError> {
        self.inner
            .directory
            .cross_region_copy(src.region_id, src.offset, dst.region_id, dst.offset, len)
            .map_err(err)
    }

    fn read_into(
        &mut self,
        src: Handle,
        dst_region: u16,
        dst_offset: u32,
        len: u32,
    ) -> Result<(), TvmError> {
        self.inner
            .directory
            .cross_region_copy(src.region_id, src.offset, dst_region, dst_offset, len)
            .map_err(err)
    }

    fn write_from(
        &mut self,
        src_region: u16,
        src_offset: u32,
        dst: Handle,
        len: u32,
    ) -> Result<(), TvmError> {
        self.inner
            .directory
            .cross_region_copy(src_region, src_offset, dst.region_id, dst.offset, len)
            .map_err(err)
    }

    fn copy_region(
        &mut self,
        src_region: u16,
        src_offset: u32,
        dst_region: u16,
        dst_offset: u32,
        len: u32,
    ) -> Result<(), TvmError> {
        self.inner
            .directory
            .cross_region_copy(src_region, src_offset, dst_region, dst_offset, len)
            .map_err(err)
    }
}

impl DiagnosticsHost for ConcurrentTvmHost {
    fn list_regions(&mut self) -> Vec<RegionInfo> {
        self.inner
            .directory
            .list_regions()
            .unwrap_or_default()
            .into_iter()
            .map(|info| RegionInfo {
                id: info.id,
                generation: info.generation,
                kind: from_core_kind(info.kind),
                capacity: info.capacity,
                used: info.used,
                residency: from_core_residency(info.residency),
            })
            .collect()
    }

    fn fault_count(&mut self, region_id: u16) -> u64 {
        self.inner
            .directory
            .metrics_snapshot(region_id)
            .map(|m| m.faults)
            .unwrap_or(0)
    }

    fn allocation_count(&mut self, region_id: u16) -> u64 {
        self.inner
            .directory
            .metrics_snapshot(region_id)
            .map(|m| m.allocations)
            .unwrap_or(0)
    }

    fn bytes_read_count(&mut self, region_id: u16) -> u64 {
        self.inner
            .directory
            .metrics_snapshot(region_id)
            .map(|m| m.bytes_read)
            .unwrap_or(0)
    }

    fn bytes_written_count(&mut self, region_id: u16) -> u64 {
        self.inner
            .directory
            .metrics_snapshot(region_id)
            .map(|m| m.bytes_written)
            .unwrap_or(0)
    }

    fn metrics_snapshot(&mut self, region_id: u16) -> Result<RegionMetrics, TvmError> {
        let snap = self
            .inner
            .directory
            .metrics_snapshot(region_id)
            .map_err(err)?;
        Ok(RegionMetrics {
            allocations: snap.allocations,
            bytes_allocated: snap.bytes_allocated,
            bytes_read: snap.bytes_read,
            bytes_written: snap.bytes_written,
            faults: snap.faults,
            promotions: snap.promotions,
            demotions: snap.demotions,
        })
    }
}
