use std::sync::{Arc, RwLock};

use crate::allocator::AllocatorKind;
use crate::backing::BackingStore;
use crate::directory::{MemoryRegion, RegionDirectory};
use crate::error::{Result, TvmError};
use crate::handle::Handle;
use crate::region::{Region, RegionKind};

/// A thread-safe wrapper around `RegionDirectory`.
///
/// Cheap to clone (`Arc` semantics). Reads take a read lock; writes take a
/// write lock. Handles are `Copy` and may be passed across threads freely; the
/// directory itself synchronises access.
pub struct SharedDirectory<M: MemoryRegion + Send + Sync> {
    inner: Arc<RwLock<RegionDirectory<M>>>,
}

impl<M: MemoryRegion + Send + Sync> Clone for SharedDirectory<M> {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}

impl<M: MemoryRegion + Send + Sync> Default for SharedDirectory<M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M: MemoryRegion + Send + Sync> SharedDirectory<M> {
    pub fn new() -> Self {
        Self { inner: Arc::new(RwLock::new(RegionDirectory::new())) }
    }

    pub fn create_region(
        &self,
        kind: RegionKind,
        capacity: u32,
        memory: M,
    ) -> Result<u16> {
        self.with_write(|d| d.create_region(kind, capacity, memory))
    }

    pub fn create_region_with(
        &self,
        kind: RegionKind,
        capacity: u32,
        allocator: AllocatorKind,
        memory: M,
    ) -> Result<u16> {
        self.with_write(|d| d.create_region_with(kind, capacity, allocator, memory))
    }

    pub fn alloc(&self, region_id: u16, size: u32) -> Result<Handle> {
        self.with_write(|d| d.alloc(region_id, size))
    }

    pub fn dealloc(&self, handle: Handle) -> Result<()> {
        self.with_write(|d| d.dealloc(handle))
    }

    pub fn read(&self, handle: Handle, buf: &mut [u8]) -> Result<()> {
        self.with_read(|d| d.read(handle, buf))
    }

    pub fn write(&self, handle: Handle, data: &[u8]) -> Result<()> {
        self.with_write(|d| d.write(handle, data))
    }

    pub fn pin(&self, region_id: u16) -> Result<()> {
        self.with_write(|d| d.pin(region_id))
    }

    pub fn unpin(&self, region_id: u16) -> Result<()> {
        self.with_write(|d| d.unpin(region_id))
    }

    pub fn spill_region<B: BackingStore>(
        &self,
        region_id: u16,
        store: &mut B,
    ) -> Result<()> {
        self.with_write(|d| d.spill_region(region_id, store))
    }

    pub fn load_region<B: BackingStore>(
        &self,
        region_id: u16,
        store: &mut B,
    ) -> Result<()> {
        self.with_write(|d| d.load_region(region_id, store))
    }

    pub fn region_info(&self, region_id: u16) -> Result<Region> {
        self.with_read(|d| d.region_info(region_id).copied())
    }

    pub fn list_regions(&self) -> Result<Vec<Region>> {
        self.with_read(|d| Ok(d.iter().copied().collect()))
    }

    fn with_read<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&RegionDirectory<M>) -> Result<R>,
    {
        let guard = self
            .inner
            .read()
            .map_err(|_| TvmError::BackingStore("directory lock poisoned".into()))?;
        f(&guard)
    }

    fn with_write<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&mut RegionDirectory<M>) -> Result<R>,
    {
        let mut guard = self
            .inner
            .write()
            .map_err(|_| TvmError::BackingStore("directory lock poisoned".into()))?;
        f(&mut guard)
    }
}
