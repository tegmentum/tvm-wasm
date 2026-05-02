//! Specialized fast-path slice access for `RegionDirectory<VecBackedRegion>`.
//!
//! Used by the wasmtime raw linker to do **zero-copy reads/writes**: the
//! host can copy bytes directly between the region and guest memory
//! without an intermediate scratch buffer. The highest-leverage perf
//! optimization short of multi-memory imports.
//!
//! Specialized to `VecBackedRegion` because the slice access requires
//! raw bytes; runtime-bound backends (wasmtime memories) need a context
//! and don't fit the same shape.

use crate::backing::VecBackedRegion;
use crate::directory::RegionDirectory;
use crate::error::{Result, TvmError};
use crate::handle::Handle;

impl RegionDirectory<VecBackedRegion> {
    /// Returns the raw pointer + length for a region's underlying bytes.
    /// Used by `TvmHost::fast_read` / `fast_write` to populate the
    /// resolve cache. Pointer is stable until the region's memory is
    /// replaced (spill/load/compact/destroy).
    pub fn region_data_raw(&self, region_id: u16) -> Result<(usize, u32)> {
        let entry = self.entry(region_id)?;
        let memory = entry.memory.as_ref().ok_or(TvmError::NotResident)?;
        Ok((memory.as_slice().as_ptr() as usize, entry.meta.capacity))
    }

    /// Validates `handle`, bounds-checks `len`, returns an immutable slice
    /// into the region's bytes. The slice is valid as long as the
    /// directory isn't mutated.
    pub fn region_slice_at(&self, handle: Handle, len: u32) -> Result<&[u8]> {
        let entry = self.validate(handle)?;
        let memory = entry.memory.as_ref().ok_or(TvmError::NotResident)?;
        let start = handle.offset as usize;
        let end = start
            .checked_add(len as usize)
            .ok_or(TvmError::OutOfBounds)?;
        if end > entry.meta.capacity as usize {
            return Err(TvmError::OutOfBounds);
        }
        entry.metrics.record_read(len as u64);
        Ok(&memory.as_slice()[start..end])
    }

    /// Same as `region_slice_at` but mutable. The slice is valid as long
    /// as the directory isn't mutated.
    pub fn region_slice_mut_at(
        &mut self,
        handle: Handle,
        len: u32,
    ) -> Result<&mut [u8]> {
        let entry = self.validate_mut(handle)?;
        let start = handle.offset as usize;
        let end = start
            .checked_add(len as usize)
            .ok_or(TvmError::OutOfBounds)?;
        if end > entry.meta.capacity as usize {
            return Err(TvmError::OutOfBounds);
        }
        entry.metrics.record_write(len as u64);
        let memory = entry.memory.as_mut().ok_or(TvmError::NotResident)?;
        Ok(&mut memory.data_mut()[start..end])
    }
}
