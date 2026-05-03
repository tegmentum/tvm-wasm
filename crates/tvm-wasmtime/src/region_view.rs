//! `RegionView` — a host-side facade shaped like `wasmtime::Memory`
//! for code that wants familiar wasm-memory ergonomics over a TVM
//! region.
//!
//! This is a **near-zero-cost ergonomic skin**. Each method maps to
//! one underlying TVM operation (slice access, read, write, copy
//! within, fill); no extra allocations or copies are introduced. The
//! goal is to let embedders coming from `wasmtime::Memory` write
//! TVM code without learning a new shape.
//!
//! ## What's intentionally not here
//!
//! - **`new(...)`** / **`grow(...)`**: TVM regions have a `RegionKind`
//!   (placement policy) and a fixed `capacity` set at create time.
//!   Use [`TvmHost::create_region`] to construct; use compaction or
//!   migration to a larger region when capacity is exceeded.
//! - **`ty()` / `MemoryType`**: TVM regions have a `RegionKind`, not
//!   a wasm `MemoryType`. Surfacing the latter would mislead.
//! - **`Deref<Target = [u8]>`**: tempting but loses error handling
//!   for non-resident regions and writes; bytes are exposed via
//!   `data()` which is fallible.
//!
//! Use the underlying [`TvmHost`] APIs (`create_region`, `pin`,
//! `region_info`, etc.) for the parts that don't translate.

use crate::host::TvmHost;
use tvm_core::{Result, TvmError};

/// A wasm-Memory-shaped view over a single TVM region.
///
/// Lifetime-tied to the host so the directory's bytes can't move
/// underneath us. Cheap to construct; just a `&mut TvmHost` and a
/// `region_id`. Re-create per scope rather than keeping one
/// long-lived.
pub struct RegionView<'a> {
    host: &'a mut TvmHost,
    region_id: u16,
}

impl<'a> RegionView<'a> {
    pub(crate) fn new(host: &'a mut TvmHost, region_id: u16) -> Self {
        Self { host, region_id }
    }

    /// Region id this view targets.
    pub fn region_id(&self) -> u16 {
        self.region_id
    }

    /// Region capacity in bytes — analogous to `wasmtime::Memory::data_size()`
    /// but reflects the TVM region's declared capacity, not a wasm
    /// page count.
    pub fn data_size(&self) -> Result<u32> {
        let info = self.host.directory.region_info(self.region_id)?;
        Ok(info.capacity)
    }

    /// Region capacity in 64 KiB pages — analogous to
    /// `wasmtime::Memory::size()`. Capacity is rounded up to the next
    /// page boundary; non-page-multiple capacities still report a
    /// whole-page count for size compatibility.
    pub fn size_pages(&self) -> Result<u32> {
        let bytes = self.data_size()?;
        Ok(bytes.div_ceil(65536))
    }

    /// Immutable byte view of the entire region. Analogous to
    /// `wasmtime::Memory::data(&store)`. Returns `NotResident` when
    /// the region is currently spilled to a backing store.
    pub fn data(&self) -> Result<&[u8]> {
        self.host.directory.region_full_slice(self.region_id)
    }

    /// Mutable byte view of the entire region. Analogous to
    /// `wasmtime::Memory::data_mut(&mut store)`.
    pub fn data_mut(&mut self) -> Result<&mut [u8]> {
        self.host.directory.region_full_slice_mut(self.region_id)
    }

    /// Raw pointer to the region's bytes. Same lifetime / aliasing
    /// caveats as `wasmtime::Memory::data_ptr`: pointer is valid
    /// until the next mutation of the directory (spill, load, compact,
    /// destroy).
    pub fn data_ptr(&self) -> Result<*const u8> {
        Ok(self.data()?.as_ptr())
    }

    /// Mutable raw pointer; same caveats.
    pub fn data_mut_ptr(&mut self) -> Result<*mut u8> {
        Ok(self.data_mut()?.as_mut_ptr())
    }

    /// Read `dst.len()` bytes starting at `offset` into `dst`.
    /// Analogous to `wasmtime::Memory::read`.
    pub fn read(&self, offset: u32, dst: &mut [u8]) -> Result<()> {
        let bytes = self.data()?;
        let start = offset as usize;
        let end = start
            .checked_add(dst.len())
            .ok_or(TvmError::OutOfBounds)?;
        if end > bytes.len() {
            return Err(TvmError::OutOfBounds);
        }
        dst.copy_from_slice(&bytes[start..end]);
        Ok(())
    }

    /// Write `src.len()` bytes starting at `offset`. Analogous to
    /// `wasmtime::Memory::write`.
    pub fn write(&mut self, offset: u32, src: &[u8]) -> Result<()> {
        let bytes = self.data_mut()?;
        let start = offset as usize;
        let end = start
            .checked_add(src.len())
            .ok_or(TvmError::OutOfBounds)?;
        if end > bytes.len() {
            return Err(TvmError::OutOfBounds);
        }
        bytes[start..end].copy_from_slice(src);
        Ok(())
    }

    /// Copy `len` bytes from `src_off` to `dst_off` within this
    /// region. Source and destination may overlap (slice::copy_within
    /// semantics). Analogous to wasm `memory.copy` with the same
    /// memory as source and destination.
    pub fn copy_within(&mut self, src_off: u32, dst_off: u32, len: u32) -> Result<()> {
        let bytes = self.data_mut()?;
        let src = src_off as usize;
        let dst = dst_off as usize;
        let n = len as usize;
        let src_end = src.checked_add(n).ok_or(TvmError::OutOfBounds)?;
        let dst_end = dst.checked_add(n).ok_or(TvmError::OutOfBounds)?;
        if src_end > bytes.len() || dst_end > bytes.len() {
            return Err(TvmError::OutOfBounds);
        }
        bytes.copy_within(src..src_end, dst);
        Ok(())
    }

    /// Set `len` bytes starting at `offset` to `value`. Analogous to
    /// wasm `memory.fill`.
    pub fn fill(&mut self, offset: u32, len: u32, value: u8) -> Result<()> {
        let bytes = self.data_mut()?;
        let start = offset as usize;
        let end = start
            .checked_add(len as usize)
            .ok_or(TvmError::OutOfBounds)?;
        if end > bytes.len() {
            return Err(TvmError::OutOfBounds);
        }
        bytes[start..end].fill(value);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tvm_core::RegionKind;

    fn host_with_region(cap: u32) -> (TvmHost, u16) {
        let mut host = TvmHost::new();
        let r = host.create_region(RegionKind::HotHeap, cap).unwrap();
        (host, r)
    }

    #[test]
    fn data_size_and_pages() {
        let (mut host, r) = host_with_region(70_000);
        let view = host.region_view(r).unwrap();
        assert_eq!(view.data_size().unwrap(), 70_000);
        // ceil(70_000 / 65_536) = 2
        assert_eq!(view.size_pages().unwrap(), 2);
    }

    #[test]
    fn read_write_round_trip() {
        let (mut host, r) = host_with_region(4096);
        {
            let mut view = host.region_view(r).unwrap();
            view.write(64, b"hello, region").unwrap();
        }
        let view = host.region_view(r).unwrap();
        let mut buf = [0u8; 13];
        view.read(64, &mut buf).unwrap();
        assert_eq!(&buf, b"hello, region");
    }

    #[test]
    fn copy_within_overlapping() {
        let (mut host, r) = host_with_region(256);
        let mut view = host.region_view(r).unwrap();
        view.write(0, &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        view.copy_within(0, 4, 8).unwrap(); // overlapping forward
        let mut buf = [0u8; 16];
        view.read(0, &mut buf).unwrap();
        assert_eq!(&buf[0..4], &[1, 2, 3, 4]);
        assert_eq!(&buf[4..12], &[1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn fill_sets_bytes() {
        let (mut host, r) = host_with_region(256);
        let mut view = host.region_view(r).unwrap();
        view.fill(32, 64, 0xab).unwrap();
        let mut buf = [0u8; 64];
        view.read(32, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0xab));
    }

    #[test]
    fn out_of_bounds_rejected() {
        let (mut host, r) = host_with_region(128);
        let mut view = host.region_view(r).unwrap();
        let mut buf = [0u8; 32];
        assert!(matches!(view.read(120, &mut buf), Err(TvmError::OutOfBounds)));
        assert!(matches!(view.write(120, &[0u8; 32]), Err(TvmError::OutOfBounds)));
    }
}
