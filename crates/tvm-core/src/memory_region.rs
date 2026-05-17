//! Pure-data types extracted from `directory.rs` so they're
//! accessible without the `std` feature. The full
//! [`crate::RegionDirectory`] requires libstd (uses `HashMap`,
//! file IO, BackingStore impls); guests that need to mirror its
//! shape (e.g. `tvm-guest-mm::GuestDirectory`) can use the
//! [`MemoryRegion`] trait + [`HandleRemap`] from here without
//! pulling std in.

use alloc::vec::Vec;

use crate::error::Result;

/// An owned, mutable region of bytes backed by some host- or
/// guest-side memory. Implementations include
/// [`crate::VecBackedRegion`] (std-only host impl) and
/// guest-side adapters that wrap a wasm linear memory range.
pub trait MemoryRegion {
    fn len(&self) -> u32;
    fn read(&self, offset: u32, buf: &mut [u8]) -> Result<()>;
    fn write(&mut self, offset: u32, buf: &[u8]) -> Result<()>;
    fn snapshot(&self) -> Vec<u8>;
    fn restore(bytes: Vec<u8>) -> Self
    where
        Self: Sized;
}

/// Mapping from old handles to new handles after compaction.
/// Returned by `compact_region`; callers must use [`migrate`] to
/// rewrite any handles they hold into the region. Old-generation
/// handles fail validation immediately.
#[derive(Debug, Clone)]
pub struct HandleRemap {
    pub region_id: u16,
    pub old_generation: u16,
    pub new_generation: u16,
    /// hashbrown::HashMap so this works in both std and no_std modes.
    pub mapping: hashbrown::HashMap<u32, u32>,
}

impl HandleRemap {
    pub fn migrate(&self, h: crate::handle::Handle) -> Option<crate::handle::Handle> {
        if h.region_id != self.region_id || h.generation != self.old_generation {
            return None;
        }
        let new_offset = self.mapping.get(&h.offset)?;
        Some(crate::handle::Handle {
            region_id: h.region_id,
            generation: self.new_generation,
            offset: *new_offset,
        })
    }
}
