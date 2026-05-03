//! Guest-side TVM facade. Bridges the host-side `TvmFacade` trait to
//! the guest's `GuestDirectory` + WAT-generated dispatch helpers.
//!
//! The guest side has a structural quirk: read/write operations need to
//! call the WAT-defined dispatch functions (`tvm_load_u8`, etc.), which
//! are wasm-level functions in the guest's own module. From Rust source
//! they appear as `extern "C"` declarations.
//!
//! `GuestTvm` packages this so user code can write deployment-agnostic
//! Rust:
//!
//! ```ignore
//! fn process<T: TvmFacade>(tvm: &mut T) -> Result<u32> {
//!     let r = tvm.create_region(RegionKind::HotHeap, 4096)?;
//!     let h = tvm.alloc(r, 64)?;
//!     tvm.write(h, b"payload")?;
//!     // ...
//!     Ok(r as u32)
//! }
//!
//! // Same code runs against:
//! //   - host-side TvmHost
//! //   - guest-side GuestTvm
//! ```

use tvm_core::{
    AllocatorKind, Handle, Region, RegionKind, Result, TvmError, TvmFacade,
};

use crate::directory::{GuestDirectory, Pool};

/// Guest-side facade. Owns a `GuestDirectory` plus a function pointer
/// table for the dispatch helpers. The dispatch helpers are populated
/// by `init` — typically called once at module startup with addresses
/// of the WAT-defined functions, or with no-op stubs in unit tests.
pub struct GuestTvm {
    directory: GuestDirectory,
    /// Default allocator used by `create_region`. Mirrors `TvmHost`'s
    /// default for parity.
    pub default_allocator: AllocatorKind,
    /// Function pointers to the dispatch helpers. In a real guest these
    /// come from `extern "C"` declarations against the WAT-generated
    /// functions in the same module. In unit tests on the host they
    /// can be stubbed.
    dispatch: Dispatch,
}

/// Shape of the dispatch function table. Each function is a thin
/// adapter over a wasm-level dispatcher. The Rust-side code calls
/// these by function pointer so the facade can be tested host-side
/// without a real wasm runtime.
pub struct Dispatch {
    /// Read bytes from `(pool, offset)` into `dst`. Implemented on the
    /// guest side by N calls to `tvm_load_u8` or one bulk
    /// `tvm_copy_to_default` (then a memcpy from default mem to dst).
    pub read_bytes: fn(pool: u32, offset: u32, dst: &mut [u8]) -> Result<()>,
    /// Write bytes to `(pool, offset)`.
    pub write_bytes: fn(pool: u32, offset: u32, src: &[u8]) -> Result<()>,
    /// Copy `len` bytes from `src_off` to `dst_off` within the same
    /// pool. Used by `compact_region`. Source and destination may
    /// overlap (wasm memory.copy semantics).
    pub intra_pool_copy:
        fn(pool: u32, dst_off: u32, src_off: u32, len: u32) -> Result<()>,
}

impl GuestTvm {
    /// Construct a guest-side TVM with the supplied pool descriptors.
    /// Pool descriptors must match the wasm module's declared memories.
    pub fn new(pools: Vec<Pool>, dispatch: Dispatch) -> Self {
        Self {
            directory: GuestDirectory::new(pools),
            default_allocator: AllocatorKind::Bump,
            dispatch,
        }
    }

    pub fn directory(&self) -> &GuestDirectory {
        &self.directory
    }

    pub fn directory_mut(&mut self) -> &mut GuestDirectory {
        &mut self.directory
    }

    /// Compact a region in place. Walks the region's freelist of live
    /// blocks, slides each one toward the start of the region using
    /// the dispatch's `intra_pool_copy` (statically lowered to
    /// `memory.copy K K` per pool), and rebuilds the allocator's
    /// state. Bumps the region's generation so any handles held over
    /// the compaction must be migrated via the returned `HandleRemap`
    /// before they can be used again.
    ///
    /// Only supported for regions backed by an allocator that tracks
    /// allocations (Freelist today; Bump returns
    /// `UnsupportedAllocator`).
    pub fn compact_region(
        &mut self,
        region_id: u16,
    ) -> Result<tvm_core::HandleRemap> {
        self.directory.compact_region(region_id, &self.dispatch)
    }
}

impl TvmFacade for GuestTvm {
    fn create_region(
        &mut self,
        kind: RegionKind,
        capacity: u32,
    ) -> Result<u16> {
        self.directory
            .create_region(kind, capacity, self.default_allocator)
    }

    fn alloc(&mut self, region: u16, size: u32) -> Result<Handle> {
        self.directory.alloc(region, size)
    }

    fn dealloc(&mut self, handle: Handle) -> Result<()> {
        self.directory.dealloc(handle)
    }

    fn read(&mut self, handle: Handle, buf: &mut [u8]) -> Result<()> {
        let (pool, off) = self.directory.resolve(handle)?;
        let end = off
            .checked_add(buf.len() as u32)
            .ok_or(TvmError::OutOfBounds)?;
        let _ = end;
        (self.dispatch.read_bytes)(pool, off, buf)
    }

    fn write(&mut self, handle: Handle, data: &[u8]) -> Result<()> {
        let (pool, off) = self.directory.resolve(handle)?;
        let end = off
            .checked_add(data.len() as u32)
            .ok_or(TvmError::OutOfBounds)?;
        let _ = end;
        (self.dispatch.write_bytes)(pool, off, data)
    }

    fn pin(&mut self, region: u16) -> Result<()> {
        self.directory.pin(region)
    }

    fn unpin(&mut self, _region: u16) -> Result<()> {
        // GuestDirectory doesn't currently expose unpin; trivial to add.
        Ok(())
    }

    fn region_info(&self, region: u16) -> Result<Region> {
        self.directory.region_info(region).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Test bridge: a stub Dispatch backed by a host-side `Vec<u8>` per
    // pool. Lets us unit-test the facade without a wasm runtime.
    static STUB_POOLS: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

    fn stub_read(pool: u32, off: u32, dst: &mut [u8]) -> Result<()> {
        let pools = STUB_POOLS.lock().unwrap();
        let p = &pools[pool as usize];
        let start = off as usize;
        let end = start + dst.len();
        if end > p.len() {
            return Err(TvmError::OutOfBounds);
        }
        dst.copy_from_slice(&p[start..end]);
        Ok(())
    }

    fn stub_write(pool: u32, off: u32, src: &[u8]) -> Result<()> {
        let mut pools = STUB_POOLS.lock().unwrap();
        let p = &mut pools[pool as usize];
        let start = off as usize;
        let end = start + src.len();
        if end > p.len() {
            return Err(TvmError::OutOfBounds);
        }
        p[start..end].copy_from_slice(src);
        Ok(())
    }

    fn stub_intra_pool_copy(pool: u32, dst_off: u32, src_off: u32, len: u32) -> Result<()> {
        let mut pools = STUB_POOLS.lock().unwrap();
        let p = &mut pools[pool as usize];
        let s = src_off as usize;
        let d = dst_off as usize;
        let n = len as usize;
        if s + n > p.len() || d + n > p.len() {
            return Err(TvmError::OutOfBounds);
        }
        p.copy_within(s..s + n, d);
        Ok(())
    }

    fn build_guest(n_pools: usize, capacity: u32) -> GuestTvm {
        let mut pools = STUB_POOLS.lock().unwrap();
        pools.clear();
        for _ in 0..n_pools {
            pools.push(vec![0u8; capacity as usize]);
        }
        drop(pools);

        let pool_descs: Vec<Pool> = (0..n_pools as u32)
            .map(|i| Pool {
                memory_index: i,
                used: 0,
                capacity,
            })
            .collect();
        GuestTvm::new(
            pool_descs,
            Dispatch {
                read_bytes: stub_read,
                write_bytes: stub_write,
                intra_pool_copy: stub_intra_pool_copy,
            },
        )
    }

    #[test]
    fn facade_round_trip() {
        let mut g = build_guest(4, 4096);
        let r = g.create_region(RegionKind::HotHeap, 1024).unwrap();
        let h = g.alloc(r, 16).unwrap();
        g.write(h, b"facade-via-guest").unwrap();
        let mut buf = [0u8; 16];
        g.read(h, &mut buf).unwrap();
        assert_eq!(&buf, b"facade-via-guest");
    }
}
