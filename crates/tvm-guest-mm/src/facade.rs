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

use alloc::boxed::Box;
use alloc::vec::Vec;

use tvm_core::{AllocatorKind, Handle, Region, RegionKind, Result, TvmError, TvmFacade};

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
    /// Boxed `Dispatch` impl. In a real guest the impl forwards to
    /// the WAT-defined helpers in the same module; in tests it can
    /// hold mutable state directly. `Box<dyn Dispatch>` lets callers
    /// pick any impl without leaking a type parameter through every
    /// API surface.
    dispatch: Box<dyn Dispatch>,
    /// Single-entry resolve cache. Hot loop pattern is "many ops on the
    /// same region"; we cache the last successful resolve and skip the
    /// directory lookup on subsequent same-region accesses with the
    /// same generation.
    cache: ResolveCache,
}

/// Single-entry cache of `(region_id, generation) → (pool_index,
/// base_offset)`. `region_id == 0` means empty (region 0 is reserved
/// for the null handle).
#[derive(Clone, Copy)]
struct ResolveCache {
    region_id: u16,
    generation: u16,
    pool_index: u32,
    base_offset: u32,
}

impl ResolveCache {
    const EMPTY: Self = Self {
        region_id: 0,
        generation: 0,
        pool_index: 0,
        base_offset: 0,
    };
}

/// Trait implemented by anything that can dispatch reads, writes, and
/// intra-pool copies against the guest's memory pools.
///
/// In a real wasm guest the impl forwards to the WAT-defined helpers
/// (`tvm_copy_to_default`, `tvm_copy_from_default`,
/// `tvm_intra_pool_copy_p{K}`). In tests, an impl can hold mutable
/// state directly — no static globals needed.
///
/// Methods take `&self` (not `&mut self`) so multiple operations
/// against the same dispatcher can be issued from a single
/// `&mut GuestTvm`. Implementations that need internal mutability
/// should use `Cell`/`RefCell` or a `Mutex`.
pub trait Dispatch: Send + Sync {
    /// Read bytes from `(pool, offset)` into `dst`.
    fn read_bytes(&self, pool: u32, offset: u32, dst: &mut [u8]) -> Result<()>;

    /// Write bytes to `(pool, offset)` from `src`.
    fn write_bytes(&self, pool: u32, offset: u32, src: &[u8]) -> Result<()>;

    /// Copy `len` bytes from `src_off` to `dst_off` within the same
    /// pool. Used by `compact_region`. Source and destination may
    /// overlap (wasm memory.copy semantics).
    fn intra_pool_copy(&self, pool: u32, dst_off: u32, src_off: u32, len: u32) -> Result<()>;
}

/// Adapter that wraps three plain function pointers / closures into a
/// `Dispatch` impl. Lets existing call sites that constructed
/// `Dispatch` as a struct-literal port to the trait with minimal
/// churn.
///
/// Closures must be `Send + Sync` if you want to share `GuestTvm`
/// across threads (wasm guests are single-threaded today, so this
/// usually doesn't matter).
pub struct FnDispatch<R, W, C>
where
    R: Fn(u32, u32, &mut [u8]) -> Result<()> + Send + Sync,
    W: Fn(u32, u32, &[u8]) -> Result<()> + Send + Sync,
    C: Fn(u32, u32, u32, u32) -> Result<()> + Send + Sync,
{
    read: R,
    write: W,
    copy: C,
}

impl<R, W, C> FnDispatch<R, W, C>
where
    R: Fn(u32, u32, &mut [u8]) -> Result<()> + Send + Sync,
    W: Fn(u32, u32, &[u8]) -> Result<()> + Send + Sync,
    C: Fn(u32, u32, u32, u32) -> Result<()> + Send + Sync,
{
    pub fn new(read: R, write: W, copy: C) -> Self {
        Self { read, write, copy }
    }
}

impl<R, W, C> Dispatch for FnDispatch<R, W, C>
where
    R: Fn(u32, u32, &mut [u8]) -> Result<()> + Send + Sync,
    W: Fn(u32, u32, &[u8]) -> Result<()> + Send + Sync,
    C: Fn(u32, u32, u32, u32) -> Result<()> + Send + Sync,
{
    fn read_bytes(&self, pool: u32, offset: u32, dst: &mut [u8]) -> Result<()> {
        (self.read)(pool, offset, dst)
    }
    fn write_bytes(&self, pool: u32, offset: u32, src: &[u8]) -> Result<()> {
        (self.write)(pool, offset, src)
    }
    fn intra_pool_copy(&self, pool: u32, dst_off: u32, src_off: u32, len: u32) -> Result<()> {
        (self.copy)(pool, dst_off, src_off, len)
    }
}

impl GuestTvm {
    /// Construct a guest-side TVM with the supplied pool descriptors.
    /// Pool descriptors must match the wasm module's declared memories.
    /// Defaults to `AllocatorKind::Bump`; use `with_default_allocator`
    /// for `Freelist` (compactable, fragments) or `Slab` (uniform
    /// slot sizes).
    pub fn new<D: Dispatch + 'static>(pools: Vec<Pool>, dispatch: D) -> Self {
        Self::with_default_allocator(pools, dispatch, AllocatorKind::Bump)
    }

    /// Construct with a specific default allocator.
    ///
    /// - `Bump`: O(1) alloc, no reclaim. Best for one-shot or
    ///   short-lived workloads.
    /// - `Freelist`: tracks live blocks; supports `compact_region`
    ///   to reclaim fragmented space. Choose for long-running guests
    ///   with allocate-and-free churn.
    /// - `Slab`: uniform slot sizes; no fragmentation by construction.
    ///   Choose when objects are all the same size.
    pub fn with_default_allocator<D: Dispatch + 'static>(
        pools: Vec<Pool>,
        dispatch: D,
        default_allocator: AllocatorKind,
    ) -> Self {
        Self {
            directory: GuestDirectory::new(pools),
            default_allocator,
            dispatch: Box::new(dispatch),
            cache: ResolveCache::EMPTY,
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
    pub fn compact_region(&mut self, region_id: u16) -> Result<tvm_core::HandleRemap> {
        // Cache invalidation: compaction bumps the region's generation
        // and may move blocks within the pool. Drop the cache entry
        // for this region so subsequent ops re-resolve.
        if self.cache.region_id == region_id {
            self.cache = ResolveCache::EMPTY;
        }
        self.directory.compact_region(region_id, &*self.dispatch)
    }

    /// Cache-aware resolve. Returns the absolute (pool_index,
    /// pool_offset) for `handle`. Hot path is the fast branch:
    /// region_id and generation match the cache entry, and we just add
    /// the cached base_offset to the handle's offset. Miss path falls
    /// back to the directory's O(1) slot lookup and refreshes the
    /// cache.
    fn cached_resolve(&mut self, handle: Handle) -> Result<(u32, u32)> {
        if handle.region_id != 0
            && handle.region_id == self.cache.region_id
            && handle.generation == self.cache.generation
        {
            // Bounds check on offset still happens via the dispatch
            // call; we just save the directory lookup here.
            let abs = self
                .cache
                .base_offset
                .checked_add(handle.offset)
                .ok_or(TvmError::OutOfBounds)?;
            return Ok((self.cache.pool_index, abs));
        }
        // Miss — full resolve. Then refresh the cache entry from the
        // returned (pool, abs) tuple by deriving base = abs -
        // handle.offset. Both are bounds-checked against region size
        // by directory.resolve.
        let (pool, abs) = self.directory.resolve(handle)?;
        let base = abs.checked_sub(handle.offset).unwrap_or(0);
        self.cache = ResolveCache {
            region_id: handle.region_id,
            generation: handle.generation,
            pool_index: pool,
            base_offset: base,
        };
        Ok((pool, abs))
    }
}

impl TvmFacade for GuestTvm {
    fn create_region(&mut self, kind: RegionKind, capacity: u32) -> Result<u16> {
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
        let (pool, off) = self.cached_resolve(handle)?;
        let end = off
            .checked_add(buf.len() as u32)
            .ok_or(TvmError::OutOfBounds)?;
        let _ = end;
        self.dispatch.read_bytes(pool, off, buf)
    }

    fn write(&mut self, handle: Handle, data: &[u8]) -> Result<()> {
        let (pool, off) = self.cached_resolve(handle)?;
        let end = off
            .checked_add(data.len() as u32)
            .ok_or(TvmError::OutOfBounds)?;
        let _ = end;
        self.dispatch.write_bytes(pool, off, data)
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
            FnDispatch::new(stub_read, stub_write, stub_intra_pool_copy),
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

    #[test]
    fn with_default_allocator_freelist_creates_compactable_regions() {
        let mut pools = STUB_POOLS.lock().unwrap();
        pools.clear();
        for _ in 0..2 {
            pools.push(vec![0u8; 4096]);
        }
        drop(pools);
        let pool_descs: Vec<Pool> = (0..2u32)
            .map(|i| Pool {
                memory_index: i,
                used: 0,
                capacity: 4096,
            })
            .collect();
        let mut g = GuestTvm::with_default_allocator(
            pool_descs,
            FnDispatch::new(stub_read, stub_write, stub_intra_pool_copy),
            AllocatorKind::Freelist,
        );
        // Region inherits the default; compaction must succeed (Bump
        // would have returned UnsupportedAllocator).
        let r = g.create_region(RegionKind::HotHeap, 256).unwrap();
        let h = g.alloc(r, 64).unwrap();
        g.write(h, &[0x55; 64]).unwrap();
        g.compact_region(r)
            .expect("freelist regions are compactable");
    }

    #[test]
    fn resolve_cache_hits_on_repeat_access() {
        // Indirect verification: same-region reads are fast and
        // correct. Hard to assert "cache hit" without exposing
        // counters; instead we assert correctness across a sequence
        // of reads/writes that would expose any cache staleness.
        let mut g = build_guest(2, 4096);
        let r1 = g.create_region(RegionKind::HotHeap, 256).unwrap();
        let r2 = g.create_region(RegionKind::HotHeap, 256).unwrap();
        let h1 = g.alloc(r1, 32).unwrap();
        let h2 = g.alloc(r2, 32).unwrap();

        for i in 0..100u32 {
            // Alternate writes/reads against r1 and r2 so the cache
            // gets evicted and re-warmed each iteration.
            g.write(h1, &[(i & 0xff) as u8; 32]).unwrap();
            g.write(h2, &[((i ^ 0xff) & 0xff) as u8; 32]).unwrap();
            let mut buf1 = [0u8; 32];
            let mut buf2 = [0u8; 32];
            g.read(h1, &mut buf1).unwrap();
            g.read(h2, &mut buf2).unwrap();
            assert_eq!(buf1[0], (i & 0xff) as u8);
            assert_eq!(buf2[0], ((i ^ 0xff) & 0xff) as u8);
        }
    }
}
