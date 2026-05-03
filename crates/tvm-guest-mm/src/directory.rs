//! Guest-side region directory. Lives in pool 0 (the workload's default
//! linear memory). Tracks pool assignment + sub-allocator state for each
//! TVM region.
//!
//! Everything here is plain Rust — no special wasm machinery. The
//! multi-memory dispatch is done by separate generated WAT helper
//! functions which the workload calls when reading/writing bytes.

use tvm_core::{
    AllocatorKind, Handle, PlacementPolicy, Region, RegionAllocator, RegionKind,
    Residency, Result, TvmError,
};

/// One memory pool in the multi-memory module. Each pool is a separate
/// wasm memory (index 0..N) that the module declares internally.
pub struct Pool {
    /// Wasm memory index this pool corresponds to (0..N).
    pub memory_index: u32,
    /// Capacity used so far (regions + their internal allocations). Bumped
    /// by `place_region`.
    pub used: u32,
    /// Total declared capacity (max-bytes for this pool).
    pub capacity: u32,
}

/// Guest-side TVM directory. Owns a fixed-size table of regions plus a
/// fixed-size table of pools. Both are `Vec`s allocated at startup
/// inside pool 0; they don't grow.
pub struct GuestDirectory {
    pools: Vec<Pool>,
    regions: Vec<Option<RegionEntry>>,
    next_id: u16,
    placement_cursor: u32, // round-robin pool selector
}

struct RegionEntry {
    meta: Region,
    pool_index: u32,
    /// Offset within the pool where this region's bytes start.
    base_offset: u32,
    allocator: RegionAllocator,
}

impl GuestDirectory {
    /// Create a directory backed by the supplied pool descriptors. The
    /// caller (typically the workload's startup code) must have declared
    /// these memories in the wasm module and must pass the matching
    /// `Pool` records in.
    pub fn new(pools: Vec<Pool>) -> Self {
        Self {
            pools,
            regions: Vec::new(),
            next_id: 1, // 0 reserved for "null"
            placement_cursor: 0,
        }
    }

    pub fn pools(&self) -> &[Pool] {
        &self.pools
    }

    /// Create a region of `capacity` bytes. Returns the region_id. The
    /// directory picks a pool with sufficient remaining capacity using
    /// round-robin; a future revision could honor PlacementPolicy hints.
    pub fn create_region(
        &mut self,
        kind: RegionKind,
        capacity: u32,
        allocator: AllocatorKind,
    ) -> Result<u16> {
        let id = self.next_id;
        self.next_id = self.next_id.checked_add(1).ok_or(TvmError::AllocationFailed)?;
        let pool_index = self.choose_pool(capacity)?;
        let pool = &mut self.pools[pool_index as usize];
        let base_offset = pool.used;
        pool.used = pool
            .used
            .checked_add(capacity)
            .ok_or(TvmError::AllocationFailed)?;
        let policy = PlacementPolicy::for_kind(kind);
        let entry = RegionEntry {
            meta: Region {
                id,
                generation: 1,
                kind,
                capacity,
                used: 0,
                residency: policy.initial_residency,
                pinned: false,
                pinnable: policy.pinnable,
                spillable: false, // guest-mm has no spill
            },
            pool_index,
            base_offset,
            allocator: RegionAllocator::new(allocator, capacity),
        };
        self.regions.push(Some(entry));
        Ok(id)
    }

    pub fn alloc(&mut self, region_id: u16, size: u32) -> Result<Handle> {
        let entry = self.entry_mut(region_id)?;
        let offset = entry.allocator.alloc(size, 1)?;
        entry.meta.used = entry.allocator.used();
        Ok(Handle {
            region_id,
            generation: entry.meta.generation,
            offset,
        })
    }

    pub fn dealloc(&mut self, handle: Handle) -> Result<()> {
        let entry = self.entry_mut(handle.region_id)?;
        if entry.meta.generation != handle.generation {
            return Err(TvmError::StaleHandle);
        }
        entry.allocator.dealloc(handle.offset)?;
        entry.meta.used = entry.allocator.used();
        Ok(())
    }

    /// Resolve a handle to its concrete (pool_index, absolute offset
    /// within pool). The workload calls this and then passes the result
    /// to the WAT-generated dispatch functions.
    pub fn resolve(&self, handle: Handle) -> Result<(u32, u32)> {
        let entry = self.entry(handle.region_id)?;
        if entry.meta.generation != handle.generation {
            return Err(TvmError::StaleHandle);
        }
        let absolute = entry
            .base_offset
            .checked_add(handle.offset)
            .ok_or(TvmError::OutOfBounds)?;
        Ok((entry.pool_index, absolute))
    }

    pub fn region_info(&self, region_id: u16) -> Result<&Region> {
        Ok(&self.entry(region_id)?.meta)
    }

    pub fn pin(&mut self, region_id: u16) -> Result<()> {
        let entry = self.entry_mut(region_id)?;
        if !entry.meta.pinnable {
            return Err(TvmError::PolicyViolation);
        }
        entry.meta.pinned = true;
        Ok(())
    }

    fn choose_pool(&mut self, capacity: u32) -> Result<u32> {
        // Round-robin starting at placement_cursor; pick first pool with
        // enough remaining capacity. Wraps around once.
        let n = self.pools.len() as u32;
        if n == 0 {
            return Err(TvmError::AllocationFailed);
        }
        for offset in 0..n {
            let idx = (self.placement_cursor + offset) % n;
            let pool = &self.pools[idx as usize];
            if pool.capacity - pool.used >= capacity {
                self.placement_cursor = (idx + 1) % n;
                return Ok(idx);
            }
        }
        Err(TvmError::AllocationFailed)
    }

    fn entry(&self, region_id: u16) -> Result<&RegionEntry> {
        self.regions
            .iter()
            .find_map(|e| e.as_ref().filter(|r| r.meta.id == region_id))
            .ok_or(TvmError::RegionNotFound(region_id))
    }

    fn entry_mut(&mut self, region_id: u16) -> Result<&mut RegionEntry> {
        self.regions
            .iter_mut()
            .find_map(|e| e.as_mut().filter(|r| r.meta.id == region_id))
            .ok_or(TvmError::RegionNotFound(region_id))
    }

    /// Drop residency-warning since it's pure-guest: regions never leave
    /// memory. Reading this avoids the unused-field warning in tvm-core's
    /// Residency import for now.
    #[doc(hidden)]
    pub fn _residency_marker(&self) -> Residency {
        Residency::Hot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(pool_count: u32, pool_capacity: u32) -> GuestDirectory {
        GuestDirectory::new(
            (0..pool_count)
                .map(|i| Pool {
                    memory_index: i,
                    used: 0,
                    capacity: pool_capacity,
                })
                .collect(),
        )
    }

    #[test]
    fn create_region_round_robin_across_pools() {
        let mut d = dir(4, 1024);
        let r0 = d.create_region(RegionKind::HotHeap, 256, AllocatorKind::Bump).unwrap();
        let r1 = d.create_region(RegionKind::HotHeap, 256, AllocatorKind::Bump).unwrap();
        let r2 = d.create_region(RegionKind::HotHeap, 256, AllocatorKind::Bump).unwrap();
        let r3 = d.create_region(RegionKind::HotHeap, 256, AllocatorKind::Bump).unwrap();

        let info0 = d.region_info(r0).unwrap();
        let info1 = d.region_info(r1).unwrap();
        let info2 = d.region_info(r2).unwrap();
        let info3 = d.region_info(r3).unwrap();
        let _ = (info0, info1, info2, info3);

        // First four regions land in different pools.
        let p0 = d.entry(r0).unwrap().pool_index;
        let p1 = d.entry(r1).unwrap().pool_index;
        let p2 = d.entry(r2).unwrap().pool_index;
        let p3 = d.entry(r3).unwrap().pool_index;
        assert_eq!([p0, p1, p2, p3], [0, 1, 2, 3]);
    }

    #[test]
    fn alloc_and_resolve_yields_correct_pool_offset() {
        let mut d = dir(2, 4096);
        let r = d
            .create_region(RegionKind::HotHeap, 1024, AllocatorKind::Bump)
            .unwrap();
        let h = d.alloc(r, 64).unwrap();
        let (pool, abs) = d.resolve(h).unwrap();
        assert_eq!(pool, 0);
        // Absolute offset = base + within-region offset. Region was first
        // allocation in pool 0 → base=0; first alloc inside region → off=0.
        assert_eq!(abs, 0);

        let h2 = d.alloc(r, 32).unwrap();
        let (_, abs2) = d.resolve(h2).unwrap();
        assert_eq!(abs2, 64);
    }

    #[test]
    fn stale_handle_rejected_on_resolve() {
        let mut d = dir(1, 4096);
        let r = d
            .create_region(RegionKind::HotHeap, 1024, AllocatorKind::Bump)
            .unwrap();
        let h = d.alloc(r, 32).unwrap();
        // Hand-craft a stale handle (different generation).
        let stale = Handle { generation: 99, ..h };
        assert!(matches!(d.resolve(stale), Err(TvmError::StaleHandle)));
    }

    #[test]
    fn out_of_capacity_falls_through_pools() {
        let mut d = dir(3, 100);
        // First two consume pools 0 and 1.
        d.create_region(RegionKind::HotHeap, 100, AllocatorKind::Bump).unwrap();
        d.create_region(RegionKind::HotHeap, 100, AllocatorKind::Bump).unwrap();
        // Third should land in pool 2.
        let r3 = d.create_region(RegionKind::HotHeap, 100, AllocatorKind::Bump).unwrap();
        assert_eq!(d.entry(r3).unwrap().pool_index, 2);
        // Fourth: nowhere to put it.
        assert!(matches!(
            d.create_region(RegionKind::HotHeap, 1, AllocatorKind::Bump),
            Err(TvmError::AllocationFailed)
        ));
    }
}
