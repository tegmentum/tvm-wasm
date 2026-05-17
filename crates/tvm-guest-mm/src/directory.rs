//! Guest-side region directory. Lives in pool 0 (the workload's default
//! linear memory). Tracks pool assignment + sub-allocator state for each
//! TVM region.
//!
//! Everything here is plain Rust — no special wasm machinery. The
//! multi-memory dispatch is done by separate generated WAT helper
//! functions which the workload calls when reading/writing bytes.

use alloc::vec;
use alloc::vec::Vec;
use tvm_core::{
    AllocatorKind, Handle, HandleRemap, PlacementPolicy, Region, RegionAllocator, RegionKind,
    Residency, Result, TvmError,
};

use crate::facade::Dispatch;

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

/// Guest-side TVM directory.
///
/// **Region lookup is O(1).** Region ids are allocated densely from 1
/// (id 0 is reserved for null), and `regions` is indexed directly by
/// `id` — no linear scan, no hashing. After `dealloc_region` is added,
/// freed slots become `None` and may be re-used by a small free-list
/// (not implemented yet); for now the slot grows monotonically.
///
/// Pool allocation honors `PlacementPolicy` derived from the region's
/// `RegionKind`. Today that means Hot regions land in low-numbered
/// pools (better cache locality for frequently-touched data) and Warm
/// regions land in high-numbered pools (less interference with the
/// hot working set). Same round-robin within each band.
pub struct GuestDirectory {
    pools: Vec<Pool>,
    /// Dense slot table indexed by region_id. `slots[0]` is the null
    /// reservation (always `None`); `slots[k]` for k ≥ 1 is the entry
    /// for region id k.
    slots: Vec<Option<RegionEntry>>,
    next_id: u16,
    /// Round-robin cursor for Hot regions (low band).
    cursor_hot: u32,
    /// Round-robin cursor for Warm regions (high band).
    cursor_warm: u32,
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
            slots: vec![None], // index 0 is reserved
            next_id: 1,
            cursor_hot: 0,
            cursor_warm: 0,
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
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or(TvmError::AllocationFailed)?;
        let policy = PlacementPolicy::for_kind(kind);
        let pool_index = self.choose_pool(capacity, policy.initial_residency)?;
        let pool = &mut self.pools[pool_index as usize];
        let base_offset = pool.used;
        pool.used = pool
            .used
            .checked_add(capacity)
            .ok_or(TvmError::AllocationFailed)?;
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
        // Dense slot insert at id (slots[id] = Some(entry)).
        // next_id was incremented past `id`, and `id == self.slots.len()`
        // here because we always push exactly one slot per allocated id.
        debug_assert_eq!(id as usize, self.slots.len());
        self.slots.push(Some(entry));
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

    /// Compact the region's live allocations toward its base. Walks
    /// the allocator's `allocated_blocks()` in ascending order, slides
    /// each block to its new packed offset via the dispatch's
    /// per-pool `intra_pool_copy` helper, then rebuilds the
    /// allocator's state.
    ///
    /// Returns a `HandleRemap` mapping each block's old offset to its
    /// new offset, plus the region's bumped generation. Old handles
    /// fail validation immediately; the caller migrates them with
    /// `HandleRemap::migrate`.
    ///
    /// Errors:
    /// - `Pinned` if the region is pinned (caller must unpin first).
    /// - `UnsupportedAllocator` if the region uses Bump (no allocated-
    ///   block tracking) or Slab (compaction n/a — uniform slots).
    /// - Anything from `intra_pool_copy` (out-of-bounds in the pool,
    ///   etc.).
    pub fn compact_region(
        &mut self,
        region_id: u16,
        dispatch: &dyn Dispatch,
    ) -> Result<HandleRemap> {
        let entry = self.entry_mut(region_id)?;
        if entry.meta.pinned {
            return Err(TvmError::Pinned);
        }
        let blocks = entry
            .allocator
            .allocated_blocks()
            .ok_or(TvmError::UnsupportedAllocator)?;
        let pool_index = entry.pool_index;
        let base_offset = entry.base_offset;

        // Walk blocks in ascending order; pack each at the next
        // available cursor. Source/destination overlap is fine —
        // wasm memory.copy handles direction.
        // hashbrown::HashMap so this works in both std and no_std modes
        // (matches the field type of HandleRemap after U2).
        let mut mapping: hashbrown::HashMap<u32, u32> =
            hashbrown::HashMap::with_capacity(blocks.len());
        let mut new_blocks: Vec<(u32, u32)> = Vec::with_capacity(blocks.len());
        let mut cursor: u32 = 0;
        for (old_off, size) in &blocks {
            if *old_off != cursor {
                dispatch.intra_pool_copy(
                    pool_index,
                    base_offset + cursor,
                    base_offset + *old_off,
                    *size,
                )?;
            }
            mapping.insert(*old_off, cursor);
            new_blocks.push((cursor, *size));
            cursor = cursor
                .checked_add(*size)
                .ok_or(TvmError::AllocationFailed)?;
        }

        let old_gen = entry.meta.generation;
        let mut next = entry.meta.generation.wrapping_add(1);
        if next == 0 {
            next = 1;
        }
        entry.meta.generation = next;
        entry
            .allocator
            .rebuild_after_compact(&new_blocks, entry.meta.capacity);
        entry.meta.used = entry.allocator.used();

        Ok(HandleRemap {
            region_id,
            old_generation: old_gen,
            new_generation: next,
            mapping,
        })
    }

    /// Pick a pool for a new region.
    ///
    /// Honors the region's initial residency hint:
    /// - `Hot` / `External` → low band `[0, mid)` (ie. lower-numbered
    ///   pools), round-robin within band
    /// - `Warm` / `Cold`    → high band `[mid, n)`, round-robin within
    ///   band
    ///
    /// Where `mid = ceil(n/2)`. With `n=1` everything lands in pool 0.
    /// If the preferred band is full, falls through to the other band
    /// before returning `AllocationFailed`.
    fn choose_pool(&mut self, capacity: u32, residency: Residency) -> Result<u32> {
        let n = self.pools.len() as u32;
        if n == 0 {
            return Err(TvmError::AllocationFailed);
        }
        let mid = n.div_ceil(2);
        let prefers_hot = matches!(residency, Residency::Hot | Residency::External);
        // Try the preferred band first, then the other.
        let bands: [(u32, u32, &mut u32); 2] = if prefers_hot {
            [
                (0, mid, &mut self.cursor_hot),
                (mid, n, &mut self.cursor_warm),
            ]
        } else {
            [
                (mid, n, &mut self.cursor_warm),
                (0, mid, &mut self.cursor_hot),
            ]
        };
        for (lo, hi, cursor) in bands {
            if lo == hi {
                continue;
            } // empty band when n=1
            let span = hi - lo;
            for offset in 0..span {
                let idx = lo + (*cursor + offset) % span;
                let pool = &self.pools[idx as usize];
                if pool.capacity - pool.used >= capacity {
                    *cursor = (*cursor + offset + 1) % span;
                    return Ok(idx);
                }
            }
        }
        Err(TvmError::AllocationFailed)
    }

    fn entry(&self, region_id: u16) -> Result<&RegionEntry> {
        self.slots
            .get(region_id as usize)
            .and_then(|s| s.as_ref())
            .ok_or(TvmError::RegionNotFound(region_id))
    }

    fn entry_mut(&mut self, region_id: u16) -> Result<&mut RegionEntry> {
        self.slots
            .get_mut(region_id as usize)
            .and_then(|s| s.as_mut())
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
    fn hot_regions_land_in_low_band_round_robin() {
        // n=4 → mid=2 → hot band is pools [0, 2). HotHeap is Hot, so the
        // four HotHeap regions cycle through pools 0,1,0,1.
        let mut d = dir(4, 1024);
        let r0 = d
            .create_region(RegionKind::HotHeap, 128, AllocatorKind::Bump)
            .unwrap();
        let r1 = d
            .create_region(RegionKind::HotHeap, 128, AllocatorKind::Bump)
            .unwrap();
        let r2 = d
            .create_region(RegionKind::HotHeap, 128, AllocatorKind::Bump)
            .unwrap();
        let r3 = d
            .create_region(RegionKind::HotHeap, 128, AllocatorKind::Bump)
            .unwrap();
        let pools: [u32; 4] = [
            d.entry(r0).unwrap().pool_index,
            d.entry(r1).unwrap().pool_index,
            d.entry(r2).unwrap().pool_index,
            d.entry(r3).unwrap().pool_index,
        ];
        assert_eq!(pools, [0, 1, 0, 1]);
    }

    #[test]
    fn warm_regions_land_in_high_band() {
        // PageStore is Warm → high band [2, 4).
        let mut d = dir(4, 1024);
        let r0 = d
            .create_region(RegionKind::PageStore, 128, AllocatorKind::Bump)
            .unwrap();
        let r1 = d
            .create_region(RegionKind::PageStore, 128, AllocatorKind::Bump)
            .unwrap();
        let r2 = d
            .create_region(RegionKind::PageStore, 128, AllocatorKind::Bump)
            .unwrap();
        let pools: [u32; 3] = [
            d.entry(r0).unwrap().pool_index,
            d.entry(r1).unwrap().pool_index,
            d.entry(r2).unwrap().pool_index,
        ];
        assert_eq!(pools, [2, 3, 2]);
    }

    #[test]
    fn hot_band_falls_through_to_warm_when_full() {
        // n=2 → mid=1 → hot band is pool 0 only. After exhausting it,
        // a Hot region must fall through to pool 1.
        let mut d = dir(2, 100);
        d.create_region(RegionKind::HotHeap, 100, AllocatorKind::Bump)
            .unwrap();
        let r1 = d
            .create_region(RegionKind::HotHeap, 100, AllocatorKind::Bump)
            .unwrap();
        assert_eq!(d.entry(r1).unwrap().pool_index, 1);
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
        let stale = Handle {
            generation: 99,
            ..h
        };
        assert!(matches!(d.resolve(stale), Err(TvmError::StaleHandle)));
    }

    #[test]
    fn out_of_capacity_falls_through_pools() {
        // n=3 → mid=2 → hot band [0,2), warm band [2,3).
        let mut d = dir(3, 100);
        d.create_region(RegionKind::HotHeap, 100, AllocatorKind::Bump)
            .unwrap();
        d.create_region(RegionKind::HotHeap, 100, AllocatorKind::Bump)
            .unwrap();
        // Third Hot region: hot band is full → falls through to warm.
        let r3 = d
            .create_region(RegionKind::HotHeap, 100, AllocatorKind::Bump)
            .unwrap();
        assert_eq!(d.entry(r3).unwrap().pool_index, 2);
        // Fourth: nowhere to put it.
        assert!(matches!(
            d.create_region(RegionKind::HotHeap, 1, AllocatorKind::Bump),
            Err(TvmError::AllocationFailed)
        ));
    }

    #[test]
    fn slot_lookup_is_o1_dense() {
        // Sanity-check: ids start at 1, slots vec grows by exactly one
        // per create_region. region_info on a non-existent id returns
        // RegionNotFound without iterating.
        let mut d = dir(4, 4096);
        let r1 = d
            .create_region(RegionKind::HotHeap, 64, AllocatorKind::Bump)
            .unwrap();
        let r2 = d
            .create_region(RegionKind::HotHeap, 64, AllocatorKind::Bump)
            .unwrap();
        assert_eq!(r1, 1);
        assert_eq!(r2, 2);
        assert_eq!(d.slots.len(), 3); // null + r1 + r2
        assert!(matches!(
            d.region_info(99),
            Err(TvmError::RegionNotFound(99))
        ));
    }
}
