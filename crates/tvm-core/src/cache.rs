//! Tiny direct-mapped cache for `(region_id) -> (generation, capacity, residency)`.
//!
//! The cache exists to make hot-path validation a branch-and-compare instead
//! of a Vec lookup + Option deref. Workloads that touch a small number of
//! regions repeatedly will hit on every call after warm-up; cold workloads
//! will miss but still produce correct results because the cache is
//! authoritative-on-write but advisory-on-read (every entry is verified
//! against the directory on hit, so a stale entry just becomes a miss).

use crate::region::Region;
use crate::residency::Residency;

const CACHE_SLOTS: usize = 8;
const CACHE_MASK: u16 = (CACHE_SLOTS - 1) as u16;

#[derive(Clone, Copy, Debug)]
struct CacheSlot {
    valid: bool,
    region_id: u16,
    generation: u16,
    capacity: u32,
    residency_hot: bool,
    /// Raw pointer to the region's bytes, if known. Stable for the
    /// lifetime of the region's underlying allocation; invalidated when
    /// the region's memory is replaced (spill/load/compaction/destroy).
    /// Null if the cache slot was installed without a data pointer.
    data_ptr: usize,
    data_len: u32,
}

impl Default for CacheSlot {
    fn default() -> Self {
        Self {
            valid: false,
            region_id: 0,
            generation: 0,
            capacity: 0,
            residency_hot: false,
            data_ptr: 0,
            data_len: 0,
        }
    }
}

#[derive(Debug, Default)]
pub struct ResolveCache {
    slots: [CacheSlot; CACHE_SLOTS],
    pub hits: u64,
    pub misses: u64,
    pub invalidations: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolveHit {
    pub generation: u16,
    pub capacity: u32,
    pub resident: bool,
}

/// A "fat" cache hit including the raw data pointer for zero-overhead
/// access. The pointer is only valid while the directory isn't mutated;
/// the caller must release any borrow before calling create_region or
/// spill/load/compact.
#[derive(Clone, Copy, Debug)]
pub struct FastHit {
    pub generation: u16,
    pub capacity: u32,
    pub resident: bool,
    pub data_ptr: usize,
    pub data_len: u32,
}

impl ResolveCache {
    pub const fn new() -> Self {
        Self {
            slots: [CacheSlot {
                valid: false,
                region_id: 0,
                generation: 0,
                capacity: 0,
                residency_hot: false,
                data_ptr: 0,
                data_len: 0,
            }; CACHE_SLOTS],
            hits: 0,
            misses: 0,
            invalidations: 0,
        }
    }

    pub fn lookup(&mut self, region_id: u16) -> Option<ResolveHit> {
        let slot = &self.slots[(region_id & CACHE_MASK) as usize];
        if slot.valid && slot.region_id == region_id {
            self.hits += 1;
            Some(ResolveHit {
                generation: slot.generation,
                capacity: slot.capacity,
                resident: slot.residency_hot,
            })
        } else {
            self.misses += 1;
            None
        }
    }

    pub fn install(&mut self, region: &Region) -> ResolveHit {
        let idx = (region.id & CACHE_MASK) as usize;
        let resident = matches!(region.residency, Residency::Hot | Residency::Warm);
        self.slots[idx] = CacheSlot {
            valid: true,
            region_id: region.id,
            generation: region.generation,
            capacity: region.capacity,
            residency_hot: resident,
            data_ptr: 0,
            data_len: 0,
        };
        ResolveHit { generation: region.generation, capacity: region.capacity, resident }
    }

    /// Install with raw data pointer. The caller is responsible for
    /// invalidating the slot before the underlying memory is freed or
    /// replaced — the directory does this on every mutating operation.
    pub fn install_with_data(
        &mut self,
        region: &Region,
        data_ptr: usize,
        data_len: u32,
    ) -> FastHit {
        let idx = (region.id & CACHE_MASK) as usize;
        let resident = matches!(region.residency, Residency::Hot | Residency::Warm);
        self.slots[idx] = CacheSlot {
            valid: true,
            region_id: region.id,
            generation: region.generation,
            capacity: region.capacity,
            residency_hot: resident,
            data_ptr,
            data_len,
        };
        FastHit {
            generation: region.generation,
            capacity: region.capacity,
            resident,
            data_ptr,
            data_len,
        }
    }

    /// Fast lookup that includes the raw data pointer when the slot has
    /// one. Returns None on miss OR on a slot installed without a pointer.
    #[inline]
    pub fn lookup_fast(&mut self, region_id: u16) -> Option<FastHit> {
        let slot = &self.slots[(region_id & CACHE_MASK) as usize];
        if slot.valid && slot.region_id == region_id && slot.data_ptr != 0 {
            self.hits += 1;
            Some(FastHit {
                generation: slot.generation,
                capacity: slot.capacity,
                resident: slot.residency_hot,
                data_ptr: slot.data_ptr,
                data_len: slot.data_len,
            })
        } else {
            self.misses += 1;
            None
        }
    }

    pub fn invalidate(&mut self, region_id: u16) {
        let idx = (region_id & CACHE_MASK) as usize;
        if self.slots[idx].valid && self.slots[idx].region_id == region_id {
            self.slots[idx].valid = false;
            self.invalidations += 1;
        }
    }

    pub fn invalidate_all(&mut self) {
        for slot in &mut self.slots {
            if slot.valid {
                self.invalidations += 1;
                slot.valid = false;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::region::RegionKind;

    fn region(id: u16, gen: u16) -> Region {
        Region {
            id,
            generation: gen,
            kind: RegionKind::HotHeap,
            capacity: 64,
            used: 0,
            residency: Residency::Hot,
            pinned: false,
            pinnable: true,
            spillable: false,
        }
    }

    #[test]
    fn miss_then_install_then_hit() {
        let mut c = ResolveCache::new();
        assert!(c.lookup(3).is_none());
        c.install(&region(3, 1));
        let hit = c.lookup(3).unwrap();
        assert_eq!(hit.generation, 1);
        assert!(hit.resident);
    }

    #[test]
    fn invalidate_forces_miss() {
        let mut c = ResolveCache::new();
        c.install(&region(5, 7));
        c.invalidate(5);
        assert!(c.lookup(5).is_none());
    }

    #[test]
    fn collision_evicts_old_slot() {
        let mut c = ResolveCache::new();
        c.install(&region(0, 1));
        c.install(&region(8, 1)); // same slot index
        assert!(c.lookup(0).is_none());
        assert!(c.lookup(8).is_some());
    }
}
