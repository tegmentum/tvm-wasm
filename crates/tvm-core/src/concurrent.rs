//! Concurrent region directory with per-region locking.
//!
//! Unlike [`crate::RegionDirectory`], this directory's methods take `&self`
//! and use interior mutability so multiple threads can operate on **different
//! regions in parallel** without blocking each other.
//!
//! ## Why this isn't deduplicated with `RegionDirectory`
//!
//! The two directories have similar shapes but different lock disciplines:
//! `RegionDirectory` is `&mut self`-driven (caller's responsibility to
//! serialize); `ConcurrentDirectory` is `&self`-driven with internal
//! `RwLock` + per-entry `Mutex`. The locking is interleaved with the
//! logic, so a "pure logic" extraction would only save ~30 lines while
//! adding indirection.
//!
//! `ConcurrentDirectory` deliberately implements a **subset** of
//! `RegionDirectory`'s features — no slice access (would expose a
//! pointer that races with other threads), no snapshot/restore, no
//! debug helpers, no external tier, no async ops. The features that
//! make sense under per-region locking are here; the rest stay
//! single-threaded-only.
//!
//! ## Lock layout
//!
//! - `regions: RwLock<Vec<Option<Arc<Mutex<RegionEntry<M>>>>>>` — outer
//!   read-lock for membership, per-entry mutex for the contents. Membership
//!   changes (`create_region`, `destroy_region`) take the write-lock; all
//!   other ops only take the read-lock.
//! - `warm_lru: Mutex<VecDeque<u16>>` — separate LRU lock; never held while
//!   waiting on a per-region lock.
//!
//! ## Lock-ordering rules
//!
//! Multi-region operations (cross-region copy) acquire per-region locks in
//! ascending `region_id` order to avoid deadlock. The LRU lock is always
//! taken last (or alone). The outer regions vector lock is always taken
//! first.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, RwLock};

use crate::allocator::{AllocatorKind, RegionAllocator};
use crate::backing::BackingStore;
use crate::directory::{HandleRemap, MemoryRegion, RegionEntry};
use crate::error::{Result, TvmError};
use crate::eviction::{counts_toward_resident, within_tier_cmp, EvictionPolicy, EvictionReport};
use crate::handle::Handle;
use crate::metrics::{MetricsSnapshot, RegionMetrics};
use crate::policy::PlacementPolicy;
use crate::region::{Region, RegionKind};
use crate::residency::Residency;

type SharedEntry<M> = Arc<Mutex<RegionEntry<M>>>;

pub struct ConcurrentDirectory<M: MemoryRegion + Send> {
    regions: RwLock<Vec<Option<SharedEntry<M>>>>,
    warm_lru: Mutex<VecDeque<u16>>,
}

impl<M: MemoryRegion + Send> Default for ConcurrentDirectory<M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M: MemoryRegion + Send> ConcurrentDirectory<M> {
    pub fn new() -> Self {
        Self {
            regions: RwLock::new(Vec::new()),
            warm_lru: Mutex::new(VecDeque::new()),
        }
    }

    pub fn create_region(&self, kind: RegionKind, capacity: u32, memory: M) -> Result<u16> {
        self.create_region_with(kind, capacity, AllocatorKind::Bump, memory)
    }

    pub fn create_region_with(
        &self,
        kind: RegionKind,
        capacity: u32,
        allocator: AllocatorKind,
        memory: M,
    ) -> Result<u16> {
        self.create_region_with_policy(
            kind,
            capacity,
            allocator,
            PlacementPolicy::for_kind(kind),
            memory,
        )
    }

    pub fn create_region_with_policy(
        &self,
        kind: RegionKind,
        capacity: u32,
        allocator: AllocatorKind,
        policy: PlacementPolicy,
        memory: M,
    ) -> Result<u16> {
        let mut regions = self.regions.write().map_err(|_| poisoned())?;
        let id = u16::try_from(regions.len()).map_err(|_| TvmError::AllocationFailed)?;
        let initial_residency = policy.initial_residency;
        let entry = RegionEntry {
            meta: Region {
                id,
                generation: 1,
                kind,
                capacity,
                used: 0,
                residency: initial_residency,
                pinned: false,
                pinnable: policy.pinnable,
                spillable: policy.spillable,
            },
            memory: Some(memory),
            metrics: RegionMetrics::default(),
            allocator: RegionAllocator::new(allocator, capacity),
        };
        regions.push(Some(Arc::new(Mutex::new(entry))));
        drop(regions);
        if initial_residency == Residency::Warm {
            self.lru_push_front(id);
        }
        Ok(id)
    }

    pub fn destroy_region(&self, region_id: u16) -> Result<()> {
        let mut regions = self.regions.write().map_err(|_| poisoned())?;
        let slot = regions
            .get_mut(region_id as usize)
            .ok_or(TvmError::RegionNotFound(region_id))?;
        if slot.is_none() {
            return Err(TvmError::RegionNotFound(region_id));
        }
        *slot = None;
        drop(regions);
        self.lru_remove(region_id);
        Ok(())
    }

    pub fn alloc(&self, region_id: u16, size: u32) -> Result<Handle> {
        let entry_arc = self.entry_arc(region_id)?;
        let mut entry = entry_arc.lock().map_err(|_| poisoned())?;
        let offset = entry.allocator.alloc(size, 1)?;
        entry.meta.used = entry.allocator.used();
        entry.metrics.record_alloc(size as u64);
        Ok(Handle {
            region_id,
            generation: entry.meta.generation,
            offset,
        })
    }

    pub fn dealloc(&self, handle: Handle) -> Result<()> {
        let entry_arc = self.entry_arc(handle.region_id)?;
        let mut entry = entry_arc.lock().map_err(|_| poisoned())?;
        if entry.meta.generation != handle.generation {
            return Err(TvmError::StaleHandle);
        }
        entry.allocator.dealloc(handle.offset)?;
        entry.meta.used = entry.allocator.used();
        Ok(())
    }

    pub fn read(&self, handle: Handle, buf: &mut [u8]) -> Result<()> {
        let entry_arc = self.entry_arc(handle.region_id)?;
        let entry = entry_arc.lock().map_err(|_| poisoned())?;
        if entry.meta.generation != handle.generation {
            return Err(TvmError::StaleHandle);
        }
        let memory = entry.memory.as_ref().ok_or(TvmError::NotResident)?;
        let end = handle
            .offset
            .checked_add(buf.len() as u32)
            .ok_or(TvmError::OutOfBounds)?;
        if end > entry.meta.capacity {
            return Err(TvmError::OutOfBounds);
        }
        memory.read(handle.offset, buf)?;
        entry.metrics.record_read(buf.len() as u64);
        Ok(())
    }

    pub fn write(&self, handle: Handle, data: &[u8]) -> Result<()> {
        let entry_arc = self.entry_arc(handle.region_id)?;
        let mut entry = entry_arc.lock().map_err(|_| poisoned())?;
        if entry.meta.generation != handle.generation {
            return Err(TvmError::StaleHandle);
        }
        let end = handle
            .offset
            .checked_add(data.len() as u32)
            .ok_or(TvmError::OutOfBounds)?;
        if end > entry.meta.capacity {
            return Err(TvmError::OutOfBounds);
        }
        let memory = entry.memory.as_mut().ok_or(TvmError::NotResident)?;
        memory.write(handle.offset, data)?;
        entry.metrics.record_write(data.len() as u64);
        Ok(())
    }

    /// Cross-region copy. Acquires per-region locks in `region_id` ascending
    /// order; if both ends are the same region, only one lock is taken.
    pub fn cross_region_copy(
        &self,
        src_region: u16,
        src_offset: u32,
        dst_region: u16,
        dst_offset: u32,
        len: u32,
    ) -> Result<()> {
        let src_arc = self.entry_arc(src_region)?;
        let mut buf = vec![0u8; len as usize];
        if src_region == dst_region {
            let mut entry = src_arc.lock().map_err(|_| poisoned())?;
            check_in_bounds(&entry, src_offset, len)?;
            let memory = entry.memory.as_ref().ok_or(TvmError::NotResident)?;
            memory.read(src_offset, &mut buf)?;
            entry.metrics.record_read(len as u64);
            check_in_bounds(&entry, dst_offset, len)?;
            let memory = entry.memory.as_mut().ok_or(TvmError::NotResident)?;
            memory.write(dst_offset, &buf)?;
            entry.metrics.record_write(len as u64);
            return Ok(());
        }
        let dst_arc = self.entry_arc(dst_region)?;
        // Lock in ascending region_id order to avoid deadlock.
        let (lo_arc, hi_arc, lo_id) = if src_region < dst_region {
            (&src_arc, &dst_arc, src_region)
        } else {
            (&dst_arc, &src_arc, dst_region)
        };
        let mut lo = lo_arc.lock().map_err(|_| poisoned())?;
        let mut hi = hi_arc.lock().map_err(|_| poisoned())?;
        let (src_lock, dst_lock) = if lo_id == src_region {
            (&mut *lo, &mut *hi)
        } else {
            (&mut *hi, &mut *lo)
        };
        check_in_bounds(src_lock, src_offset, len)?;
        let memory = src_lock.memory.as_ref().ok_or(TvmError::NotResident)?;
        memory.read(src_offset, &mut buf)?;
        src_lock.metrics.record_read(len as u64);

        check_in_bounds(dst_lock, dst_offset, len)?;
        let memory = dst_lock.memory.as_mut().ok_or(TvmError::NotResident)?;
        memory.write(dst_offset, &buf)?;
        dst_lock.metrics.record_write(len as u64);
        Ok(())
    }

    pub fn region_info(&self, region_id: u16) -> Result<Region> {
        let entry_arc = self.entry_arc(region_id)?;
        let entry = entry_arc.lock().map_err(|_| poisoned())?;
        Ok(entry.meta)
    }

    pub fn metrics_snapshot(&self, region_id: u16) -> Result<MetricsSnapshot> {
        let entry_arc = self.entry_arc(region_id)?;
        let entry = entry_arc.lock().map_err(|_| poisoned())?;
        Ok(entry.metrics.snapshot())
    }

    pub fn list_regions(&self) -> Result<Vec<Region>> {
        let regions = self.regions.read().map_err(|_| poisoned())?;
        let arcs: Vec<SharedEntry<M>> =
            regions.iter().filter_map(|s| s.as_ref().cloned()).collect();
        drop(regions);
        let mut out = Vec::with_capacity(arcs.len());
        for arc in arcs {
            let entry = arc.lock().map_err(|_| poisoned())?;
            out.push(entry.meta);
        }
        Ok(out)
    }

    pub fn pin(&self, region_id: u16) -> Result<()> {
        let entry_arc = self.entry_arc(region_id)?;
        let mut entry = entry_arc.lock().map_err(|_| poisoned())?;
        if !entry.meta.pinnable {
            return Err(TvmError::PolicyViolation);
        }
        entry.meta.pinned = true;
        Ok(())
    }

    pub fn unpin(&self, region_id: u16) -> Result<()> {
        let entry_arc = self.entry_arc(region_id)?;
        let mut entry = entry_arc.lock().map_err(|_| poisoned())?;
        entry.meta.pinned = false;
        Ok(())
    }

    pub fn spill_region<B: BackingStore>(&self, region_id: u16, store: &mut B) -> Result<()> {
        let entry_arc = self.entry_arc(region_id)?;
        let mut entry = entry_arc.lock().map_err(|_| poisoned())?;
        if entry.meta.pinned {
            return Err(TvmError::Pinned);
        }
        if !entry.meta.spillable {
            return Err(TvmError::PolicyViolation);
        }
        if entry.meta.residency == Residency::Cold {
            return Ok(());
        }
        let memory = entry.memory.take().ok_or(TvmError::NotResident)?;
        let bytes = memory.snapshot();
        store.spill(entry.meta.id, entry.meta.generation, &bytes)?;
        entry.meta.residency = Residency::Cold;
        entry.metrics.record_demotion();
        drop(entry);
        self.lru_remove(region_id);
        Ok(())
    }

    /// Compact a region under per-region locking. Takes the outer regions
    /// read-lock (no membership change), then the per-region mutex. Old
    /// handles fail validation immediately because the generation bumps
    /// while the lock is held.
    pub fn compact_region(&self, region_id: u16) -> Result<HandleRemap>
    where
        M: MemoryRegion,
    {
        let entry_arc = self.entry_arc(region_id)?;
        let mut entry = entry_arc.lock().map_err(|_| poisoned())?;
        if entry.meta.pinned {
            return Err(TvmError::Pinned);
        }
        let blocks = entry
            .allocator
            .allocated_blocks()
            .ok_or(TvmError::UnsupportedAllocator)?;
        let memory = entry.memory.as_ref().ok_or(TvmError::NotResident)?;
        let mut new_data = vec![0u8; entry.meta.capacity as usize];
        let mut mapping: hashbrown::HashMap<u32, u32> =
            hashbrown::HashMap::with_capacity(blocks.len());
        let mut new_blocks = Vec::with_capacity(blocks.len());
        let mut cursor = 0u32;
        for (old_off, size) in blocks {
            let mut buf = vec![0u8; size as usize];
            memory.read(old_off, &mut buf)?;
            let dst = cursor as usize;
            new_data[dst..dst + size as usize].copy_from_slice(&buf);
            mapping.insert(old_off, cursor);
            new_blocks.push((cursor, size));
            cursor += size;
        }
        let old_gen = entry.meta.generation;
        let mut next = entry.meta.generation.wrapping_add(1);
        if next == 0 {
            next = 1;
        }
        entry.meta.generation = next;
        entry.memory = Some(M::restore(new_data));
        let cap = entry.meta.capacity;
        entry.allocator.rebuild_after_compact(&new_blocks, cap);
        entry.meta.used = entry.allocator.used();
        Ok(HandleRemap {
            region_id,
            old_generation: old_gen,
            new_generation: next,
            mapping,
        })
    }

    pub fn load_region<B: BackingStore>(&self, region_id: u16, store: &mut B) -> Result<()>
    where
        M: MemoryRegion,
    {
        let entry_arc = self.entry_arc(region_id)?;
        let mut entry = entry_arc.lock().map_err(|_| poisoned())?;
        if entry.meta.residency == Residency::Hot && entry.memory.is_some() {
            return Ok(());
        }
        let bytes = store.load(entry.meta.id, entry.meta.generation)?;
        entry.memory = Some(M::restore(bytes));
        entry.meta.residency = Residency::Hot;
        entry.metrics.record_promotion();
        entry.metrics.record_fault();
        Ok(())
    }

    /// Evict regions until total resident bytes (sum of `used` over
    /// `Hot` + `Warm` regions) is at or below `target`.
    ///
    /// Semantics are documented on [`crate::eviction`]. In brief:
    /// `target` is absolute (a residency ceiling, not a delta);
    /// pinned and non-spillable regions are silently skipped;
    /// `target_met=false` is a successful `Ok` outcome, not an
    /// error.
    pub fn demote_until<B: BackingStore>(
        &self,
        target: u64,
        policy: EvictionPolicy,
        store: &mut B,
    ) -> Result<EvictionReport> {
        let mut report = EvictionReport::default();

        // Snapshot residency state. `list_regions` clones the meta
        // structs out, releasing the per-entry locks before we
        // start spilling — so the spill calls below take their own
        // locks without nesting.
        let snapshot = self.list_regions()?;
        let total_resident = |regs: &[Region]| -> u64 {
            regs.iter()
                .filter(|r| counts_toward_resident(r.residency))
                .map(|r| r.used as u64)
                .sum()
        };
        let mut current = total_resident(&snapshot);
        if current <= target {
            report.target_met = true;
            return Ok(report);
        }

        // Bucket eligible candidates by tier. Warm before Hot
        // ("coldest first") so we touch the colder layer first.
        // External and Cold contribute 0 to resident bytes and are
        // never visited. Pinned / non-spillable are filtered here
        // — and again at the spill site (race-safe).
        let EvictionPolicy::ColdestFirst { within_tier } = policy;
        let mut warm: Vec<(u16, u32)> = Vec::new();
        let mut hot: Vec<(u16, u32)> = Vec::new();
        for r in &snapshot {
            if r.pinned || !r.spillable {
                continue;
            }
            match r.residency {
                Residency::Warm => warm.push((r.id, r.used)),
                Residency::Hot => hot.push((r.id, r.used)),
                _ => {}
            }
        }
        let cmp = within_tier_cmp(within_tier);
        warm.sort_by(cmp);
        hot.sort_by(cmp);

        for (id, used) in warm.into_iter().chain(hot) {
            if current <= target {
                break;
            }
            // `spill_region` may legitimately fail with `Pinned`,
            // `PolicyViolation`, or `NotResident` if another thread
            // mutated state between our snapshot and the call.
            // These are not propagated as errors — they just mean
            // the candidate is no longer viable; continue.
            match self.spill_region(id, store) {
                Ok(()) => {
                    current = current.saturating_sub(used as u64);
                    report.bytes_freed = report.bytes_freed.saturating_add(used as u64);
                    report.regions_spilled = report.regions_spilled.saturating_add(1);
                }
                Err(TvmError::Pinned)
                | Err(TvmError::PolicyViolation)
                | Err(TvmError::NotResident)
                | Err(TvmError::RegionNotFound(_)) => {}
                Err(e) => return Err(e),
            }
        }

        report.target_met = current <= target;
        Ok(report)
    }

    /// Try `alloc(region_id, size)`; on `AllocationFailed`, run
    /// `demote_until(target, policy, store)` once, then retry the
    /// alloc. Returns the second alloc's error if it still fails
    /// (no further retries). Errors other than `AllocationFailed`
    /// (e.g. `RegionNotFound`, `StaleHandle`) short-circuit
    /// without invoking eviction.
    pub fn alloc_or_demote<B: BackingStore>(
        &self,
        region_id: u16,
        size: u32,
        target: u64,
        policy: EvictionPolicy,
        store: &mut B,
    ) -> Result<Handle> {
        match self.alloc(region_id, size) {
            Ok(h) => Ok(h),
            Err(TvmError::AllocationFailed) => {
                self.demote_until(target, policy, store)?;
                self.alloc(region_id, size)
            }
            Err(e) => Err(e),
        }
    }

    fn entry_arc(&self, region_id: u16) -> Result<SharedEntry<M>> {
        let regions = self.regions.read().map_err(|_| poisoned())?;
        regions
            .get(region_id as usize)
            .and_then(|s| s.as_ref().cloned())
            .ok_or(TvmError::RegionNotFound(region_id))
    }

    fn lru_push_front(&self, region_id: u16) {
        if let Ok(mut lru) = self.warm_lru.lock() {
            lru.retain(|id| *id != region_id);
            lru.push_front(region_id);
        }
    }

    fn lru_remove(&self, region_id: u16) {
        if let Ok(mut lru) = self.warm_lru.lock() {
            lru.retain(|id| *id != region_id);
        }
    }
}

fn check_in_bounds<M>(entry: &RegionEntry<M>, offset: u32, len: u32) -> Result<()> {
    let end = offset.checked_add(len).ok_or(TvmError::OutOfBounds)?;
    if end > entry.meta.capacity {
        return Err(TvmError::OutOfBounds);
    }
    Ok(())
}

fn poisoned() -> TvmError {
    TvmError::BackingStore("directory lock poisoned".into())
}
