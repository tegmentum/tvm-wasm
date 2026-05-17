use std::collections::{HashMap, VecDeque};

use crate::allocator::{AllocatorKind, RegionAllocator};
use crate::async_backing::AsyncBackingStore;
use crate::backing::BackingStore;
use crate::error::{Result, TvmError};
use crate::external::ExternalLoader;
use crate::handle::Handle;
use crate::metrics::RegionMetrics;
use crate::policy::PlacementPolicy;
use crate::region::{Region, RegionKind};
use crate::residency::Residency;

pub trait MemoryRegion {
    fn len(&self) -> u32;
    fn read(&self, offset: u32, buf: &mut [u8]) -> Result<()>;
    fn write(&mut self, offset: u32, buf: &[u8]) -> Result<()>;
    fn snapshot(&self) -> Vec<u8>;
    fn restore(bytes: Vec<u8>) -> Self
    where
        Self: Sized;
}

pub struct RegionEntry<M> {
    pub meta: Region,
    pub memory: Option<M>,
    pub metrics: RegionMetrics,
    pub allocator: RegionAllocator,
}

/// Mapping from old handles to new handles after compaction. Returned by
/// `compact_region`; callers must use `migrate` to rewrite any handles they
/// hold into the region. Old-generation handles fail validation immediately.
#[derive(Debug, Clone)]
pub struct HandleRemap {
    pub region_id: u16,
    pub old_generation: u16,
    pub new_generation: u16,
    pub mapping: HashMap<u32, u32>,
}

impl HandleRemap {
    pub fn migrate(&self, h: Handle) -> Option<Handle> {
        if h.region_id != self.region_id || h.generation != self.old_generation {
            return None;
        }
        let new_offset = self.mapping.get(&h.offset)?;
        Some(Handle {
            region_id: h.region_id,
            generation: self.new_generation,
            offset: *new_offset,
        })
    }
}

pub struct RegionDirectory<M> {
    // NOTE on storage choice: `Vec<Option<RegionEntry<M>>>` adds 8 bytes
    // per slot for the Option discriminant. Measured on Apple Silicon
    // with `VecBackedRegion`:
    //   - sizeof RegionEntry            ≈ 156 bytes
    //   - sizeof Option<RegionEntry>    ≈ 164 bytes  (8 bytes overhead)
    //
    // For 1,000 regions:  8 KiB overhead (fits in L1)
    // For 10,000 regions: 80 KiB overhead (crosses into L2 territory)
    //
    // A compact alternative (`Vec<MaybeUninit<RegionEntry>>` + alive
    // bitmap, or a `slab::Slab`) would save the discriminant but add
    // ~100 lines of unsafe code with subtle invariants. Skipped because:
    //   (a) The hot path goes through `ResolveCache` and never touches
    //       this Vec under load.
    //   (b) No workload in `bench-framework/` exercises >100 regions.
    //   (c) The unsafe burden is real; the perf win is sub-noise at our
    //       scales.
    //
    // Revisit if a real workload appears that creates 10K+ regions with
    // non-trivial churn.
    regions: Vec<Option<RegionEntry<M>>>,
    /// Region IDs that are currently `Warm`. Front = most recently demoted,
    /// back = LRU candidate for eviction.
    warm_lru: VecDeque<u16>,
}

impl<M: MemoryRegion> Default for RegionDirectory<M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M: MemoryRegion> RegionDirectory<M> {
    pub fn new() -> Self {
        Self {
            regions: Vec::new(),
            warm_lru: VecDeque::new(),
        }
    }

    pub(crate) fn lru_remove(&mut self, region_id: u16) {
        if let Some(pos) = self.warm_lru.iter().position(|id| *id == region_id) {
            self.warm_lru.remove(pos);
        }
    }

    fn lru_push_front(&mut self, region_id: u16) {
        self.lru_remove(region_id);
        self.warm_lru.push_front(region_id);
    }

    pub fn warm_lru_back(&self) -> Option<u16> {
        self.warm_lru.back().copied()
    }

    pub fn create_region(&mut self, kind: RegionKind, capacity: u32, memory: M) -> Result<u16> {
        self.create_region_with(kind, capacity, AllocatorKind::Bump, memory)
    }

    pub fn create_region_with(
        &mut self,
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

    /// Most general constructor. Lets the caller override the placement
    /// policy — useful for tests or for regions whose lifecycle differs from
    /// the kind's defaults.
    pub fn create_region_with_policy(
        &mut self,
        kind: RegionKind,
        capacity: u32,
        allocator: AllocatorKind,
        policy: PlacementPolicy,
        memory: M,
    ) -> Result<u16> {
        let id = u16::try_from(self.regions.len()).map_err(|_| TvmError::AllocationFailed)?;
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
        self.regions.push(Some(entry));
        // Warm-on-create regions (e.g. PageStore) are LRU candidates from the
        // start.
        if initial_residency == Residency::Warm {
            self.warm_lru.push_front(id);
        }
        Ok(id)
    }

    pub fn metrics(&self, region_id: u16) -> Result<&RegionMetrics> {
        self.entry(region_id).map(|e| &e.metrics)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Region> {
        self.regions
            .iter()
            .filter_map(|slot| slot.as_ref().map(|e| &e.meta))
    }

    pub fn len(&self) -> usize {
        self.regions.iter().filter(|s| s.is_some()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn spill_region<B: BackingStore>(&mut self, region_id: u16, store: &mut B) -> Result<()> {
        let entry = self.entry_mut(region_id)?;
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
        self.lru_remove(region_id);
        Ok(())
    }

    /// Async cousin of `spill_region`. Same semantics, but the underlying
    /// backing-store call is awaited rather than blocking. Use when the
    /// cold tier is I/O-bound (S3, network-attached storage, etc.).
    pub async fn spill_region_async<B: AsyncBackingStore>(
        &mut self,
        region_id: u16,
        store: &mut B,
    ) -> Result<()> {
        // Pre-flight checks + memory snapshot under the synchronous
        // borrow, then release the entry borrow before awaiting so the
        // future doesn't carry it across .await.
        let (id, generation, bytes) = {
            let entry = self.entry_mut(region_id)?;
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
            (entry.meta.id, entry.meta.generation, memory.snapshot())
        };
        // Now safe to await — no &mut self borrow held.
        store.spill_async(id, generation, &bytes).await?;
        // Reacquire and finish state transition.
        let entry = self.entry_mut(region_id)?;
        entry.meta.residency = Residency::Cold;
        entry.metrics.record_demotion();
        self.lru_remove(region_id);
        Ok(())
    }

    /// Compact a region by packing all live allocations contiguously starting
    /// at offset 0. Bumps the region's generation, invalidating any handle
    /// held by callers. The returned `HandleRemap` lets them rewrite stale
    /// handles into the new layout. Only freelist regions can be compacted.
    pub fn compact_region(&mut self, region_id: u16) -> Result<HandleRemap>
    where
        M: MemoryRegion,
    {
        let entry = self.entry_mut(region_id)?;
        if entry.meta.pinned {
            return Err(TvmError::Pinned);
        }
        let blocks = entry
            .allocator
            .allocated_blocks()
            .ok_or(TvmError::UnsupportedAllocator)?;

        let memory = entry.memory.as_ref().ok_or(TvmError::NotResident)?;
        let mut new_data = vec![0u8; entry.meta.capacity as usize];
        let mut mapping = HashMap::with_capacity(blocks.len());
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

    /// Load an External-tier region by calling the supplied loader. The
    /// loader returns the full byte contents of the region; the directory
    /// installs them via `M::restore` and transitions the region to Hot.
    pub fn load_external_region(&mut self, region_id: u16, loader: &ExternalLoader) -> Result<()>
    where
        M: MemoryRegion,
    {
        let entry = self.entry_mut(region_id)?;
        if entry.meta.residency != Residency::External {
            return Ok(());
        }
        let bytes = loader(entry.meta.id, entry.meta.generation)?;
        entry.memory = Some(M::restore(bytes));
        entry.meta.residency = Residency::Hot;
        entry.metrics.record_promotion();
        entry.metrics.record_fault();
        Ok(())
    }

    /// Mark a region as External — its memory is dropped and the next
    /// access faults through the supplied loader. Pinned regions are
    /// rejected.
    pub fn mark_external(&mut self, region_id: u16) -> Result<()> {
        let entry = self.entry_mut(region_id)?;
        if entry.meta.pinned {
            return Err(TvmError::Pinned);
        }
        entry.memory = None;
        entry.meta.residency = Residency::External;
        self.lru_remove(region_id);
        Ok(())
    }

    /// Move a region toward the Hot tier. Cold → Hot loads from the backing
    /// store; Warm → Hot just clears the LRU eligibility flag; Hot is a no-op.
    /// Records a promotion in the metrics and updates the cache invalidation
    /// is the caller's responsibility (TvmHost handles it).
    pub fn promote_region<B: BackingStore>(&mut self, region_id: u16, backing: &mut B) -> Result<()>
    where
        M: MemoryRegion,
    {
        let current = self.entry(region_id)?.meta.residency;
        match current {
            Residency::Hot => Ok(()),
            Residency::Warm => {
                let entry = self.entry_mut(region_id)?;
                entry.meta.residency = Residency::Hot;
                entry.metrics.record_promotion();
                self.lru_remove(region_id);
                Ok(())
            }
            Residency::Cold => {
                self.load_region(region_id, backing)?;
                Ok(())
            }
            Residency::External => Err(TvmError::NotResident),
        }
    }

    /// Move a region one tier toward Cold. Hot → Warm marks the region as
    /// LRU-eligible (still resident, but available for eviction). Warm → Cold
    /// spills to the backing store. Cold and pinned regions are rejected.
    pub fn demote_region<B: BackingStore>(&mut self, region_id: u16, backing: &mut B) -> Result<()>
    where
        M: MemoryRegion,
    {
        let entry = self.entry(region_id)?;
        if entry.meta.pinned {
            return Err(TvmError::Pinned);
        }
        if !entry.meta.spillable {
            return Err(TvmError::PolicyViolation);
        }
        match entry.meta.residency {
            Residency::Hot => {
                let entry = self.entry_mut(region_id)?;
                entry.meta.residency = Residency::Warm;
                entry.metrics.record_demotion();
                self.lru_push_front(region_id);
                Ok(())
            }
            Residency::Warm => self.spill_region(region_id, backing),
            Residency::Cold => Ok(()),
            Residency::External => Err(TvmError::NotResident),
        }
    }

    /// Evict the oldest warm region by spilling it to the backing store.
    /// Returns `Some(id)` if a region was evicted, `None` if no warm regions
    /// are available. Skips pinned regions transparently.
    pub fn evict_warm_region<B: BackingStore>(&mut self, backing: &mut B) -> Result<Option<u16>>
    where
        M: MemoryRegion,
    {
        // Scan from oldest to newest; skip pinned ones (which shouldn't be in
        // the LRU but defend against stale entries) until we find one we can
        // spill.
        let snapshot: Vec<u16> = self.warm_lru.iter().rev().copied().collect();
        for id in snapshot {
            let entry = match self.entry(id) {
                Ok(e) => e,
                Err(_) => {
                    self.lru_remove(id);
                    continue;
                }
            };
            if entry.meta.pinned || entry.meta.residency != Residency::Warm {
                self.lru_remove(id);
                continue;
            }
            self.spill_region(id, backing)?;
            return Ok(Some(id));
        }
        Ok(None)
    }

    /// Read with auto-fault: if the source region is Cold, load it from the
    /// backing store first (recording a fault), then read. Falls back to an
    /// error if the region is Cold and no backing is provided via the
    /// `read` method.
    pub fn read_or_fault<B: BackingStore>(
        &mut self,
        handle: Handle,
        buf: &mut [u8],
        backing: &mut B,
    ) -> Result<()>
    where
        M: MemoryRegion,
    {
        let cold = {
            let entry = self.entry(handle.region_id)?;
            entry.meta.residency == Residency::Cold
        };
        if cold {
            self.load_region(handle.region_id, backing)?;
        }
        self.read(handle, buf)
    }

    /// Write with auto-fault. Same semantics as `read_or_fault` but for
    /// writes — Cold regions are auto-loaded so the write lands on real bytes
    /// instead of failing with `NotResident`.
    pub fn write_or_fault<B: BackingStore>(
        &mut self,
        handle: Handle,
        data: &[u8],
        backing: &mut B,
    ) -> Result<()>
    where
        M: MemoryRegion,
    {
        let cold = {
            let entry = self.entry(handle.region_id)?;
            entry.meta.residency == Residency::Cold
        };
        if cold {
            self.load_region(handle.region_id, backing)?;
        }
        self.write(handle, data)
    }

    pub fn pin(&mut self, region_id: u16) -> Result<()> {
        let entry = self.entry_mut(region_id)?;
        if !entry.meta.pinnable {
            return Err(TvmError::PolicyViolation);
        }
        entry.meta.pinned = true;
        Ok(())
    }

    pub fn unpin(&mut self, region_id: u16) -> Result<()> {
        self.entry_mut(region_id)?.meta.pinned = false;
        Ok(())
    }

    /// Snapshot a region to the given file path. The file holds the raw bytes
    /// of the memory; metadata (kind, generation, allocator state) is not
    /// recorded. Use `restore_region` against a freshly-created region with
    /// matching capacity.
    pub fn snapshot_region(&self, region_id: u16, path: impl AsRef<std::path::Path>) -> Result<()>
    where
        M: MemoryRegion,
    {
        let entry = self.entry(region_id)?;
        let memory = entry.memory.as_ref().ok_or(TvmError::NotResident)?;
        let bytes = memory.snapshot();
        std::fs::write(path, bytes).map_err(|e| TvmError::BackingStore(e.to_string()))?;
        Ok(())
    }

    /// Replace a region's contents from a file written by `snapshot_region`.
    /// The file must be no larger than the region's capacity.
    pub fn restore_region(
        &mut self,
        region_id: u16,
        path: impl AsRef<std::path::Path>,
    ) -> Result<()>
    where
        M: MemoryRegion,
    {
        let bytes = std::fs::read(path).map_err(|e| TvmError::BackingStore(e.to_string()))?;
        let entry = self.entry_mut(region_id)?;
        if bytes.len() as u32 > entry.meta.capacity {
            return Err(TvmError::OutOfBounds);
        }
        entry.memory = Some(M::restore(bytes));
        entry.meta.residency = Residency::Hot;
        Ok(())
    }

    pub fn load_region<B: BackingStore>(&mut self, region_id: u16, store: &mut B) -> Result<()>
    where
        M: MemoryRegion,
    {
        let entry = self.entry_mut(region_id)?;
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

    /// Async cousin of `load_region`.
    pub async fn load_region_async<B: AsyncBackingStore>(
        &mut self,
        region_id: u16,
        store: &mut B,
    ) -> Result<()>
    where
        M: MemoryRegion,
    {
        // Read meta under sync borrow, drop, await, reacquire.
        let (id, generation, already_hot) = {
            let entry = self.entry_mut(region_id)?;
            (
                entry.meta.id,
                entry.meta.generation,
                entry.meta.residency == Residency::Hot && entry.memory.is_some(),
            )
        };
        if already_hot {
            return Ok(());
        }
        let bytes = store.load_async(id, generation).await?;
        let entry = self.entry_mut(region_id)?;
        entry.memory = Some(M::restore(bytes));
        entry.meta.residency = Residency::Hot;
        entry.metrics.record_promotion();
        entry.metrics.record_fault();
        Ok(())
    }

    /// Async auto-fault read: if Cold, awaits a load via the async backing
    /// before completing the access. Sync flow is unchanged via
    /// `read_or_fault`.
    pub async fn read_or_fault_async<B: AsyncBackingStore>(
        &mut self,
        handle: Handle,
        buf: &mut [u8],
        backing: &mut B,
    ) -> Result<()>
    where
        M: MemoryRegion,
    {
        let cold = {
            let entry = self.entry(handle.region_id)?;
            entry.meta.residency == Residency::Cold
        };
        if cold {
            self.load_region_async(handle.region_id, backing).await?;
        }
        self.read(handle, buf)
    }

    /// Async auto-fault write — symmetric to `read_or_fault_async`.
    pub async fn write_or_fault_async<B: AsyncBackingStore>(
        &mut self,
        handle: Handle,
        data: &[u8],
        backing: &mut B,
    ) -> Result<()>
    where
        M: MemoryRegion,
    {
        let cold = {
            let entry = self.entry(handle.region_id)?;
            entry.meta.residency == Residency::Cold
        };
        if cold {
            self.load_region_async(handle.region_id, backing).await?;
        }
        self.write(handle, data)
    }

    pub fn destroy_region(&mut self, region_id: u16) -> Result<()> {
        let slot = self
            .regions
            .get_mut(region_id as usize)
            .ok_or(TvmError::RegionNotFound(region_id))?;
        if slot.is_none() {
            return Err(TvmError::RegionNotFound(region_id));
        }
        *slot = None;
        self.lru_remove(region_id);
        Ok(())
    }

    pub fn region_info(&self, region_id: u16) -> Result<&Region> {
        self.entry(region_id).map(|e| &e.meta)
    }

    pub fn alloc(&mut self, region_id: u16, size: u32) -> Result<Handle> {
        self.alloc_aligned(region_id, size, 1)
    }

    pub fn alloc_aligned(&mut self, region_id: u16, size: u32, align: u32) -> Result<Handle> {
        let entry = self.entry_mut(region_id)?;
        let cap = entry.meta.capacity;
        let used = entry.allocator.used();
        let offset = entry.allocator.alloc(size, align).map_err(|e| {
            crate::error::set_last_error_context(crate::error::ErrorContext {
                region_id: Some(region_id),
                len: Some(size),
                capacity: Some(cap),
                note: Some("alloc: requested size exceeds available"),
                ..Default::default()
            });
            let _ = used;
            e
        })?;
        entry.meta.used = entry.allocator.used();
        entry.metrics.record_alloc(size as u64);
        Ok(Handle {
            region_id,
            generation: entry.meta.generation,
            offset,
        })
    }

    pub fn dealloc(&mut self, handle: Handle) -> Result<()> {
        let entry = self.validate_mut(handle)?;
        entry.allocator.dealloc(handle.offset)?;
        entry.meta.used = entry.allocator.used();
        Ok(())
    }

    pub fn read(&self, handle: Handle, buf: &mut [u8]) -> Result<()> {
        let entry = self.validate(handle)?;
        let memory = entry.memory.as_ref().ok_or(TvmError::NotResident)?;
        let end = handle.offset.checked_add(buf.len() as u32).ok_or_else(|| {
            crate::error::set_last_error_context(crate::error::ErrorContext {
                region_id: Some(handle.region_id),
                generation: Some(handle.generation),
                offset: Some(handle.offset),
                len: Some(buf.len() as u32),
                note: Some("read: offset+len overflow u32"),
                ..Default::default()
            });
            TvmError::OutOfBounds
        })?;
        if end > entry.meta.capacity {
            crate::error::set_last_error_context(crate::error::ErrorContext {
                region_id: Some(handle.region_id),
                generation: Some(handle.generation),
                offset: Some(handle.offset),
                len: Some(buf.len() as u32),
                capacity: Some(entry.meta.capacity),
                note: Some("read: end > capacity"),
            });
            return Err(TvmError::OutOfBounds);
        }
        memory.read(handle.offset, buf)?;
        entry.metrics.record_read(buf.len() as u64);
        Ok(())
    }

    pub fn write(&mut self, handle: Handle, data: &[u8]) -> Result<()> {
        let entry = self.validate_mut(handle)?;
        let memory = entry.memory.as_mut().ok_or(TvmError::NotResident)?;
        let end = handle
            .offset
            .checked_add(data.len() as u32)
            .ok_or(TvmError::OutOfBounds)?;
        if end > entry.meta.capacity {
            return Err(TvmError::OutOfBounds);
        }
        memory.write(handle.offset, data)?;
        entry.metrics.record_write(data.len() as u64);
        Ok(())
    }

    /// Copy bytes from one region to another (or within the same region) in
    /// host memory, without round-tripping through guest linear memory.
    pub fn cross_region_copy(
        &mut self,
        src_region: u16,
        src_offset: u32,
        dst_region: u16,
        dst_offset: u32,
        len: u32,
    ) -> Result<()> {
        let mut buf = vec![0u8; len as usize];
        {
            let entry = self.entry(src_region)?;
            let memory = entry.memory.as_ref().ok_or(TvmError::NotResident)?;
            let end = src_offset.checked_add(len).ok_or(TvmError::OutOfBounds)?;
            if end > entry.meta.capacity {
                return Err(TvmError::OutOfBounds);
            }
            memory.read(src_offset, &mut buf)?;
            entry.metrics.record_read(len as u64);
        }
        let entry = self.entry_mut(dst_region)?;
        let memory = entry.memory.as_mut().ok_or(TvmError::NotResident)?;
        let end = dst_offset.checked_add(len).ok_or(TvmError::OutOfBounds)?;
        if end > entry.meta.capacity {
            return Err(TvmError::OutOfBounds);
        }
        memory.write(dst_offset, &buf)?;
        entry.metrics.record_write(len as u64);
        Ok(())
    }

    /// Read `len` bytes from `src` into `(dst_region, dst_offset)`. Validates
    /// the source handle's generation; the destination is treated as a raw
    /// region offset (use a fresh handle's offset if you want generation
    /// safety on both sides).
    pub fn read_into(
        &mut self,
        src: Handle,
        dst_region: u16,
        dst_offset: u32,
        len: u32,
    ) -> Result<()> {
        // Generation check on source.
        let src_entry = self.entry(src.region_id)?;
        if src_entry.meta.generation != src.generation {
            return Err(TvmError::StaleHandle);
        }
        self.cross_region_copy(src.region_id, src.offset, dst_region, dst_offset, len)
    }

    /// Write `len` bytes from `(src_region, src_offset)` into `dst`. Validates
    /// the destination handle's generation.
    pub fn write_from(
        &mut self,
        src_region: u16,
        src_offset: u32,
        dst: Handle,
        len: u32,
    ) -> Result<()> {
        let dst_entry = self.entry(dst.region_id)?;
        if dst_entry.meta.generation != dst.generation {
            return Err(TvmError::StaleHandle);
        }
        self.cross_region_copy(src_region, src_offset, dst.region_id, dst.offset, len)
    }

    pub fn bump_generation(&mut self, region_id: u16) -> Result<u16> {
        let entry = self.entry_mut(region_id)?;
        entry.meta.generation = entry.meta.generation.wrapping_add(1);
        if entry.meta.generation == 0 {
            entry.meta.generation = 1;
        }
        Ok(entry.meta.generation)
    }

    pub(crate) fn entry(&self, region_id: u16) -> Result<&RegionEntry<M>> {
        self.regions
            .get(region_id as usize)
            .and_then(|x| x.as_ref())
            .ok_or(TvmError::RegionNotFound(region_id))
    }

    pub(crate) fn entry_mut(&mut self, region_id: u16) -> Result<&mut RegionEntry<M>> {
        self.regions
            .get_mut(region_id as usize)
            .and_then(|x| x.as_mut())
            .ok_or(TvmError::RegionNotFound(region_id))
    }

    pub(crate) fn validate(&self, handle: Handle) -> Result<&RegionEntry<M>> {
        let entry = self.entry(handle.region_id)?;
        if entry.meta.generation != handle.generation {
            return Err(TvmError::StaleHandle);
        }
        Ok(entry)
    }

    pub(crate) fn validate_mut(&mut self, handle: Handle) -> Result<&mut RegionEntry<M>> {
        let entry = self.entry_mut(handle.region_id)?;
        if entry.meta.generation != handle.generation {
            return Err(TvmError::StaleHandle);
        }
        Ok(entry)
    }
}

// `RegionDirectory<VecBackedRegion>` specialized slice access lives in
// `directory_slices.rs` for file-size hygiene.
