//! `MultiGuestTvm` — fan-out across N child `GuestTvm` instances so the
//! addressable working set scales beyond a single guest's `pool_count ×
//! 4 GiB` ceiling.
//!
//! ## How it works
//!
//! - Owns a `Vec<GuestTvm>` (the "shards"). Each shard is a complete
//!   independent guest-side TVM with its own pools.
//! - Maintains an **external region id space** (u16, like a regular
//!   `GuestTvm`) and translates external_region_id → (shard_index,
//!   inner_region_id) at every operation.
//! - Implements `TvmFacade` so generic code (`fn workload<T:
//!   TvmFacade>`) sees no difference between a single guest and a
//!   sharded fan-out.
//!
//! ## What this buys
//!
//! - **No fixed addressable ceiling.** N shards × 64 pools × 4 GiB =
//!   N × 256 GiB. Add shards to grow the cap.
//! - **Independent failure domains.** A pool-exhausted error in shard 3
//!   doesn't prevent allocations in shard 7.
//! - **Same access pattern.** Each shard still exposes the
//!   bulk-copy-via-`memory.copy` fast path; bandwidth per region is
//!   unchanged.
//!
//! ## What this does NOT solve
//!
//! - **Cross-shard `memory.copy` is impossible.** A shard cannot
//!   directly copy bytes to/from another shard's wasm memory — that's
//!   a wasm-engine boundary. If a workload needs to move bytes
//!   across shards it goes through host RAM (read from shard A → write
//!   to shard B). For workloads where each region is consumed by one
//!   reader at a time this is invisible.
//! - **Region IDs are still u16.** Shard count × inner regions per
//!   shard must stay under 65535 total live regions. Plenty for any
//!   realistic deployment but worth knowing.
//!
//! ## Placement policy
//!
//! Round-robin by default. The first shard with sufficient remaining
//! capacity wins. A future revision could honor `RegionKind` hints
//! (e.g. "place hot regions on shard 0, cold on shard N-1") or a custom
//! scoring callback — both fit cleanly because all routing already
//! happens here.

use tvm_core::{
    Handle, Region, RegionKind, Result, TvmError, TvmFacade,
};

use crate::facade::GuestTvm;

/// Shard index. Just an alias for clarity in signatures.
pub type ShardId = u32;

/// Maps external region_id → (shard, inner_region_id). External IDs are
/// allocated densely from 1 (0 reserved for null), so a `Vec` works as
/// a perfect-hash table indexed by id.
struct RegionMap {
    /// `slots[id]` = `Some((shard, inner_id))` when allocated. `None`
    /// after dealloc-of-region (not implemented yet — regions live for
    /// the lifetime of the multi guest).
    slots: Vec<Option<(ShardId, u16)>>,
    next_id: u16,
}

impl RegionMap {
    fn new() -> Self {
        Self { slots: vec![None], next_id: 1 } // index 0 reserved
    }

    fn insert(&mut self, shard: ShardId, inner: u16) -> Result<u16> {
        let id = self.next_id;
        self.next_id = self.next_id.checked_add(1).ok_or(TvmError::AllocationFailed)?;
        self.slots.push(Some((shard, inner)));
        Ok(id)
    }

    fn lookup(&self, id: u16) -> Result<(ShardId, u16)> {
        self.slots
            .get(id as usize)
            .copied()
            .flatten()
            .ok_or(TvmError::RegionNotFound(id))
    }
}

/// Multi-shard guest TVM. Implements `TvmFacade`, fans operations out
/// to a child `GuestTvm` per shard.
pub struct MultiGuestTvm {
    shards: Vec<GuestTvm>,
    map: RegionMap,
    placement_cursor: ShardId,
}

impl MultiGuestTvm {
    pub fn new(shards: Vec<GuestTvm>) -> Self {
        assert!(!shards.is_empty(), "MultiGuestTvm requires at least one shard");
        Self { shards, map: RegionMap::new(), placement_cursor: 0 }
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Look up which shard a region lives on. Useful for placement
    /// debugging and stat collection.
    pub fn shard_of(&self, region: u16) -> Result<ShardId> {
        self.map.lookup(region).map(|(s, _)| s)
    }

    /// Try each shard in round-robin order; first one to accept the
    /// `create_region` call wins. Returns (shard_index, inner_region_id).
    fn place(&mut self, kind: RegionKind, capacity: u32) -> Result<(ShardId, u16)> {
        let n = self.shards.len() as ShardId;
        for offset in 0..n {
            let idx = (self.placement_cursor + offset) % n;
            match self.shards[idx as usize].create_region(kind, capacity) {
                Ok(inner) => {
                    self.placement_cursor = (idx + 1) % n;
                    return Ok((idx, inner));
                }
                Err(TvmError::AllocationFailed) => continue,
                Err(e) => return Err(e),
            }
        }
        Err(TvmError::AllocationFailed)
    }

    /// Build an inner-shard handle from an external one by swapping the
    /// region_id. Generation and offset pass through unchanged.
    fn inner_handle(&self, h: Handle) -> Result<(ShardId, Handle)> {
        let (shard, inner_region) = self.map.lookup(h.region_id)?;
        Ok((
            shard,
            Handle {
                region_id: inner_region,
                generation: h.generation,
                offset: h.offset,
            },
        ))
    }
}

impl TvmFacade for MultiGuestTvm {
    fn create_region(&mut self, kind: RegionKind, capacity: u32) -> Result<u16> {
        let (shard, inner) = self.place(kind, capacity)?;
        self.map.insert(shard, inner)
    }

    fn alloc(&mut self, region: u16, size: u32) -> Result<Handle> {
        let (shard, inner_region) = self.map.lookup(region)?;
        let inner_handle = self.shards[shard as usize].alloc(inner_region, size)?;
        // Re-stamp with the external region_id so callers see consistent
        // ids end-to-end.
        Ok(Handle {
            region_id: region,
            generation: inner_handle.generation,
            offset: inner_handle.offset,
        })
    }

    fn dealloc(&mut self, handle: Handle) -> Result<()> {
        let (shard, inner) = self.inner_handle(handle)?;
        self.shards[shard as usize].dealloc(inner)
    }

    fn read(&mut self, handle: Handle, buf: &mut [u8]) -> Result<()> {
        let (shard, inner) = self.inner_handle(handle)?;
        self.shards[shard as usize].read(inner, buf)
    }

    fn write(&mut self, handle: Handle, data: &[u8]) -> Result<()> {
        let (shard, inner) = self.inner_handle(handle)?;
        self.shards[shard as usize].write(inner, data)
    }

    fn pin(&mut self, region: u16) -> Result<()> {
        let (shard, inner) = self.map.lookup(region)?;
        self.shards[shard as usize].pin(inner)
    }

    fn unpin(&mut self, region: u16) -> Result<()> {
        let (shard, inner) = self.map.lookup(region)?;
        self.shards[shard as usize].unpin(inner)
    }

    fn region_info(&self, region: u16) -> Result<Region> {
        let (shard, inner) = self.map.lookup(region)?;
        let mut info = self.shards[shard as usize].region_info(inner)?;
        // Surface the external region_id to the caller. Internal id is
        // an implementation detail.
        info.id = region;
        Ok(info)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::facade::Dispatch;
    use crate::directory::Pool;
    use std::sync::Mutex;

    // Stub backing shared across shards. Each shard claims a contiguous
    // slice of pool indices; the stub treats them as one flat namespace.
    static STUB_POOLS: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

    fn stub_read(pool: u32, off: u32, dst: &mut [u8]) -> Result<()> {
        let pools = STUB_POOLS.lock().unwrap();
        let p = &pools[pool as usize];
        dst.copy_from_slice(&p[off as usize..off as usize + dst.len()]);
        Ok(())
    }

    fn stub_write(pool: u32, off: u32, src: &[u8]) -> Result<()> {
        let mut pools = STUB_POOLS.lock().unwrap();
        let p = &mut pools[pool as usize];
        p[off as usize..off as usize + src.len()].copy_from_slice(src);
        Ok(())
    }

    fn stub_intra_pool_copy(pool: u32, dst_off: u32, src_off: u32, len: u32) -> Result<()> {
        let mut pools = STUB_POOLS.lock().unwrap();
        let p = &mut pools[pool as usize];
        p.copy_within(src_off as usize..src_off as usize + len as usize, dst_off as usize);
        Ok(())
    }

    fn build(n_shards: usize, pools_per_shard: usize, capacity: u32) -> MultiGuestTvm {
        let total_pools = n_shards * pools_per_shard;
        let mut g = STUB_POOLS.lock().unwrap();
        g.clear();
        for _ in 0..total_pools {
            g.push(vec![0u8; capacity as usize]);
        }
        drop(g);

        let shards: Vec<GuestTvm> = (0..n_shards)
            .map(|s| {
                let pool_descs: Vec<Pool> = (0..pools_per_shard)
                    .map(|i| Pool {
                        memory_index: (s * pools_per_shard + i) as u32,
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
            })
            .collect();
        MultiGuestTvm::new(shards)
    }

    #[test]
    fn round_trip_via_multi() {
        let mut m = build(3, 2, 4096);
        let r = m.create_region(RegionKind::HotHeap, 1024).unwrap();
        let h = m.alloc(r, 16).unwrap();
        m.write(h, b"multi-guest-test").unwrap();
        let mut buf = [0u8; 16];
        m.read(h, &mut buf).unwrap();
        assert_eq!(&buf, b"multi-guest-test");
    }

    #[test]
    fn places_round_robin_across_shards() {
        let mut m = build(4, 2, 4096);
        let r0 = m.create_region(RegionKind::HotHeap, 256).unwrap();
        let r1 = m.create_region(RegionKind::HotHeap, 256).unwrap();
        let r2 = m.create_region(RegionKind::HotHeap, 256).unwrap();
        let r3 = m.create_region(RegionKind::HotHeap, 256).unwrap();
        assert_eq!(m.shard_of(r0).unwrap(), 0);
        assert_eq!(m.shard_of(r1).unwrap(), 1);
        assert_eq!(m.shard_of(r2).unwrap(), 2);
        assert_eq!(m.shard_of(r3).unwrap(), 3);
    }

    #[test]
    fn falls_through_to_next_shard_when_one_is_full() {
        // Tiny shards: each holds exactly one region of 100 bytes.
        let mut m = build(3, 1, 100);
        m.create_region(RegionKind::HotHeap, 100).unwrap(); // shard 0
        m.create_region(RegionKind::HotHeap, 100).unwrap(); // shard 1
        let r2 = m.create_region(RegionKind::HotHeap, 100).unwrap(); // shard 2
        assert_eq!(m.shard_of(r2).unwrap(), 2);
        // No room anywhere now.
        assert!(matches!(
            m.create_region(RegionKind::HotHeap, 1),
            Err(TvmError::AllocationFailed)
        ));
    }

    #[test]
    fn region_info_surfaces_external_id() {
        let mut m = build(2, 2, 4096);
        let r = m.create_region(RegionKind::HotHeap, 256).unwrap();
        let info = m.region_info(r).unwrap();
        assert_eq!(info.id, r); // external id, not the inner shard's id
    }

    #[test]
    fn pin_routes_to_correct_shard() {
        let mut m = build(2, 1, 4096);
        let r = m.create_region(RegionKind::HotHeap, 256).unwrap();
        m.pin(r).unwrap();
        assert!(m.region_info(r).unwrap().pinned);
        m.unpin(r).unwrap();
    }

    #[test]
    fn stale_handle_rejected_by_underlying_shard() {
        let mut m = build(2, 1, 4096);
        let r = m.create_region(RegionKind::HotHeap, 256).unwrap();
        let h = m.alloc(r, 32).unwrap();
        let stale = Handle { generation: 99, ..h };
        assert!(matches!(m.read(stale, &mut [0u8; 32]), Err(TvmError::StaleHandle)));
    }
}
