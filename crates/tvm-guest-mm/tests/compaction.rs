//! Compaction end-to-end: create a freelist-backed region, allocate
//! several blocks, free a middle block, compact, verify live data is
//! preserved and packed contiguously and that old handles are migrated
//! correctly via the returned `HandleRemap`.

use std::sync::Mutex;
use tvm_guest_mm::{
    AllocatorKind, Dispatch, GuestTvm, Pool, RegionKind, Result, TvmError,
};

// Stub backing for the test — one Vec<u8> per pool. The intra-pool
// copy stub emulates `memory.copy K K` semantically (with overlap
// support) so the directory's compactor can be exercised host-side
// without a wasm runtime.
static STUB_POOLS: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

fn stub_read(pool: u32, off: u32, dst: &mut [u8]) -> Result<()> {
    let pools = STUB_POOLS.lock().unwrap();
    let p = &pools[pool as usize];
    let s = off as usize;
    let n = dst.len();
    if s + n > p.len() { return Err(TvmError::OutOfBounds); }
    dst.copy_from_slice(&p[s..s + n]);
    Ok(())
}

fn stub_write(pool: u32, off: u32, src: &[u8]) -> Result<()> {
    let mut pools = STUB_POOLS.lock().unwrap();
    let p = &mut pools[pool as usize];
    let s = off as usize;
    let n = src.len();
    if s + n > p.len() { return Err(TvmError::OutOfBounds); }
    p[s..s + n].copy_from_slice(src);
    Ok(())
}

fn stub_intra_pool_copy(pool: u32, dst_off: u32, src_off: u32, len: u32) -> Result<()> {
    let mut pools = STUB_POOLS.lock().unwrap();
    let p = &mut pools[pool as usize];
    let s = src_off as usize;
    let d = dst_off as usize;
    let n = len as usize;
    if s + n > p.len() || d + n > p.len() { return Err(TvmError::OutOfBounds); }
    p.copy_within(s..s + n, d);
    Ok(())
}

fn build(n_pools: usize, capacity: u32) -> GuestTvm {
    let mut pools = STUB_POOLS.lock().unwrap();
    pools.clear();
    for _ in 0..n_pools {
        pools.push(vec![0u8; capacity as usize]);
    }
    drop(pools);
    let descs: Vec<Pool> = (0..n_pools as u32)
        .map(|i| Pool { memory_index: i, used: 0, capacity })
        .collect();
    let mut g = GuestTvm::new(descs, Dispatch {
        read_bytes: stub_read,
        write_bytes: stub_write,
        intra_pool_copy: stub_intra_pool_copy,
    });
    g.default_allocator = AllocatorKind::Freelist;
    g
}

#[test]
fn compaction_packs_live_blocks_and_migrates_handles() {
    let mut g = build(2, 4096);
    let r = g.directory_mut()
        .create_region(RegionKind::HotHeap, 1024, AllocatorKind::Freelist)
        .unwrap();

    // Allocate four blocks of distinct sizes with distinct payloads.
    let h_a = g.directory_mut().alloc(r, 64).unwrap();
    let h_b = g.directory_mut().alloc(r, 128).unwrap();
    let h_c = g.directory_mut().alloc(r, 32).unwrap();
    let h_d = g.directory_mut().alloc(r, 96).unwrap();

    use tvm_core::TvmFacade;
    g.write(h_a, &[0xaa; 64]).unwrap();
    g.write(h_b, &[0xbb; 128]).unwrap();
    g.write(h_c, &[0xcc; 32]).unwrap();
    g.write(h_d, &[0xdd; 96]).unwrap();

    // Free B → leaves a 128-byte hole between A and C.
    g.directory_mut().dealloc(h_b).unwrap();

    // Compact. Live blocks A (64), C (32), D (96) should pack to
    // offsets 0, 64, 96.
    let remap = g.compact_region(r).expect("compaction must succeed");
    assert_eq!(remap.region_id, r);
    assert_eq!(remap.old_generation, 1);
    assert_eq!(remap.new_generation, 2);
    assert_eq!(remap.mapping.len(), 3);
    assert_eq!(remap.mapping.get(&h_a.offset).copied(), Some(0));
    assert_eq!(remap.mapping.get(&h_c.offset).copied(), Some(64));
    assert_eq!(remap.mapping.get(&h_d.offset).copied(), Some(96));

    // Old handles must fail validation now.
    let mut buf = [0u8; 64];
    assert!(matches!(g.read(h_a, &mut buf), Err(TvmError::StaleHandle)));

    // Migrated handles must read the correct data.
    let h_a2 = remap.migrate(h_a).expect("A migrate");
    let h_c2 = remap.migrate(h_c).expect("C migrate");
    let h_d2 = remap.migrate(h_d).expect("D migrate");

    let mut a_buf = [0u8; 64];
    g.read(h_a2, &mut a_buf).unwrap();
    assert!(a_buf.iter().all(|&b| b == 0xaa), "A bytes preserved");

    let mut c_buf = [0u8; 32];
    g.read(h_c2, &mut c_buf).unwrap();
    assert!(c_buf.iter().all(|&b| b == 0xcc), "C bytes preserved");

    let mut d_buf = [0u8; 96];
    g.read(h_d2, &mut d_buf).unwrap();
    assert!(d_buf.iter().all(|&b| b == 0xdd), "D bytes preserved");

    // After compaction, used = 64 + 32 + 96 = 192.
    let info = g.region_info(r).unwrap();
    assert_eq!(info.used, 192);
    assert_eq!(info.generation, 2);

    // Subsequent allocations come from the freed tail, not the hole.
    let h_e = g.directory_mut().alloc(r, 200).unwrap();
    assert_eq!(h_e.offset, 192, "next alloc starts at compacted cursor");
}

#[test]
fn compaction_rejects_pinned_region() {
    let mut g = build(1, 4096);
    let r = g.directory_mut()
        .create_region(RegionKind::HotHeap, 512, AllocatorKind::Freelist)
        .unwrap();
    g.directory_mut().alloc(r, 64).unwrap();
    g.directory_mut().pin(r).unwrap();
    assert!(matches!(g.compact_region(r), Err(TvmError::Pinned)));
}

#[test]
fn compaction_rejects_bump_allocator() {
    let mut g = build(1, 4096);
    let r = g.directory_mut()
        .create_region(RegionKind::HotHeap, 512, AllocatorKind::Bump)
        .unwrap();
    g.directory_mut().alloc(r, 64).unwrap();
    assert!(matches!(g.compact_region(r), Err(TvmError::UnsupportedAllocator)));
}
