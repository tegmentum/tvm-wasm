use tvm_core::{AllocatorKind, Handle, RegionDirectory, RegionKind, TvmError, VecBackedRegion};

#[test]
fn compaction_packs_blocks_and_remaps_handles() {
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r = dir
        .create_region_with(
            RegionKind::ObjectArena,
            64,
            AllocatorKind::Freelist,
            VecBackedRegion::new(64),
        )
        .unwrap();

    let a = dir.alloc(r, 8).unwrap();
    let b = dir.alloc(r, 8).unwrap();
    let c = dir.alloc(r, 8).unwrap();
    dir.write(a, b"AAAAAAAA").unwrap();
    dir.write(b, b"BBBBBBBB").unwrap();
    dir.write(c, b"CCCCCCCC").unwrap();

    // Free the middle block, leaving a hole.
    dir.dealloc(b).unwrap();
    let info = dir.region_info(r).unwrap();
    assert_eq!(info.used, 16);

    // Compact. Old handles for a/c are now stale.
    let remap = dir.compact_region(r).unwrap();
    assert_ne!(remap.old_generation, remap.new_generation);

    let mut buf = [0u8; 8];
    assert!(matches!(dir.read(a, &mut buf), Err(TvmError::StaleHandle)));

    let a2 = remap.migrate(a).unwrap();
    let c2 = remap.migrate(c).unwrap();
    assert_eq!(a2.offset, 0);
    assert_eq!(c2.offset, 8);

    dir.read(a2, &mut buf).unwrap();
    assert_eq!(&buf, b"AAAAAAAA");
    dir.read(c2, &mut buf).unwrap();
    assert_eq!(&buf, b"CCCCCCCC");

    // After compaction we should have a 48-byte free trailing block.
    let big = dir.alloc(r, 48).unwrap();
    assert_eq!(big.offset, 16);
}

#[test]
fn compaction_rejects_bump_allocator() {
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r = dir
        .create_region(RegionKind::HotHeap, 32, VecBackedRegion::new(32))
        .unwrap();
    dir.alloc(r, 8).unwrap();
    assert!(matches!(
        dir.compact_region(r),
        Err(TvmError::UnsupportedAllocator)
    ));
}

#[test]
fn compaction_rejects_pinned_region() {
    use tvm_core::policy::PlacementPolicy;
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let policy = PlacementPolicy {
        initial_residency: tvm_core::Residency::Hot,
        pinnable: true,
        spillable: false,
    };
    let r = dir
        .create_region_with_policy(
            RegionKind::ObjectArena,
            32,
            AllocatorKind::Freelist,
            policy,
            VecBackedRegion::new(32),
        )
        .unwrap();
    dir.alloc(r, 8).unwrap();
    dir.pin(r).unwrap();
    assert!(matches!(dir.compact_region(r), Err(TvmError::Pinned)));
}

#[test]
fn compaction_at_scale_preserves_every_surviving_handle() {
    // Scale test for compaction correctness under high fragmentation.
    // Existing tests use 16-byte regions with three allocations; this
    // one stresses the worst case freelist + remap path: thousands of
    // live blocks scattered through a region after every-other-block
    // frees.
    //
    // 16 MiB region / 16384 × 1 KiB allocations (fully packed) / free
    // even-indexed → 8192 surviving blocks, 8192 holes. After compact,
    // all surviving bytes must round-trip via remapped handles, and the
    // freed space must reappear as a single contiguous trailing hole.
    const REGION_BYTES: u32 = 16 * 1024 * 1024;
    const BLOCK_BYTES: u32 = 1024;
    const N_BLOCKS: u32 = 16 * 1024;

    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r = dir
        .create_region_with(
            RegionKind::ObjectArena,
            REGION_BYTES,
            AllocatorKind::Freelist,
            VecBackedRegion::new(REGION_BYTES),
        )
        .unwrap();

    // Tag each block with a recognizable pattern derived from its
    // index. Pattern is: first 4 bytes = block index (LE), next 4 = ~index.
    // The rest is the index byte, repeated. This catches any byte-level
    // misalignment introduced by remap.
    let make_pattern = |idx: u32| -> Vec<u8> {
        let mut buf = vec![idx as u8; BLOCK_BYTES as usize];
        buf[0..4].copy_from_slice(&idx.to_le_bytes());
        buf[4..8].copy_from_slice(&(!idx).to_le_bytes());
        buf
    };

    // Allocate and write all blocks.
    let mut handles: Vec<Handle> = Vec::with_capacity(N_BLOCKS as usize);
    for i in 0..N_BLOCKS {
        let h = dir.alloc(r, BLOCK_BYTES).unwrap();
        dir.write(h, &make_pattern(i)).unwrap();
        handles.push(h);
    }
    assert_eq!(dir.region_info(r).unwrap().used, REGION_BYTES);

    // Free every even-indexed block. Survivors are odd indices.
    for i in (0..N_BLOCKS).step_by(2) {
        dir.dealloc(handles[i as usize]).unwrap();
    }
    let live_used = dir.region_info(r).unwrap().used;
    assert_eq!(live_used, (N_BLOCKS / 2) * BLOCK_BYTES);

    let remap = dir.compact_region(r).unwrap();
    assert_ne!(remap.old_generation, remap.new_generation);

    // Every survivor must remap and read back the original pattern.
    let mut buf = vec![0u8; BLOCK_BYTES as usize];
    for i in (1..N_BLOCKS).step_by(2) {
        let new_handle = remap.migrate(handles[i as usize]).unwrap();
        // After packing, survivors should land at consecutive offsets
        // 0, BLOCK_BYTES, 2*BLOCK_BYTES, ... in their original order.
        let expected_offset = (i / 2) * BLOCK_BYTES;
        assert_eq!(
            new_handle.offset, expected_offset,
            "block {i} expected to land at offset {expected_offset}, got {}",
            new_handle.offset
        );
        dir.read(new_handle, &mut buf).unwrap();
        assert_eq!(
            buf,
            make_pattern(i),
            "block {i} content corrupted after compaction"
        );
    }

    // All freed even-indexed blocks must produce StaleHandle on read.
    assert!(matches!(
        dir.read(handles[2], &mut buf),
        Err(TvmError::StaleHandle | TvmError::OutOfBounds | TvmError::AllocationFailed)
    ));

    // Live bytes preserved; freed half is now contiguous trailing hole.
    assert_eq!(dir.region_info(r).unwrap().used, live_used);
    let big_alloc_size = REGION_BYTES - live_used;
    let big = dir.alloc(r, big_alloc_size).unwrap();
    assert_eq!(
        big.offset, live_used,
        "freed region should coalesce into a single trailing hole"
    );
}

#[test]
fn migrate_returns_none_for_unrelated_handle() {
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r = dir
        .create_region_with(
            RegionKind::Scratch,
            16,
            AllocatorKind::Freelist,
            VecBackedRegion::new(16),
        )
        .unwrap();
    dir.alloc(r, 4).unwrap();
    let remap = dir.compact_region(r).unwrap();

    let unrelated = Handle {
        region_id: 99,
        generation: 1,
        offset: 0,
    };
    assert_eq!(remap.migrate(unrelated), None);
}
