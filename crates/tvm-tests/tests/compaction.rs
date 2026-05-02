use tvm_core::{
    AllocatorKind, Handle, RegionDirectory, RegionKind, TvmError, VecBackedRegion,
};

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

    let unrelated = Handle { region_id: 99, generation: 1, offset: 0 };
    assert_eq!(remap.migrate(unrelated), None);
}
