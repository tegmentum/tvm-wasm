use tvm_core::{AllocatorKind, ConcurrentDirectory, RegionKind, TvmError, VecBackedRegion};

#[test]
fn concurrent_compact_packs_blocks() {
    let dir: ConcurrentDirectory<VecBackedRegion> = ConcurrentDirectory::new();
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
    dir.dealloc(b).unwrap();

    let remap = dir.compact_region(r).unwrap();
    assert_ne!(remap.old_generation, remap.new_generation);
    assert_eq!(remap.mapping.len(), 2);

    let a2 = remap.migrate(a).unwrap();
    let c2 = remap.migrate(c).unwrap();

    let mut buf = [0u8; 8];
    dir.read(a2, &mut buf).unwrap();
    assert_eq!(&buf, b"AAAAAAAA");
    dir.read(c2, &mut buf).unwrap();
    assert_eq!(&buf, b"CCCCCCCC");

    // Old-gen handle is stale.
    assert!(matches!(dir.read(a, &mut buf), Err(TvmError::StaleHandle)));
}

#[test]
fn concurrent_compact_rejects_bump_allocator() {
    let dir: ConcurrentDirectory<VecBackedRegion> = ConcurrentDirectory::new();
    let r = dir
        .create_region(RegionKind::HotHeap, 32, VecBackedRegion::new(32))
        .unwrap();
    dir.alloc(r, 8).unwrap();
    assert!(matches!(
        dir.compact_region(r),
        Err(TvmError::UnsupportedAllocator)
    ));
}
