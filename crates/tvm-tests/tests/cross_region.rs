use tvm_core::{AllocatorKind, RegionDirectory, RegionKind, TvmError, VecBackedRegion};

#[test]
fn copy_between_regions() {
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let src = dir
        .create_region(RegionKind::HotHeap, 64, VecBackedRegion::new(64))
        .unwrap();
    let dst = dir
        .create_region(RegionKind::Scratch, 64, VecBackedRegion::new(64))
        .unwrap();
    let h_src = dir.alloc(src, 8).unwrap();
    dir.write(h_src, b"abcdefgh").unwrap();

    dir.cross_region_copy(src, h_src.offset, dst, 0, 8).unwrap();

    let h_dst = tvm_core::Handle {
        region_id: dst,
        generation: 1,
        offset: 0,
    };
    let mut buf = [0u8; 8];
    dir.read(h_dst, &mut buf).unwrap();
    assert_eq!(&buf, b"abcdefgh");
}

#[test]
fn copy_within_same_region() {
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r = dir
        .create_region_with(
            RegionKind::Scratch,
            64,
            AllocatorKind::Freelist,
            VecBackedRegion::new(64),
        )
        .unwrap();
    let a = dir.alloc(r, 8).unwrap();
    let b = dir.alloc(r, 8).unwrap();
    dir.write(a, b"01234567").unwrap();

    dir.cross_region_copy(r, a.offset, r, b.offset, 8).unwrap();
    let mut buf = [0u8; 8];
    dir.read(b, &mut buf).unwrap();
    assert_eq!(&buf, b"01234567");
}

#[test]
fn read_into_validates_source_handle() {
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let src = dir
        .create_region(RegionKind::HotHeap, 32, VecBackedRegion::new(32))
        .unwrap();
    let dst = dir
        .create_region(RegionKind::Scratch, 32, VecBackedRegion::new(32))
        .unwrap();
    let h = dir.alloc(src, 4).unwrap();
    dir.bump_generation(src).unwrap();

    assert!(matches!(
        dir.read_into(h, dst, 0, 4),
        Err(TvmError::StaleHandle)
    ));
}

#[test]
fn write_from_validates_destination_handle() {
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let src = dir
        .create_region(RegionKind::HotHeap, 32, VecBackedRegion::new(32))
        .unwrap();
    let dst = dir
        .create_region(RegionKind::Scratch, 32, VecBackedRegion::new(32))
        .unwrap();
    let h = dir.alloc(dst, 4).unwrap();
    dir.bump_generation(dst).unwrap();

    assert!(matches!(
        dir.write_from(src, 0, h, 4),
        Err(TvmError::StaleHandle)
    ));
}

#[test]
fn out_of_bounds_copy_rejected() {
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let src = dir
        .create_region(RegionKind::HotHeap, 32, VecBackedRegion::new(32))
        .unwrap();
    let dst = dir
        .create_region(RegionKind::Scratch, 32, VecBackedRegion::new(32))
        .unwrap();
    assert!(matches!(
        dir.cross_region_copy(src, 0, dst, 0, 64),
        Err(TvmError::OutOfBounds)
    ));
}
