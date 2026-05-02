use tvm_core::backing::VecBackedRegion;
use tvm_core::{RegionDirectory, RegionKind, TvmError};

#[test]
fn multiple_regions_have_distinct_ids() {
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r0 = dir
        .create_region(RegionKind::HotHeap, 64, VecBackedRegion::new(64))
        .unwrap();
    let r1 = dir
        .create_region(RegionKind::BlobArena, 64, VecBackedRegion::new(64))
        .unwrap();
    assert_ne!(r0, r1);
}

#[test]
fn alloc_beyond_capacity_fails() {
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r = dir
        .create_region(RegionKind::Scratch, 32, VecBackedRegion::new(32))
        .unwrap();
    dir.alloc(r, 24).unwrap();
    assert!(matches!(dir.alloc(r, 16), Err(TvmError::AllocationFailed)));
}

#[test]
fn out_of_bounds_write_rejected() {
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r = dir
        .create_region(RegionKind::Scratch, 16, VecBackedRegion::new(16))
        .unwrap();
    let h = dir.alloc(r, 8).unwrap();
    let big = vec![0u8; 32];
    assert!(matches!(dir.write(h, &big), Err(TvmError::OutOfBounds)));
}

#[test]
fn destroy_unknown_region_errors() {
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    assert!(matches!(dir.destroy_region(7), Err(TvmError::RegionNotFound(7))));
}
