use tvm_core::backing::VecBackedRegion;
use tvm_core::{Handle, RegionDirectory, RegionKind};

#[test]
fn pack_unpack_handle_roundtrip() {
    let h = Handle { region_id: 3, generation: 7, offset: 1024 };
    assert_eq!(Handle::unpack(h.pack()), h);
}

#[test]
fn alloc_returns_distinct_handles() {
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let region = dir
        .create_region(RegionKind::HotHeap, 1024, VecBackedRegion::new(1024))
        .unwrap();

    let a = dir.alloc(region, 16).unwrap();
    let b = dir.alloc(region, 16).unwrap();
    assert_ne!(a, b);
    assert_eq!(a.region_id, region);
    assert_eq!(b.offset, 16);
}

#[test]
fn read_write_roundtrip() {
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let region = dir
        .create_region(RegionKind::Scratch, 256, VecBackedRegion::new(256))
        .unwrap();
    let h = dir.alloc(region, 8).unwrap();

    dir.write(h, &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    let mut out = [0u8; 8];
    dir.read(h, &mut out).unwrap();
    assert_eq!(out, [1, 2, 3, 4, 5, 6, 7, 8]);
}
