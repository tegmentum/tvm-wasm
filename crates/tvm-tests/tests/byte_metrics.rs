use tvm_core::{RegionDirectory, RegionKind, VecBackedRegion};

#[test]
fn read_increments_bytes_read() {
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r = dir
        .create_region(RegionKind::Scratch, 64, VecBackedRegion::new(64))
        .unwrap();
    let h = dir.alloc(r, 16).unwrap();
    dir.write(h, &[0u8; 16]).unwrap();

    let before = dir.metrics(r).unwrap().snapshot();
    let mut buf = [0u8; 8];
    dir.read(h, &mut buf).unwrap();
    dir.read(h, &mut buf).unwrap();
    let after = dir.metrics(r).unwrap().snapshot();

    assert_eq!(after.bytes_read - before.bytes_read, 16);
}

#[test]
fn write_increments_bytes_written() {
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r = dir
        .create_region(RegionKind::Scratch, 64, VecBackedRegion::new(64))
        .unwrap();
    let h = dir.alloc(r, 16).unwrap();

    let before = dir.metrics(r).unwrap().snapshot();
    dir.write(h, &[1u8; 8]).unwrap();
    dir.write(h, &[2u8; 8]).unwrap();
    let after = dir.metrics(r).unwrap().snapshot();

    assert_eq!(after.bytes_written - before.bytes_written, 16);
}

#[test]
fn cross_region_copy_records_both_sides() {
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let src = dir
        .create_region(RegionKind::HotHeap, 32, VecBackedRegion::new(32))
        .unwrap();
    let dst = dir
        .create_region(RegionKind::Scratch, 32, VecBackedRegion::new(32))
        .unwrap();
    let h = dir.alloc(src, 8).unwrap();
    dir.write(h, b"PAYLOAD!").unwrap();
    let src_before = dir.metrics(src).unwrap().snapshot();
    let dst_before = dir.metrics(dst).unwrap().snapshot();

    dir.cross_region_copy(src, h.offset, dst, 0, 8).unwrap();

    let src_after = dir.metrics(src).unwrap().snapshot();
    let dst_after = dir.metrics(dst).unwrap().snapshot();
    assert_eq!(src_after.bytes_read - src_before.bytes_read, 8);
    assert_eq!(dst_after.bytes_written - dst_before.bytes_written, 8);
}
