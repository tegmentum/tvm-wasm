use tempfile::tempdir;
use tvm_core::{RegionDirectory, RegionKind, TvmError, VecBackedRegion};

#[test]
fn snapshot_then_restore_round_trip() {
    let tmp = tempdir().unwrap();
    let path = tmp.path().join("region.bin");

    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r = dir
        .create_region(RegionKind::ObjectArena, 32, VecBackedRegion::new(32))
        .unwrap();
    let h = dir.alloc(r, 8).unwrap();
    dir.write(h, b"abcdefgh").unwrap();
    dir.snapshot_region(r, &path).unwrap();

    // Restore into a fresh region.
    let mut dir2: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r2 = dir2
        .create_region(RegionKind::ObjectArena, 32, VecBackedRegion::new(32))
        .unwrap();
    dir2.restore_region(r2, &path).unwrap();

    let mut buf = [0u8; 8];
    let h2 = tvm_core::Handle {
        region_id: r2,
        generation: 1,
        offset: 0,
    };
    dir2.read(h2, &mut buf).unwrap();
    assert_eq!(&buf, b"abcdefgh");
}

#[test]
fn restore_rejects_oversized_file() {
    let tmp = tempdir().unwrap();
    let path = tmp.path().join("big.bin");
    std::fs::write(&path, vec![0u8; 64]).unwrap();

    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r = dir
        .create_region(RegionKind::Scratch, 32, VecBackedRegion::new(32))
        .unwrap();
    assert!(matches!(
        dir.restore_region(r, &path),
        Err(TvmError::OutOfBounds)
    ));
}
