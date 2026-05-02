use tempfile::tempdir;
use tvm_core::{
    FileBackingStore, RegionDirectory, RegionKind, Residency, TvmError, VecBackedRegion,
};

#[test]
fn region_starts_at_policy_default() {
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    // HotHeap policy → Hot.
    let hot = dir
        .create_region(RegionKind::HotHeap, 64, VecBackedRegion::new(64))
        .unwrap();
    assert_eq!(dir.region_info(hot).unwrap().residency, Residency::Hot);
    // PageStore policy → Warm (LRU-eligible from creation).
    let warm = dir
        .create_region(RegionKind::PageStore, 64, VecBackedRegion::new(64))
        .unwrap();
    assert_eq!(dir.region_info(warm).unwrap().residency, Residency::Warm);
}

#[test]
fn spill_then_load_preserves_bytes() {
    let tmp = tempdir().unwrap();
    let mut backing = FileBackingStore::new(tmp.path()).unwrap();

    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r = dir
        .create_region(RegionKind::ObjectArena, 64, VecBackedRegion::new(64))
        .unwrap();
    let h = dir.alloc(r, 16).unwrap();
    dir.write(h, b"abcdefghijklmnop").unwrap();

    dir.spill_region(r, &mut backing).unwrap();
    assert_eq!(dir.region_info(r).unwrap().residency, Residency::Cold);

    let mut buf = [0u8; 16];
    assert!(matches!(dir.read(h, &mut buf), Err(TvmError::NotResident)));

    dir.load_region(r, &mut backing).unwrap();
    assert_eq!(dir.region_info(r).unwrap().residency, Residency::Hot);

    dir.read(h, &mut buf).unwrap();
    assert_eq!(&buf, b"abcdefghijklmnop");
}

#[test]
fn fault_counter_increments_on_load() {
    let tmp = tempdir().unwrap();
    let mut backing = FileBackingStore::new(tmp.path()).unwrap();
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r = dir
        // Scratch policy is non-spillable; use ObjectArena for this test.
        .create_region(RegionKind::ObjectArena, 32, VecBackedRegion::new(32))
        .unwrap();
    dir.alloc(r, 8).unwrap();

    let before = dir.metrics(r).unwrap().snapshot();
    dir.spill_region(r, &mut backing).unwrap();
    dir.load_region(r, &mut backing).unwrap();
    let after = dir.metrics(r).unwrap().snapshot();

    assert_eq!(after.faults, before.faults + 1);
    assert_eq!(after.demotions, before.demotions + 1);
    assert_eq!(after.promotions, before.promotions + 1);
}

#[test]
fn double_spill_is_idempotent() {
    let tmp = tempdir().unwrap();
    let mut backing = FileBackingStore::new(tmp.path()).unwrap();
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r = dir
        .create_region(RegionKind::PageStore, 16, VecBackedRegion::new(16))
        .unwrap();
    dir.spill_region(r, &mut backing).unwrap();
    // second spill on already-cold region should be a no-op
    dir.spill_region(r, &mut backing).unwrap();
    assert_eq!(dir.region_info(r).unwrap().residency, Residency::Cold);
}
