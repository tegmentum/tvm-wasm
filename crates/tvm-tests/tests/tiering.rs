use tempfile::tempdir;
use tvm_core::policy::PlacementPolicy;
use tvm_core::{
    AllocatorKind, FileBackingStore, RegionDirectory, RegionKind, Residency, TvmError,
    VecBackedRegion,
};

fn dir_with_backing() -> (
    RegionDirectory<VecBackedRegion>,
    FileBackingStore,
    tempfile::TempDir,
) {
    let tmp = tempdir().unwrap();
    let backing = FileBackingStore::new(tmp.path()).unwrap();
    (RegionDirectory::new(), backing, tmp)
}

fn flexible_policy() -> PlacementPolicy {
    PlacementPolicy {
        initial_residency: Residency::Hot,
        pinnable: true,
        spillable: true,
    }
}

fn create_flexible(dir: &mut RegionDirectory<VecBackedRegion>, capacity: u32) -> u16 {
    dir.create_region_with_policy(
        RegionKind::ObjectArena,
        capacity,
        AllocatorKind::Bump,
        flexible_policy(),
        VecBackedRegion::new(capacity),
    )
    .unwrap()
}

#[test]
fn demote_hot_to_warm_then_cold() {
    let (mut dir, mut backing, _tmp) = dir_with_backing();
    let r = dir
        .create_region(RegionKind::ObjectArena, 32, VecBackedRegion::new(32))
        .unwrap();

    dir.demote_region(r, &mut backing).unwrap();
    assert_eq!(dir.region_info(r).unwrap().residency, Residency::Warm);

    dir.demote_region(r, &mut backing).unwrap();
    assert_eq!(dir.region_info(r).unwrap().residency, Residency::Cold);
}

#[test]
fn promote_warm_back_to_hot() {
    let (mut dir, mut backing, _tmp) = dir_with_backing();
    let r = dir
        .create_region(RegionKind::ObjectArena, 32, VecBackedRegion::new(32))
        .unwrap();
    dir.demote_region(r, &mut backing).unwrap(); // → Warm
    dir.promote_region(r, &mut backing).unwrap();
    assert_eq!(dir.region_info(r).unwrap().residency, Residency::Hot);
}

#[test]
fn promote_cold_loads_from_backing() {
    let (mut dir, mut backing, _tmp) = dir_with_backing();
    let r = create_flexible(&mut dir, 32);
    let h = dir.alloc(r, 8).unwrap();
    dir.write(h, b"VALUE-XX").unwrap();
    dir.spill_region(r, &mut backing).unwrap();
    assert_eq!(dir.region_info(r).unwrap().residency, Residency::Cold);

    dir.promote_region(r, &mut backing).unwrap();
    assert_eq!(dir.region_info(r).unwrap().residency, Residency::Hot);

    let mut buf = [0u8; 8];
    dir.read(h, &mut buf).unwrap();
    assert_eq!(&buf, b"VALUE-XX");
}

#[test]
fn read_or_fault_auto_loads_cold_region() {
    let (mut dir, mut backing, _tmp) = dir_with_backing();
    let r = dir
        .create_region(RegionKind::ObjectArena, 16, VecBackedRegion::new(16))
        .unwrap();
    let h = dir.alloc(r, 8).unwrap();
    dir.write(h, b"AUTOLOAD").unwrap();
    dir.spill_region(r, &mut backing).unwrap();

    let before = dir.metrics(r).unwrap().snapshot();
    let mut buf = [0u8; 8];
    dir.read_or_fault(h, &mut buf, &mut backing).unwrap();
    assert_eq!(&buf, b"AUTOLOAD");
    let after = dir.metrics(r).unwrap().snapshot();
    assert_eq!(after.faults, before.faults + 1);
    assert_eq!(dir.region_info(r).unwrap().residency, Residency::Hot);
}

#[test]
fn write_or_fault_auto_loads_cold_region() {
    let (mut dir, mut backing, _tmp) = dir_with_backing();
    let r = create_flexible(&mut dir, 16);
    let h = dir.alloc(r, 4).unwrap();
    dir.spill_region(r, &mut backing).unwrap();

    dir.write_or_fault(h, b"GOOD", &mut backing).unwrap();
    assert_eq!(dir.region_info(r).unwrap().residency, Residency::Hot);
    let mut buf = [0u8; 4];
    dir.read(h, &mut buf).unwrap();
    assert_eq!(&buf, b"GOOD");
}

#[test]
fn evict_warm_region_picks_oldest() {
    let (mut dir, mut backing, _tmp) = dir_with_backing();
    let r0 = dir
        .create_region(RegionKind::ObjectArena, 16, VecBackedRegion::new(16))
        .unwrap();
    let r1 = dir
        .create_region(RegionKind::ObjectArena, 16, VecBackedRegion::new(16))
        .unwrap();
    let r2 = dir
        .create_region(RegionKind::ObjectArena, 16, VecBackedRegion::new(16))
        .unwrap();

    dir.demote_region(r0, &mut backing).unwrap(); // oldest in LRU back
    dir.demote_region(r1, &mut backing).unwrap();
    dir.demote_region(r2, &mut backing).unwrap(); // most recent at front

    let evicted = dir.evict_warm_region(&mut backing).unwrap();
    assert_eq!(evicted, Some(r0));
    assert_eq!(dir.region_info(r0).unwrap().residency, Residency::Cold);
    assert_eq!(dir.region_info(r1).unwrap().residency, Residency::Warm);
    assert_eq!(dir.region_info(r2).unwrap().residency, Residency::Warm);
}

#[test]
fn evict_warm_region_skips_pinned() {
    let (mut dir, mut backing, _tmp) = dir_with_backing();
    let r0 = create_flexible(&mut dir, 16);
    let r1 = create_flexible(&mut dir, 16);
    dir.demote_region(r0, &mut backing).unwrap();
    dir.demote_region(r1, &mut backing).unwrap();
    dir.pin(r0).unwrap();

    let evicted = dir.evict_warm_region(&mut backing).unwrap();
    assert_eq!(evicted, Some(r1));
}

#[test]
fn evict_warm_region_returns_none_when_empty() {
    let (mut dir, mut backing, _tmp) = dir_with_backing();
    assert_eq!(dir.evict_warm_region(&mut backing).unwrap(), None);
}

#[test]
fn demote_pinned_region_errors() {
    let (mut dir, mut backing, _tmp) = dir_with_backing();
    let r = dir
        .create_region(RegionKind::HotHeap, 16, VecBackedRegion::new(16))
        .unwrap();
    dir.pin(r).unwrap();
    assert!(matches!(
        dir.demote_region(r, &mut backing),
        Err(TvmError::Pinned)
    ));
}

#[test]
fn promote_hot_region_is_noop() {
    let (mut dir, mut backing, _tmp) = dir_with_backing();
    let r = dir
        .create_region(RegionKind::HotHeap, 16, VecBackedRegion::new(16))
        .unwrap();
    let before = dir.metrics(r).unwrap().snapshot();
    dir.promote_region(r, &mut backing).unwrap();
    let after = dir.metrics(r).unwrap().snapshot();
    assert_eq!(after.promotions, before.promotions);
}
