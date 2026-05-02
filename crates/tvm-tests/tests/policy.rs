use tvm_core::{RegionDirectory, RegionKind, Residency, VecBackedRegion};

#[test]
fn page_store_starts_warm_per_policy() {
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r = dir
        .create_region(RegionKind::PageStore, 32, VecBackedRegion::new(32))
        .unwrap();
    assert_eq!(dir.region_info(r).unwrap().residency, Residency::Warm);
}

#[test]
fn hot_heap_is_pinnable_not_spillable() {
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r = dir
        .create_region(RegionKind::HotHeap, 32, VecBackedRegion::new(32))
        .unwrap();
    let info = dir.region_info(r).unwrap();
    assert!(info.pinnable);
    assert!(!info.spillable);
}

#[test]
fn object_arena_is_spillable_not_pinnable() {
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r = dir
        .create_region(RegionKind::ObjectArena, 32, VecBackedRegion::new(32))
        .unwrap();
    let info = dir.region_info(r).unwrap();
    assert!(!info.pinnable);
    assert!(info.spillable);
}

#[test]
fn warm_on_create_appears_in_lru() {
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r = dir
        .create_region(RegionKind::PageStore, 32, VecBackedRegion::new(32))
        .unwrap();
    // PageStore starts Warm, so should be the back of the LRU (only entry).
    assert_eq!(dir.warm_lru_back(), Some(r));
}
