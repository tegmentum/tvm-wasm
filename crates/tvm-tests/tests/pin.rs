use tempfile::tempdir;
use tvm_core::policy::PlacementPolicy;
use tvm_core::{
    AllocatorKind, FileBackingStore, RegionDirectory, RegionKind, Residency, TvmError,
    VecBackedRegion,
};

fn pinnable_and_spillable() -> PlacementPolicy {
    PlacementPolicy {
        initial_residency: Residency::Hot,
        pinnable: true,
        spillable: true,
    }
}

#[test]
fn pinned_region_cannot_be_spilled() {
    let tmp = tempdir().unwrap();
    let mut backing = FileBackingStore::new(tmp.path()).unwrap();
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r = dir
        .create_region_with_policy(
            RegionKind::HotHeap,
            32,
            AllocatorKind::Bump,
            pinnable_and_spillable(),
            VecBackedRegion::new(32),
        )
        .unwrap();
    dir.pin(r).unwrap();
    assert!(matches!(dir.spill_region(r, &mut backing), Err(TvmError::Pinned)));
}

#[test]
fn unpinned_region_can_be_spilled() {
    let tmp = tempdir().unwrap();
    let mut backing = FileBackingStore::new(tmp.path()).unwrap();
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r = dir
        .create_region_with_policy(
            RegionKind::HotHeap,
            32,
            AllocatorKind::Bump,
            pinnable_and_spillable(),
            VecBackedRegion::new(32),
        )
        .unwrap();
    dir.pin(r).unwrap();
    dir.unpin(r).unwrap();
    dir.spill_region(r, &mut backing).unwrap();
}

#[test]
fn pin_rejects_non_pinnable_kind() {
    // ObjectArena policy: pinnable = false.
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r = dir
        .create_region(RegionKind::ObjectArena, 16, VecBackedRegion::new(16))
        .unwrap();
    assert!(matches!(dir.pin(r), Err(TvmError::PolicyViolation)));
}

#[test]
fn spill_rejects_non_spillable_kind() {
    let tmp = tempdir().unwrap();
    let mut backing = FileBackingStore::new(tmp.path()).unwrap();
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    // HotHeap policy: spillable = false.
    let r = dir
        .create_region(RegionKind::HotHeap, 16, VecBackedRegion::new(16))
        .unwrap();
    assert!(matches!(
        dir.spill_region(r, &mut backing),
        Err(TvmError::PolicyViolation)
    ));
}
