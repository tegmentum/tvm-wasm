use tvm_core::{debug, Handle, HandleStatus, RegionDirectory, RegionKind, VecBackedRegion};

#[test]
fn dump_layout_lists_all_regions() {
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    dir.create_region(RegionKind::HotHeap, 64, VecBackedRegion::new(64))
        .unwrap();
    dir.create_region(RegionKind::BlobArena, 32, VecBackedRegion::new(32))
        .unwrap();

    let dump = debug::dump_region_layout(&dir);
    assert!(dump.contains("HotHeap"));
    assert!(dump.contains("BlobArena"));
    assert!(dump.contains("regions: 2"));
}

#[test]
fn validate_handle_classifies_each_state() {
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r = dir
        .create_region(RegionKind::Scratch, 32, VecBackedRegion::new(32))
        .unwrap();
    let h = dir.alloc(r, 8).unwrap();

    assert_eq!(debug::validate_handle(&dir, h), HandleStatus::Valid);

    let unknown = Handle {
        region_id: 99,
        generation: 1,
        offset: 0,
    };
    assert_eq!(
        debug::validate_handle(&dir, unknown),
        HandleStatus::UnknownRegion
    );

    let stale = Handle {
        region_id: r,
        generation: 99,
        offset: 0,
    };
    assert_eq!(
        debug::validate_handle(&dir, stale),
        HandleStatus::StaleGeneration
    );

    let oob = Handle {
        region_id: r,
        generation: 1,
        offset: 999,
    };
    assert_eq!(debug::validate_handle(&dir, oob), HandleStatus::OutOfBounds);
}

#[test]
fn validate_handles_batch_produces_pairs() {
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r = dir
        .create_region(RegionKind::HotHeap, 32, VecBackedRegion::new(32))
        .unwrap();
    let h1 = dir.alloc(r, 4).unwrap();
    let h2 = Handle {
        region_id: r,
        generation: 99,
        offset: 0,
    };

    let result = debug::validate_handles(&dir, &[h1, h2]);
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].1, HandleStatus::Valid);
    assert_eq!(result[1].1, HandleStatus::StaleGeneration);
}

#[test]
fn fault_counts_starts_at_zero() {
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r = dir
        .create_region(RegionKind::PageStore, 16, VecBackedRegion::new(16))
        .unwrap();
    let counts = debug::fault_counts(&dir);
    assert_eq!(counts, vec![(r, 0)]);
}
