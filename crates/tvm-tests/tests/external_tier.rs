use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use tvm_core::external::ExternalLoader;
use tvm_core::{
    Handle, RegionDirectory, RegionKind, Residency, TvmError, VecBackedRegion,
};

#[test]
fn mark_external_drops_memory() {
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r = dir
        .create_region(RegionKind::ObjectArena, 32, VecBackedRegion::new(32))
        .unwrap();
    let h = dir.alloc(r, 8).unwrap();
    dir.write(h, b"abcdefgh").unwrap();
    dir.mark_external(r).unwrap();
    assert_eq!(dir.region_info(r).unwrap().residency, Residency::External);
    let mut buf = [0u8; 8];
    assert!(matches!(dir.read(h, &mut buf), Err(TvmError::NotResident)));
}

#[test]
fn external_loader_invoked_on_load() {
    let calls = Arc::new(AtomicU32::new(0));
    let calls_for_loader = Arc::clone(&calls);
    let loader: ExternalLoader = Box::new(move |_region_id, _gen| {
        calls_for_loader.fetch_add(1, Ordering::Relaxed);
        let mut bytes = vec![0u8; 32];
        bytes[0..4].copy_from_slice(b"REMT");
        Ok(bytes)
    });

    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r = dir
        .create_region(RegionKind::ObjectArena, 32, VecBackedRegion::new(32))
        .unwrap();
    dir.mark_external(r).unwrap();
    dir.load_external_region(r, &loader).unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(dir.region_info(r).unwrap().residency, Residency::Hot);

    let h = Handle { region_id: r, generation: 1, offset: 0 };
    let mut buf = [0u8; 4];
    dir.read(h, &mut buf).unwrap();
    assert_eq!(&buf, b"REMT");
}

#[test]
fn mark_external_pinned_region_errors() {
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r = dir
        .create_region(RegionKind::HotHeap, 16, VecBackedRegion::new(16))
        .unwrap();
    dir.pin(r).unwrap();
    assert!(matches!(dir.mark_external(r), Err(TvmError::Pinned)));
}
