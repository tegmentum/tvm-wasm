use std::thread;

use tvm_core::{RegionKind, SharedDirectory, VecBackedRegion};

#[test]
fn handles_can_be_used_across_threads() {
    let dir: SharedDirectory<VecBackedRegion> = SharedDirectory::new();
    let region = dir
        .create_region(RegionKind::ObjectArena, 256, VecBackedRegion::new(256))
        .unwrap();
    let handle = dir.alloc(region, 16).unwrap();

    let writer = {
        let dir = dir.clone();
        thread::spawn(move || {
            dir.write(handle, b"thread-written!!").unwrap();
        })
    };
    writer.join().unwrap();

    let mut buf = [0u8; 16];
    dir.read(handle, &mut buf).unwrap();
    assert_eq!(&buf, b"thread-written!!");
}

#[test]
fn parallel_allocs_in_same_region_succeed() {
    let dir: SharedDirectory<VecBackedRegion> = SharedDirectory::new();
    let region = dir
        .create_region(RegionKind::HotHeap, 1024, VecBackedRegion::new(1024))
        .unwrap();

    let mut threads = Vec::new();
    for _ in 0..8 {
        let dir = dir.clone();
        threads.push(thread::spawn(move || {
            for _ in 0..4 {
                let _ = dir.alloc(region, 16).unwrap();
            }
        }));
    }
    for t in threads {
        t.join().unwrap();
    }

    let info = dir.region_info(region).unwrap();
    assert_eq!(info.used, 8 * 4 * 16);
}

#[test]
fn list_regions_under_lock() {
    let dir: SharedDirectory<VecBackedRegion> = SharedDirectory::new();
    dir.create_region(RegionKind::HotHeap, 32, VecBackedRegion::new(32))
        .unwrap();
    dir.create_region(RegionKind::Scratch, 32, VecBackedRegion::new(32))
        .unwrap();
    let regions = dir.list_regions().unwrap();
    assert_eq!(regions.len(), 2);
}
