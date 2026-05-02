use tvm_core::ResolveCache;
use tvm_core::{RegionDirectory, RegionKind, VecBackedRegion};

#[test]
fn cache_collisions_evict_old_slot() {
    // The cache is 8-way direct-mapped, keyed by `region_id & 7`. Region IDs
    // 0 and 8 collide on slot 0; both lookups should produce correct
    // results, but installing region 8 evicts region 0 from the cache.
    let mut cache = ResolveCache::new();

    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let mut ids = Vec::new();
    for _ in 0..9 {
        let id = dir
            .create_region(RegionKind::HotHeap, 16, VecBackedRegion::new(16))
            .unwrap();
        ids.push(id);
    }

    // Warm up the cache for the first 8 regions.
    for &id in &ids[..8] {
        let info = *dir.region_info(id).unwrap();
        cache.install(&info);
    }
    // All 8 hit.
    for &id in &ids[..8] {
        assert!(cache.lookup(id).is_some(), "region {id} should hit");
    }

    // Region 8 collides with region 0. Installing it evicts region 0.
    let info_8 = *dir.region_info(ids[8]).unwrap();
    cache.install(&info_8);
    assert!(
        cache.lookup(ids[0]).is_none(),
        "collision should have evicted region 0"
    );
    assert!(cache.lookup(ids[8]).is_some());
}

#[test]
fn cache_lookup_after_collision_returns_correct_data() {
    let mut cache = ResolveCache::new();
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();

    // Skip region IDs 0..7 by creating dummies, so the next two regions
    // share slot 0 (id 8 and id 16). Easier: just install hand-crafted
    // entries.
    use tvm_core::{Region, Residency};
    let r0 = Region {
        id: 0,
        generation: 1,
        kind: RegionKind::HotHeap,
        capacity: 100,
        used: 0,
        residency: Residency::Hot,
        pinned: false,
        pinnable: true,
        spillable: false,
    };
    let r8 = Region { id: 8, capacity: 200, ..r0 };

    cache.install(&r0);
    let hit0 = cache.lookup(0).unwrap();
    assert_eq!(hit0.capacity, 100);

    cache.install(&r8);
    let hit8 = cache.lookup(8).unwrap();
    assert_eq!(hit8.capacity, 200);
    // Region 0 is gone from the cache (collision), even though it still
    // exists in the directory.
    assert!(cache.lookup(0).is_none());

    // Avoid unused-variable warning.
    let _ = &dir;
}
