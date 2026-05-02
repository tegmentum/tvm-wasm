use tvm_core::{
    AllocatorKind, RegionDirectory, RegionKind, SlabAllocator, TvmError, VecBackedRegion,
};

#[test]
fn slab_alloc_dealloc_reuse() {
    let mut a = SlabAllocator::new(64, 16);
    let x = a.alloc(16).unwrap();
    let y = a.alloc(16).unwrap();
    let z = a.alloc(16).unwrap();
    let w = a.alloc(16).unwrap();
    assert!(a.alloc(16).is_err()); // full
    assert_eq!(a.used(), 64);

    a.dealloc(y).unwrap();
    a.dealloc(w).unwrap();

    // After freeing two slots, two more allocs succeed (LIFO order).
    let _e = a.alloc(16).unwrap();
    let _f = a.alloc(16).unwrap();
    let _ = (x, z);
}

#[test]
fn slab_rejects_wrong_size() {
    let mut a = SlabAllocator::new(64, 16);
    assert!(matches!(a.alloc(8), Err(TvmError::AllocationFailed)));
    assert!(matches!(a.alloc(20), Err(TvmError::AllocationFailed)));
}

#[test]
fn slab_dealloc_unaligned_offset_errors() {
    let mut a = SlabAllocator::new(32, 8);
    a.alloc(8).unwrap();
    assert!(matches!(a.dealloc(3), Err(TvmError::OutOfBounds)));
    assert!(matches!(a.dealloc(64), Err(TvmError::OutOfBounds)));
}

#[test]
fn slab_dealloc_double_free_errors() {
    let mut a = SlabAllocator::new(32, 8);
    let x = a.alloc(8).unwrap();
    a.dealloc(x).unwrap();
    assert!(matches!(a.dealloc(x), Err(TvmError::OutOfBounds)));
}

#[test]
fn slab_in_directory_alloc_dealloc() {
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r = dir
        .create_region_with(
            RegionKind::ObjectArena,
            64,
            AllocatorKind::Slab { class_size: 16 },
            VecBackedRegion::new(64),
        )
        .unwrap();
    let h = dir.alloc(r, 16).unwrap();
    dir.write(h, &[0xAA; 16]).unwrap();
    dir.dealloc(h).unwrap();
    let h2 = dir.alloc(r, 16).unwrap();
    assert_eq!(h2.offset, h.offset); // freed slot reused
}
