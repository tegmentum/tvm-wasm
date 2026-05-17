use tvm_core::backing::VecBackedRegion;
use tvm_core::{RegionDirectory, RegionKind, TvmError};

#[test]
fn handle_from_old_generation_is_stale() {
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r = dir
        .create_region(RegionKind::ObjectArena, 64, VecBackedRegion::new(64))
        .unwrap();
    let h = dir.alloc(r, 8).unwrap();

    dir.bump_generation(r).unwrap();

    let mut out = [0u8; 8];
    assert!(matches!(dir.read(h, &mut out), Err(TvmError::StaleHandle)));
}

#[test]
fn handle_for_unknown_region_errors() {
    let dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let bad = tvm_core::Handle {
        region_id: 99,
        generation: 1,
        offset: 0,
    };
    let mut out = [0u8; 4];
    assert!(matches!(
        dir.read(bad, &mut out),
        Err(TvmError::RegionNotFound(99))
    ));
}
