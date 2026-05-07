//! Spill / paging at scale — validates that the directory's
//! spill+load round-trip is correct across many regions and many
//! cycles, far past the single-region single-cycle coverage in
//! `paging.rs`.
//!
//! ## What this exercises
//!
//! `paging.rs` proves the round-trip works once on a 64-byte region.
//! This test:
//!   * stands up `N_REGIONS = 16` independent regions of 1 MiB each;
//!   * writes a unique per-region pattern into every region;
//!   * runs `CYCLES = 4` outer rounds where every region is spilled,
//!     re-loaded, partially mutated, spilled, and re-loaded again;
//!   * verifies bytes after every load and verifies fault/promotion/
//!     demotion counters increased by the expected amount per region.
//!
//! 16 MiB × 4 cycles ≈ 128 MiB of disk traffic — small enough that
//! the test runs in <1 s on local disk but big enough that any
//! per-byte corruption in the spill/load path will surface.
//!
//! Not `#[ignore]`'d: this is meant to be part of the default test
//! suite as the canonical "is paging correct?" check.

use tempfile::tempdir;
use tvm_core::{
    AllocatorKind, FileBackingStore, RegionDirectory, RegionKind, Residency, VecBackedRegion,
};

const N_REGIONS: u16 = 16;
const REGION_BYTES: u32 = 1024 * 1024; // 1 MiB
const CYCLES: u32 = 4;

/// Per-region byte pattern. Mixes the region id and offset so any
/// cross-wired spill/load (e.g. region A's bytes ending up under
/// region B's id) is detectable byte-for-byte.
fn pattern_byte(region_id: u16, offset: u32) -> u8 {
    let r = region_id as u32;
    ((r.wrapping_mul(0x9E37_79B1)).wrapping_add(offset).wrapping_mul(0x85EB_CA77) >> 24) as u8
}

fn fill_pattern(region_id: u16) -> Vec<u8> {
    (0..REGION_BYTES).map(|off| pattern_byte(region_id, off)).collect()
}

#[test]
fn spill_load_at_scale_preserves_bytes_across_cycles() {
    let tmp = tempdir().unwrap();
    let mut backing = FileBackingStore::new(tmp.path()).unwrap();

    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let mut region_ids = Vec::with_capacity(N_REGIONS as usize);
    for _ in 0..N_REGIONS {
        let r = dir
            .create_region_with(
                // ObjectArena is spillable; PageStore would also work,
                // but ObjectArena starts in Hot residency which makes
                // the demotion->Cold transition unambiguous in metrics.
                RegionKind::ObjectArena,
                REGION_BYTES,
                AllocatorKind::Bump,
                VecBackedRegion::new(REGION_BYTES),
            )
            .unwrap();
        region_ids.push(r);
    }

    // One bump allocation per region for the full capacity, then write
    // the per-region pattern. `expected[i]` tracks the authoritative
    // bytes for `region_ids[i]` and is updated alongside any in-place
    // mutations we make so post-load verification stays in sync.
    let mut handles = Vec::with_capacity(N_REGIONS as usize);
    let mut expected: Vec<Vec<u8>> = Vec::with_capacity(N_REGIONS as usize);
    for &r in &region_ids {
        let h = dir.alloc(r, REGION_BYTES).unwrap();
        let bytes = fill_pattern(r);
        dir.write(h, &bytes).unwrap();
        handles.push(h);
        expected.push(bytes);
    }

    // Snapshot per-region metrics before any spill.
    let baseline: Vec<_> = region_ids
        .iter()
        .map(|&r| dir.metrics(r).unwrap().snapshot())
        .collect();

    let mut buf = vec![0u8; REGION_BYTES as usize];
    for cycle in 0..CYCLES {
        // Spill every region.
        for &r in &region_ids {
            dir.spill_region(r, &mut backing).unwrap();
            assert_eq!(
                dir.region_info(r).unwrap().residency,
                Residency::Cold,
                "region {r} should be Cold after spill (cycle {cycle})"
            );
        }

        // Reload + verify pattern survives byte-for-byte.
        for (idx, &r) in region_ids.iter().enumerate() {
            dir.load_region(r, &mut backing).unwrap();
            assert_eq!(
                dir.region_info(r).unwrap().residency,
                Residency::Hot,
                "region {r} should be Hot after load (cycle {cycle})"
            );
            dir.read(handles[idx], &mut buf).unwrap();
            assert!(
                buf == expected[idx],
                "region {r} cycle {cycle}: byte-level mismatch after load"
            );
        }

        // Mutate one byte in each region so the next spill carries
        // fresh content (catches any path that silently re-uses the
        // earlier-spilled bytes instead of re-spilling current state).
        // Update `expected` to track the mutation.
        let stamp_offset = (cycle * 17) % (REGION_BYTES - 1);
        for (idx, &r) in region_ids.iter().enumerate() {
            let stamp_byte = (r as u8).wrapping_add(cycle as u8);
            let h = handles[idx];
            dir.write(
                tvm_core::Handle {
                    region_id: r,
                    generation: h.generation,
                    offset: h.offset + stamp_offset,
                },
                &[stamp_byte],
            )
            .unwrap();
            expected[idx][stamp_offset as usize] = stamp_byte;
        }
    }

    // After CYCLES round-trips, every region should show:
    //   demotions += CYCLES   (one per spill_region)
    //   promotions += CYCLES  (one per load_region)
    //   faults     += CYCLES  (one per load_region)
    for (idx, &r) in region_ids.iter().enumerate() {
        let after = dir.metrics(r).unwrap().snapshot();
        let before = baseline[idx];
        assert_eq!(
            after.demotions - before.demotions,
            CYCLES as u64,
            "region {r}: demotions delta != CYCLES"
        );
        assert_eq!(
            after.promotions - before.promotions,
            CYCLES as u64,
            "region {r}: promotions delta != CYCLES"
        );
        assert_eq!(
            after.faults - before.faults,
            CYCLES as u64,
            "region {r}: faults delta != CYCLES"
        );
    }
}
