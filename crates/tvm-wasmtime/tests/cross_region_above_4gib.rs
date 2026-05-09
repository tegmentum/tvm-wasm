//! Cross-region >4 GiB stress test — validates the central architectural
//! claim that TVM addresses aggregate data >4 GiB by composing multiple
//! imported wasm32 memories. Each region is its own 32-bit memory; the
//! guest switches between them via the `memory` immediate on each load,
//! so there is no per-instruction address-width tax.
//!
//! ## What this exercises that other tests don't
//!
//! Every other test in this crate uses small regions (≤64 KiB). This is
//! the only test that:
//!   * allocates individual regions large enough to push wasmtime's
//!     static-memory reservation past 1 GiB;
//!   * writes to byte offsets the guest must reach via native `i64.load
//!     $rN` against memory immediates;
//!   * sums to >4 GiB across the regions, exceeding what any single
//!     wasm32 memory could address.
//!
//! ## Runtime cost
//!
//! With `imported_region_engine_config` (`memory_may_move(false)`) and
//! `MemoryType::new(pages, Some(pages))` set in `ImportedRegion::new`,
//! each memory reserves exactly its declared capacity plus the default
//! 32 MiB guard. Three 2 GiB regions ≈ 6.1 GiB virtual reservation total.
//! Touched pages account for under 1 MiB of physical RSS — only the
//! offsets we actually write fault in.

use std::time::Instant;

use tvm_core::RegionKind;
use tvm_wasmtime::{create_imported_in_store, imported_region_engine_config, TvmHost};
use wasmtime::{Engine, Linker, Module, Store};

/// 2 GiB per region. Three regions = 6 GiB aggregate, comfortably past
/// the 4 GiB single-memory ceiling. `1 << 31` fits in `u32` and leaves
/// room below `i32::MAX` so `i32`-typed wasm offsets stay valid.
const REGION_BYTES: u32 = 1u32 << 31;
const N_REGIONS: u16 = 3;

/// WAT module that imports three memories and exposes one native load
/// per memory plus an aggregate-sum function that reads from all three.
/// `i64.load $rN` is fixed at compile time to a specific memory by the
/// memory immediate — no runtime selector needed.
const PROBE_WAT: &str = r#"
(module
  (import "tvm" "r0" (memory $r0 1))
  (import "tvm" "r1" (memory $r1 1))
  (import "tvm" "r2" (memory $r2 1))

  (func (export "read_r0") (param $off i32) (result i64)
    (i64.load $r0 (local.get $off)))
  (func (export "read_r1") (param $off i32) (result i64)
    (i64.load $r1 (local.get $off)))
  (func (export "read_r2") (param $off i32) (result i64)
    (i64.load $r2 (local.get $off)))

  ;; Sum `count` u64s starting at `base` across all three memories.
  ;; Each iteration reads one u64 from each memory and accumulates.
  ;; Used to measure aggregate throughput across regions.
  (func (export "sum3") (param $base i32) (param $count i32) (result i64)
    (local $i i32) (local $acc i64) (local $off i32)
    (block $break
      (loop $continue
        (br_if $break (i32.eq (local.get $i) (local.get $count)))
        (local.set $off
          (i32.add (local.get $base)
                   (i32.shl (local.get $i) (i32.const 3))))
        (local.set $acc
          (i64.add (local.get $acc) (i64.load $r0 (local.get $off))))
        (local.set $acc
          (i64.add (local.get $acc) (i64.load $r1 (local.get $off))))
        (local.set $acc
          (i64.add (local.get $acc) (i64.load $r2 (local.get $off))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $continue)))
    (local.get $acc))
)
"#;

fn make_engine() -> anyhow::Result<Engine> {
    Ok(Engine::new(&imported_region_engine_config())?)
}

fn wire_imports(
    engine: &Engine,
    store: &mut Store<TvmHost>,
) -> anyhow::Result<Linker<TvmHost>> {
    let imports: Vec<_> = store
        .data()
        .imported
        .iter()
        .map(|r| (r.import_name(), r.memory()))
        .collect();
    let mut linker: Linker<TvmHost> = Linker::new(engine);
    for (name, m) in imports {
        linker.define(&mut *store, "tvm", &name, m)?;
    }
    Ok(linker)
}

#[test]
fn cross_region_above_4gib_sentinels() -> anyhow::Result<()> {
    let engine = make_engine()?;
    let mut store = Store::new(&engine, TvmHost::new());

    for _ in 0..N_REGIONS {
        create_imported_in_store(&mut store, RegionKind::HotHeap, REGION_BYTES)?;
    }
    let aggregate = REGION_BYTES as u64 * N_REGIONS as u64;
    assert!(
        aggregate > (1u64 << 32),
        "test premise: aggregate {aggregate} should exceed 4 GiB"
    );

    // For each region, plant unique 8-byte sentinels at start, middle,
    // and end-minus-8. Middle and end force commits past the 1 GiB mark
    // *within a single region*, and the aggregate spans >4 GiB.
    let offsets: [u32; 3] = [0, REGION_BYTES / 2, REGION_BYTES - 8];
    for region_idx in 0..N_REGIONS {
        let memory = store.data().imported_region(region_idx).unwrap().memory();
        let base: u64 = 0xDEAD_BEEF_0000_0000 | (region_idx as u64);
        for (slot, off) in offsets.iter().enumerate() {
            let value = base.wrapping_add(slot as u64 * 0x1_0000);
            memory.write(&mut store, *off as usize, &value.to_le_bytes())?;
        }
    }

    let linker = wire_imports(&engine, &mut store)?;
    let module = Module::new(&engine, PROBE_WAT)?;
    let instance = linker.instantiate(&mut store, &module)?;
    let read_fns: Vec<_> = (0..N_REGIONS)
        .map(|i| {
            instance
                .get_typed_func::<i32, i64>(&mut store, &format!("read_r{i}"))
                .unwrap()
        })
        .collect();

    for region_idx in 0..N_REGIONS {
        let base: u64 = 0xDEAD_BEEF_0000_0000 | (region_idx as u64);
        for (slot, off) in offsets.iter().enumerate() {
            let expected = base.wrapping_add(slot as u64 * 0x1_0000);
            let got = read_fns[region_idx as usize].call(&mut store, *off as i32)? as u64;
            assert_eq!(
                got, expected,
                "region {region_idx} offset {off}: got {got:#x}, expected {expected:#x}"
            );
        }
    }
    Ok(())
}

#[test]
fn cross_region_above_4gib_throughput() -> anyhow::Result<()> {
    // Touch a 1 MiB window starting at offset 0 in each region. The window
    // is small (so the test runs in milliseconds) but the regions still
    // span >4 GiB aggregate virtual address space — the point is to prove
    // the *guest* can issue native loads across all three memories without
    // a single one of them being >4 GiB.
    const WINDOW_U64S: u32 = 128 * 1024; // 1 MiB / 8 B = 128 K u64s per region
    const ITERS: u32 = 64; // amortize trampoline cost

    let engine = make_engine()?;
    let mut store = Store::new(&engine, TvmHost::new());

    for _ in 0..N_REGIONS {
        create_imported_in_store(&mut store, RegionKind::HotHeap, REGION_BYTES)?;
    }

    // Fill the 1 MiB window in each region with a known pattern so we
    // can validate the sum.
    let mut buf = vec![0u8; WINDOW_U64S as usize * 8];
    for region_idx in 0..N_REGIONS {
        for i in 0..WINDOW_U64S as usize {
            let v = (region_idx as u64).wrapping_add(i as u64);
            buf[i * 8..(i + 1) * 8].copy_from_slice(&v.to_le_bytes());
        }
        let memory = store.data().imported_region(region_idx).unwrap().memory();
        memory.write(&mut store, 0, &buf)?;
    }

    let linker = wire_imports(&engine, &mut store)?;
    let module = Module::new(&engine, PROBE_WAT)?;
    let instance = linker.instantiate(&mut store, &module)?;
    let sum3 = instance.get_typed_func::<(i32, i32), i64>(&mut store, "sum3")?;

    // Expected: sum over i in [0, WINDOW_U64S) of (0 + i) + (1 + i) + (2 + i).
    let n = WINDOW_U64S as u64;
    let triangular = n * (n - 1) / 2;
    let single_iter_sum = 3 * triangular + n * (0 + 1 + 2);

    // Warm up.
    let _ = sum3.call(&mut store, (0, WINDOW_U64S as i32))?;

    let start = Instant::now();
    let mut acc: u64 = 0;
    for _ in 0..ITERS {
        acc = acc.wrapping_add(sum3.call(&mut store, (0, WINDOW_U64S as i32))? as u64);
    }
    let elapsed = start.elapsed();

    assert_eq!(
        acc,
        single_iter_sum.wrapping_mul(ITERS as u64),
        "guest sum across 3 imported memories did not match expected aggregate"
    );

    let bytes_per_iter = (WINDOW_U64S as u64) * 8 * (N_REGIONS as u64); // 3 reads × 8 B per inner step
    let total_bytes = bytes_per_iter * ITERS as u64;
    let gib_per_s = (total_bytes as f64) / elapsed.as_secs_f64() / (1u64 << 30) as f64;
    eprintln!(
        "cross-region throughput: {:.2} GiB/s ({} bytes in {:?}, {} iters × 3 regions × {} KiB)",
        gib_per_s,
        total_bytes,
        elapsed,
        ITERS,
        WINDOW_U64S * 8 / 1024,
    );

    // Soft floor — native wasm loads should clear 1 GiB/s easily on
    // anything modern. This is a sanity check, not a perf gate.
    assert!(
        gib_per_s > 1.0,
        "cross-region throughput {gib_per_s:.2} GiB/s below sanity floor"
    );

    Ok(())
}
