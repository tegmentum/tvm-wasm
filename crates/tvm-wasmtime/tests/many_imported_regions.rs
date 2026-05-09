//! Many-imported-regions stress test — validates that one wasmtime
//! `Store` can host enough imported memories to span tens of GiB and
//! that the linker, instantiation, and dispatch keep working past the
//! single-digit memory counts other tests use.
//!
//! ## What this exercises
//!
//! Other imported-region tests use 1–3 memories. This test stands up
//! 64 imported regions in a single store and instantiates a WAT module
//! with 64 memory imports + 64 native read functions, proving:
//!   * `Linker::define` scales to dozens of memory imports.
//!   * Wasmtime's per-store bookkeeping holds up.
//!   * The WAT validator accepts a module with that many memory imports
//!     (the wasm-spec ceiling per `wasmparser` is 100).
//!
//! ## Aggregate size
//!
//! 64 regions × 256 MiB capacity = 16 GiB aggregate addressable, well
//! past the 4 GiB single-memory ceiling. The test uses the pooling
//! allocator (`pooling_imported_region_engine_config`) so memories are
//! served from a pre-reserved pool sized exactly for the workload —
//! 64 slots of 256 MiB each — bringing the virtual footprint to
//! ~16 GiB with zero waste and zero per-memory mmap churn. Committed
//! RSS stays under a few MiB — only the sentinel pages we touch fault
//! in.

use std::fmt::Write as _;

use tvm_core::RegionKind;
use tvm_wasmtime::{create_imported_in_store, pooling_imported_region_engine_config, TvmHost};
use wasmtime::{Engine, Linker, Module, Store};

const N_REGIONS: u16 = 64;
const REGION_BYTES: u32 = 256 * 1024 * 1024; // 256 MiB

fn build_wat(n: u16) -> String {
    let mut s = String::with_capacity(8 * 1024);
    s.push_str("(module\n");
    for i in 0..n {
        writeln!(s, "  (import \"tvm\" \"r{i}\" (memory $r{i} 1))").unwrap();
    }
    for i in 0..n {
        writeln!(
            s,
            "  (func (export \"read_r{i}\") (param $off i32) (result i64) \
             (i64.load $r{i} (local.get $off)))"
        )
        .unwrap();
    }
    s.push_str(")\n");
    s
}

fn make_engine() -> anyhow::Result<Engine> {
    Ok(Engine::new(&pooling_imported_region_engine_config(
        N_REGIONS as u32,
        REGION_BYTES,
    ))?)
}

#[test]
fn many_imported_regions_round_trip() -> anyhow::Result<()> {
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

    // Plant a unique sentinel u64 at offset 0 in each region. The
    // sentinel encodes the region id so we can detect any cross-wiring
    // bug at instantiation time.
    for region_id in 0..N_REGIONS {
        let memory = store.data().imported_region(region_id).unwrap().memory();
        let value: u64 = 0xCAFE_BABE_0000_0000 | (region_id as u64);
        memory.write(&mut store, 0, &value.to_le_bytes())?;
    }

    let imports: Vec<_> = store
        .data()
        .imported
        .iter()
        .map(|r| (r.import_name(), r.memory()))
        .collect();
    let mut linker: Linker<TvmHost> = Linker::new(&engine);
    for (name, m) in imports {
        linker.define(&mut store, "tvm", &name, m)?;
    }

    let wat = build_wat(N_REGIONS);
    let module = Module::new(&engine, &wat)?;
    let instance = linker.instantiate(&mut store, &module)?;

    for region_id in 0..N_REGIONS {
        let f = instance
            .get_typed_func::<i32, i64>(&mut store, &format!("read_r{region_id}"))?;
        let got = f.call(&mut store, 0)? as u64;
        let expected = 0xCAFE_BABE_0000_0000 | (region_id as u64);
        assert_eq!(
            got, expected,
            "region {region_id}: got {got:#x}, expected {expected:#x}"
        );
    }
    Ok(())
}
