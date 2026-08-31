//! ADR-0029 Phase 6.9.b — wasmos-overhead benchmark.
//!
//! Compares the wasmtime-native `raw_linker` path against the
//! wasmos-backed `raw_linker_wasmos` path on the same workloads,
//! against the same guest wat, with the same host state.
//!
//! # Why this bench exists
//!
//! Phase 6.9.a Session 1 introduced `raw_linker_wasmos` as a
//! portable-across-adapters twin of `raw_linker`, but paid two
//! per-call costs the wasmtime path avoids:
//!
//! 1. `SharedTvmHost::lock()` per handler invocation (no `Store<T>`
//!    data slot to reach through under `CoreImports`).
//! 2. Guest-memory fetch through the adapter's `Caller::get_export`
//!    every call (no `cached_memory` cache).
//!
//! Rough back-of-envelope: an uncontended `std::sync::Mutex` is
//! 30-80ns, `Caller::get_export` is a HashMap lookup + Extern
//! downcast, probably 40-80ns. A raw `tvm.alloc` handler runs in
//! ~100-200ns wasmtime-native, so worst-case the wasmos overhead is
//! close to a doubling for pure-call workloads. Memory-touching
//! handlers should see less relative overhead (memcpy dominates).
//!
//! This bench measures the actual costs. If the delta on a hot
//! handler is large, the two escape hatches (`with_guest_memory_mut`
//! + `register_static`) can be brought in per-handler.
//!
//! # Shape
//!
//! Same statistics pattern as `benches/reducer_imports.rs`:
//! `SAMPLES=50` iterations after `WARMUP=5`, per size class,
//! `mann_whitney_u` for statistical significance. Instead of
//! comparing classic vs reducer within one path, we compare the
//! same handler across the two paths.
//!
//! Async cost: the wasmos path is async (`ModuleInstance::
//! call_function` is async). We run one tokio `current_thread`
//! runtime for the entire bench (one setup cost) and each iteration
//! `block_on`s a single call. Comparing this to the sync wasmtime
//! path measures the true per-call delta.

use std::time::{Duration, Instant};

use tvm_core::RegionKind;
use tvm_test_harness::mann_whitney_u;
use tvm_wasmtime::raw_linker_wasmos::add_raw_imports as add_raw_imports_wasmos;
use tvm_wasmtime::shared_host::SharedTvmHost;
use tvm_wasmtime::{add_raw_imports, TvmHost};
use wasmos_runtime_api::{
    Bytes, CompileOptions, ComponentSource, CoreImports, CoreValue, ExecutionContext,
    ModuleInstance, Runtime,
};
use wasmos_runtime_wasmtime_v48::WasmtimeV48Runtime;
use wasmtime::{Config, Engine, Linker, Module, Store};

const SAMPLES: usize = 50;
const WARMUP: usize = 5;
const SIZES: &[u32] = &[64, 1024, 16 * 1024];

fn pct(sorted: &[Duration], p: f64) -> u128 {
    let n = sorted.len();
    if n == 0 {
        return 0;
    }
    let idx = ((p / 100.0) * (n as f64 - 1.0)).round() as usize;
    sorted[idx.min(n - 1)].as_nanos()
}

fn time_loop<F: FnMut() -> anyhow::Result<()>>(mut f: F) -> anyhow::Result<Vec<Duration>> {
    for _ in 0..WARMUP {
        f()?;
    }
    let mut t = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let s = Instant::now();
        f()?;
        t.push(s.elapsed());
    }
    Ok(t)
}

fn report(label: &str, size: u32, t: &mut [Duration]) {
    t.sort();
    let mean = t.iter().map(|d| d.as_nanos() as f64).sum::<f64>() / t.len() as f64;
    let p99 = pct(t, 99.0);
    let bw = if size > 0 {
        (size as f64) / (mean / 1e9) / (1u64 << 30) as f64
    } else {
        0.0
    };
    println!(
        "  {:<38} mean={:>10.0}ns  p99={:>10}ns  {:>5.2} GiB/s",
        label, mean, p99, bw
    );
}

fn pair_summary(
    label_wt: &str,
    label_wo: &str,
    wt: Vec<Duration>,
    wo: Vec<Duration>,
    size: u32,
) {
    let mut wt = wt;
    let mut wo = wo;
    report(label_wt, size, &mut wt);
    report(label_wo, size, &mut wo);
    let raw_wt: Vec<u128> = wt.iter().map(|d| d.as_nanos()).collect();
    let raw_wo: Vec<u128> = wo.iter().map(|d| d.as_nanos()).collect();
    let mean = |v: &[u128]| v.iter().map(|n| *n as f64).sum::<f64>() / v.len() as f64;
    let overhead = (mean(&raw_wo) - mean(&raw_wt)) / mean(&raw_wt) * 100.0;
    let u = mann_whitney_u(&raw_wt, &raw_wo);
    println!(
        "    wasmos overhead vs wasmtime = {:+.1}%   U={:.3}",
        overhead, u
    );
}

// ── Workload wats ────────────────────────────────────────────────────

/// alloc/dealloc round trip — no memory touch. Isolates per-call
/// vtable + mutex + guest-memory-fetch costs from the memcpy path.
const ALLOC_DEALLOC_WAT: &str = r#"
(module
  (import "tvm" "alloc"   (func $alloc   (param i32 i32) (result i64)))
  (import "tvm" "dealloc" (func $dealloc (param i64) (result i32)))
  (memory (export "memory") 1)
  (func (export "spin") (param $region i32) (param $size i32) (result i32)
    (local $h i64)
    (local.set $h (call $alloc (local.get $region) (local.get $size)))
    (call $dealloc (local.get $h)))
)
"#;

/// sum_u8 — pure reducer, region-only. No guest memory access at all
/// (the reducer runs entirely inside TvmHost). Measures per-call
/// dispatch cost + reducer scan cost.
const SUM_U8_WAT: &str = r#"
(module
  (import "tvm" "sum_u8" (func $sum (param i64 i32) (result i64)))
  (memory (export "memory") 1)
  (func (export "sum") (param $h i64) (param $len i32) (result i64)
    (call $sum (local.get $h) (local.get $len)))
)
"#;

/// write — memory-touching. Guest already has bytes staged at
/// offset 0; each iteration copies them into the region. Measures
/// per-call dispatch + guest_memory_read + region write.
const WRITE_WAT: &str = r#"
(module
  (import "tvm" "write" (func $write (param i64 i32 i32) (result i32)))
  (memory (export "memory") 8)
  (func (export "write") (param $h i64) (param $len i32) (result i32)
    (call $write (local.get $h) (i32.const 0) (local.get $len)))
)
"#;

/// read — memory-touching. Copies region bytes into guest memory
/// at offset 0. Measures per-call dispatch + region read +
/// guest_memory_write.
const READ_WAT: &str = r#"
(module
  (import "tvm" "read" (func $read (param i64 i32 i32) (result i32)))
  (memory (export "memory") 8)
  (func (export "read") (param $h i64) (param $len i32) (result i32)
    (call $read (local.get $h) (i32.const 0) (local.get $len)))
)
"#;

// ── Wasmtime-native runner ───────────────────────────────────────────

fn setup_wasmtime(size: u32, data: &[u8]) -> anyhow::Result<(Store<TvmHost>, u16, i64)> {
    let host = TvmHost::new();
    let engine = Engine::new(Config::new().wasm_multi_memory(true))?;
    let mut store = Store::new(&engine, host);
    let region = store.data_mut().create_region(RegionKind::HotHeap, size + 4096)?;
    let h = store.data_mut().alloc(region, size)?;
    store.data_mut().write_bytes(h, data)?;
    Ok((store, region, h.pack() as i64))
}

fn run_wasmtime_alloc(size: u32) -> anyhow::Result<Vec<Duration>> {
    let (mut store, region, _) = setup_wasmtime(size, &vec![0u8; size as usize])?;
    // Grow the region so alloc/dealloc pairs have space.
    // (setup_wasmtime already gave +4096 headroom.)
    let module = Module::new(store.engine(), ALLOC_DEALLOC_WAT)?;
    let mut linker: Linker<TvmHost> = Linker::new(store.engine());
    add_raw_imports(&mut linker)?;
    let instance = linker.instantiate(&mut store, &module)?;
    let f = instance.get_typed_func::<(i32, i32), i32>(&mut store, "spin")?;
    time_loop(|| {
        let _ = f.call(&mut store, (region as i32, 16))?;
        Ok(())
    })
}

fn run_wasmtime_workload(wat: &str, fn_name: &str, size: u32, data: &[u8]) -> anyhow::Result<Vec<Duration>> {
    let (mut store, _, packed) = setup_wasmtime(size, data)?;
    let module = Module::new(store.engine(), wat)?;
    let mut linker: Linker<TvmHost> = Linker::new(store.engine());
    add_raw_imports(&mut linker)?;
    let instance = linker.instantiate(&mut store, &module)?;
    match fn_name {
        "sum" => {
            let f = instance.get_typed_func::<(i64, i32), i64>(&mut store, "sum")?;
            time_loop(|| {
                let _ = f.call(&mut store, (packed, size as i32))?;
                Ok(())
            })
        }
        "write" | "read" => {
            let f = instance.get_typed_func::<(i64, i32), i32>(&mut store, fn_name)?;
            time_loop(|| {
                let _ = f.call(&mut store, (packed, size as i32))?;
                Ok(())
            })
        }
        other => anyhow::bail!("unknown fn_name {other:?}"),
    }
}

// ── Wasmos runner ────────────────────────────────────────────────────

struct WasmosSetup {
    _shared: SharedTvmHost,
    _rt: WasmtimeV48Runtime,
    packed: i64,
    region: u16,
    instance: ModuleInstance,
    rt_handle: tokio::runtime::Handle,
}

fn setup_wasmos(
    tokio_rt: &tokio::runtime::Runtime,
    wat: &str,
    size: u32,
    data: &[u8],
) -> anyhow::Result<WasmosSetup> {
    let shared = SharedTvmHost::new();
    let (region, packed) = {
        let mut g = shared.lock();
        let r = g.create_region(RegionKind::HotHeap, size + 4096)?;
        let h = g.alloc(r, size)?;
        g.write_bytes(h, data)?;
        (r, h.pack() as i64)
    };
    let rt = WasmtimeV48Runtime::new(Default::default())?;
    let instance = tokio_rt.block_on(async {
        let wasm: Vec<u8> = wat::parse_str(wat)?;
        let compiled = rt
            .compile_module(
                ComponentSource::Bytes {
                    bytes: Bytes::from(wasm),
                    name: None,
                },
                CompileOptions::default(),
            )
            .await?;
        let core_imports = add_raw_imports_wasmos(CoreImports::new(), shared.clone());
        let ctx = ExecutionContext {
            core_imports,
            ..ExecutionContext::new()
        };
        anyhow::Ok(rt.instantiate_module(&compiled, ctx).await?)
    })?;
    Ok(WasmosSetup {
        _shared: shared,
        _rt: rt,
        packed,
        region,
        instance,
        rt_handle: tokio_rt.handle().clone(),
    })
}

fn run_wasmos_alloc(tokio_rt: &tokio::runtime::Runtime, size: u32) -> anyhow::Result<Vec<Duration>> {
    let data = vec![0u8; size as usize];
    let mut s = setup_wasmos(tokio_rt, ALLOC_DEALLOC_WAT, size, &data)?;
    let region = s.region;
    let handle = s.rt_handle.clone();
    time_loop(|| {
        let _ = handle.block_on(async {
            s.instance
                .call_function("spin", &[CoreValue::I32(region as i32), CoreValue::I32(16)])
                .await
        })?;
        Ok(())
    })
}

fn run_wasmos_workload(
    tokio_rt: &tokio::runtime::Runtime,
    wat: &str,
    fn_name: &str,
    size: u32,
    data: &[u8],
) -> anyhow::Result<Vec<Duration>> {
    let mut s = setup_wasmos(tokio_rt, wat, size, data)?;
    let packed = s.packed;
    let handle = s.rt_handle.clone();
    let sz = size as i32;
    match fn_name {
        "sum" => time_loop(|| {
            let out = handle.block_on(async {
                s.instance
                    .call_function("sum", &[CoreValue::I64(packed), CoreValue::I32(sz)])
                    .await
            })?;
            assert!(matches!(out.as_slice(), [CoreValue::I64(_)]));
            Ok(())
        }),
        "write" | "read" => time_loop(|| {
            let _ = handle.block_on(async {
                s.instance
                    .call_function(fn_name, &[CoreValue::I64(packed), CoreValue::I32(sz)])
                    .await
            })?;
            Ok(())
        }),
        other => anyhow::bail!("unknown fn_name {other:?}"),
    }
}

// ── Main ─────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    println!("==> wasmos-overhead benchmark (Phase 6.9.b)");
    println!("    {} samples + {} warmup", SAMPLES, WARMUP);
    println!("    compares wasmtime-native raw_linker vs wasmos-backed raw_linker_wasmos");
    println!();

    let tokio_rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    // ── alloc/dealloc — pure call overhead ──────────────────────────
    println!("--- alloc + dealloc (region-only, no memory touch) ---");
    let wt = run_wasmtime_alloc(4096)?;
    let wo = run_wasmos_alloc(&tokio_rt, 4096)?;
    pair_summary(
        "    wasmtime (Linker::func_wrap)",
        "    wasmos   (CoreImports::register)",
        wt,
        wo,
        0,
    );
    println!();

    for &size in SIZES {
        let data: Vec<u8> = (0..size).map(|i| (i & 0xFF) as u8).collect();
        println!("--- size = {} bytes ---", size);

        // sum_u8 — region-only reducer
        println!("  [sum_u8]");
        let wt = run_wasmtime_workload(SUM_U8_WAT, "sum", size, &data)?;
        let wo = run_wasmos_workload(&tokio_rt, SUM_U8_WAT, "sum", size, &data)?;
        pair_summary(
            "    wasmtime  sum_u8",
            "    wasmos    sum_u8",
            wt,
            wo,
            size,
        );

        // write — guest linear memory -> region
        println!("  [write]");
        let wt = run_wasmtime_workload(WRITE_WAT, "write", size, &data)?;
        let wo = run_wasmos_workload(&tokio_rt, WRITE_WAT, "write", size, &data)?;
        pair_summary(
            "    wasmtime  write",
            "    wasmos    write",
            wt,
            wo,
            size,
        );

        // read — region -> guest linear memory
        println!("  [read]");
        let wt = run_wasmtime_workload(READ_WAT, "read", size, &data)?;
        let wo = run_wasmos_workload(&tokio_rt, READ_WAT, "read", size, &data)?;
        pair_summary(
            "    wasmtime  read",
            "    wasmos    read",
            wt,
            wo,
            size,
        );

        println!();
    }

    Ok(())
}
