//! Reducer imports benchmark — proves that collapsing
//! `host.read → guest scalar loop` into a single host call wins for
//! wasm-guest callers.
//!
//! Two paths under test, both compute "sum of bytes in a region":
//!
//!   1. **classic**: guest calls `tvm.read` to copy region bytes into
//!      its own linear memory, then sums in a scalar wasm loop. Two
//!      host trampolines effectively (one read, one return) plus a
//!      memcpy of `len` bytes plus a per-byte wasm loop.
//!
//!   2. **reducer**: guest calls `tvm.sum_u8(handle, len)` and the
//!      host returns the answer. One trampoline, no copy out, no
//!      guest-side loop. Host implementation is autovec'd Rust over
//!      the region's `&[u8]` slice.
//!
//! Both paths share the same `TvmHost` so the underlying region
//! representation is identical — the only difference is the call
//! shape across the boundary.

use std::time::{Duration, Instant};
use tvm_core::RegionKind;
use tvm_test_harness::mann_whitney_u;
use tvm_wasmtime::{add_raw_imports, TvmHost};
use wasmtime::{Config, Engine, Linker, Module, Store};

const SAMPLES: usize = 50;
const WARMUP: usize = 5;
const SIZES: &[u32] = &[1024, 16 * 1024, 65536, 262_144];

fn pct(sorted: &[Duration], p: f64) -> u128 {
    let n = sorted.len();
    if n == 0 { return 0; }
    let idx = ((p / 100.0) * (n as f64 - 1.0)).round() as usize;
    sorted[idx.min(n - 1)].as_nanos()
}

fn time_loop<F: FnMut() -> anyhow::Result<()>>(
    mut f: F,
) -> anyhow::Result<Vec<Duration>> {
    for _ in 0..WARMUP { f()?; }
    let mut t = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let s = Instant::now();
        f()?;
        t.push(s.elapsed());
    }
    Ok(t)
}

fn report(label: &str, size: u32, t: &mut Vec<Duration>) {
    t.sort();
    let mean = t.iter().map(|d| d.as_nanos() as f64).sum::<f64>()
        / t.len() as f64;
    let p99 = pct(t, 99.0);
    let bw = (size as f64) / (mean / 1e9) / (1u64 << 30) as f64;
    println!(
        "  {:<22} mean={:>10.0}ns  p99={:>10}ns  {:>5.2} GiB/s",
        label, mean, p99, bw
    );
}

const CLASSIC_WAT: &str = r#"
(module
  (import "tvm" "read" (func $read (param i64 i32 i32) (result i32)))
  (memory (export "memory") 8)
  (func (export "sum_via_read")
        (param $h i64) (param $len i32) (result i64)
    (local $cur i32) (local $end i32) (local $acc i64)
    (drop (call $read (local.get $h) (i32.const 0) (local.get $len)))
    (local.set $end (local.get $len))
    (block $break
      (loop $continue
        (br_if $break (i32.eq (local.get $cur) (local.get $end)))
        (local.set $acc
          (i64.add (local.get $acc)
                   (i64.load8_u (local.get $cur))))
        (local.set $cur (i32.add (local.get $cur) (i32.const 1)))
        (br $continue)))
    (local.get $acc))
)
"#;

const REDUCER_WAT: &str = r#"
(module
  (import "tvm" "sum_u8" (func $sum (param i64 i32) (result i64)))
  (memory (export "memory") 1)
  (func (export "sum_via_reducer") (param $h i64) (param $len i32) (result i64)
    (call $sum (local.get $h) (local.get $len)))
)
"#;

fn setup(size: u32, data: &[u8]) -> anyhow::Result<(Store<TvmHost>, i64)> {
    let host = TvmHost::new();
    let engine = Engine::new(Config::new().wasm_multi_memory(true))?;
    let mut store = Store::new(&engine, host);
    let region = store
        .data_mut()
        .create_region(RegionKind::HotHeap, size + 1024)?;
    let h = store.data_mut().alloc(region, size)?;
    store.data_mut().write_bytes(h, data)?;
    Ok((store, h.pack() as i64))
}

fn run(wat_src: &str, fn_name: &str, size: u32, data: &[u8])
    -> anyhow::Result<Vec<Duration>>
{
    let (mut store, packed) = setup(size, data)?;
    let module = Module::new(store.engine(), wat_src)?;
    let mut linker: Linker<TvmHost> = Linker::new(store.engine());
    add_raw_imports(&mut linker)?;
    let instance = linker.instantiate(&mut store, &module)?;
    let f = instance.get_typed_func::<(i64, i32), i64>(&mut store, fn_name)?;
    time_loop(|| { let _ = f.call(&mut store, (packed, size as i32))?; Ok(()) })
}

fn main() -> anyhow::Result<()> {
    println!("==> reducer-imports benchmark");
    println!("    {} samples + {} warmup", SAMPLES, WARMUP);
    println!();

    for &size in SIZES {
        let data: Vec<u8> = (0..size).map(|i| (i & 0xFF) as u8).collect();
        let mut classic = run(CLASSIC_WAT, "sum_via_read", size, &data)?;
        let mut reducer = run(REDUCER_WAT, "sum_via_reducer", size, &data)?;

        println!("--- size = {} bytes ---", size);
        report("classic (read+loop)", size, &mut classic);
        report("reducer (sum_u8)",    size, &mut reducer);

        let raw_c: Vec<u128> = classic.iter().map(|d| d.as_nanos()).collect();
        let raw_r: Vec<u128> = reducer.iter().map(|d| d.as_nanos()).collect();
        let mean = |v: &[u128]| v.iter().map(|n| *n as f64).sum::<f64>() / v.len() as f64;
        let speedup = mean(&raw_c) / mean(&raw_r);
        let u = mann_whitney_u(&raw_c, &raw_r);
        println!("    speedup reducer / classic = {:.2}x   U={:.3}", speedup, u);

        // Sanity: both should produce the same scalar result.
        let (mut s1, p1) = setup(size, &data)?;
        let m1 = Module::new(s1.engine(), CLASSIC_WAT)?;
        let mut l1: Linker<TvmHost> = Linker::new(s1.engine());
        add_raw_imports(&mut l1)?;
        let i1 = l1.instantiate(&mut s1, &m1)?;
        let f1 = i1.get_typed_func::<(i64, i32), i64>(&mut s1, "sum_via_read")?;
        let v1 = f1.call(&mut s1, (p1, size as i32))?;
        let (mut s2, p2) = setup(size, &data)?;
        let m2 = Module::new(s2.engine(), REDUCER_WAT)?;
        let mut l2: Linker<TvmHost> = Linker::new(s2.engine());
        add_raw_imports(&mut l2)?;
        let i2 = l2.instantiate(&mut s2, &m2)?;
        let f2 = i2.get_typed_func::<(i64, i32), i64>(&mut s2, "sum_via_reducer")?;
        let v2 = f2.call(&mut s2, (p2, size as i32))?;
        assert_eq!(v1, v2, "classic and reducer disagreed at size {}", size);
        println!();
    }

    Ok(())
}
