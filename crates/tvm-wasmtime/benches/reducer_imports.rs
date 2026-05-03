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

// classic count-byte: read into guest, then linear scan counting matches
const COUNT_CLASSIC_WAT: &str = r#"
(module
  (import "tvm" "read" (func $read (param i64 i32 i32) (result i32)))
  (memory (export "memory") 8)
  (func (export "count_via_read")
        (param $h i64) (param $len i32) (param $byte i32) (result i32)
    (local $cur i32) (local $end i32) (local $acc i32)
    (drop (call $read (local.get $h) (i32.const 0) (local.get $len)))
    (local.set $end (local.get $len))
    (block $break
      (loop $continue
        (br_if $break (i32.eq (local.get $cur) (local.get $end)))
        (if (i32.eq (i32.load8_u (local.get $cur)) (local.get $byte))
          (then (local.set $acc (i32.add (local.get $acc) (i32.const 1)))))
        (local.set $cur (i32.add (local.get $cur) (i32.const 1)))
        (br $continue)))
    (local.get $acc))
)
"#;

const COUNT_REDUCER_WAT: &str = r#"
(module
  (import "tvm" "count_byte" (func $count (param i64 i32 i32) (result i32)))
  (memory (export "memory") 1)
  (func (export "count_via_reducer")
        (param $h i64) (param $len i32) (param $byte i32) (result i32)
    (call $count (local.get $h) (local.get $len) (local.get $byte)))
)
"#;

// popcount: classic = read+per-byte popcnt loop
const POPCOUNT_CLASSIC_WAT: &str = r#"
(module
  (import "tvm" "read" (func $read (param i64 i32 i32) (result i32)))
  (memory (export "memory") 8)
  (func (export "popcount_via_read") (param $h i64) (param $len i32) (result i64)
    (local $cur i32) (local $end i32) (local $acc i64)
    (drop (call $read (local.get $h) (i32.const 0) (local.get $len)))
    (local.set $end (local.get $len))
    (block $break
      (loop $continue
        (br_if $break (i32.eq (local.get $cur) (local.get $end)))
        (local.set $acc
          (i64.add (local.get $acc)
                   (i64.extend_i32_u (i32.popcnt (i32.load8_u (local.get $cur))))))
        (local.set $cur (i32.add (local.get $cur) (i32.const 1)))
        (br $continue)))
    (local.get $acc))
)
"#;

const POPCOUNT_REDUCER_WAT: &str = r#"
(module
  (import "tvm" "popcount" (func $pc (param i64 i32) (result i64)))
  (memory (export "memory") 1)
  (func (export "popcount_via_reducer") (param $h i64) (param $len i32) (result i64)
    (call $pc (local.get $h) (local.get $len)))
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

fn run3(
    wat_src: &str,
    fn_name: &str,
    size: u32,
    data: &[u8],
    extra: i32,
) -> anyhow::Result<Vec<Duration>> {
    let (mut store, packed) = setup(size, data)?;
    let module = Module::new(store.engine(), wat_src)?;
    let mut linker: Linker<TvmHost> = Linker::new(store.engine());
    add_raw_imports(&mut linker)?;
    let instance = linker.instantiate(&mut store, &module)?;
    let f = instance.get_typed_func::<(i64, i32, i32), i32>(&mut store, fn_name)?;
    time_loop(|| { let _ = f.call(&mut store, (packed, size as i32, extra))?; Ok(()) })
}

fn pair_summary(label_c: &str, label_r: &str, classic: Vec<Duration>, reducer: Vec<Duration>, size: u32) {
    let mut classic = classic;
    let mut reducer = reducer;
    report(label_c, size, &mut classic);
    report(label_r, size, &mut reducer);
    let raw_c: Vec<u128> = classic.iter().map(|d| d.as_nanos()).collect();
    let raw_r: Vec<u128> = reducer.iter().map(|d| d.as_nanos()).collect();
    let mean = |v: &[u128]| v.iter().map(|n| *n as f64).sum::<f64>() / v.len() as f64;
    let speedup = mean(&raw_c) / mean(&raw_r);
    let u = mann_whitney_u(&raw_c, &raw_r);
    println!("    speedup reducer / classic = {:.2}x   U={:.3}", speedup, u);
}

fn main() -> anyhow::Result<()> {
    println!("==> reducer-imports benchmark");
    println!("    {} samples + {} warmup", SAMPLES, WARMUP);
    println!();

    for &size in SIZES {
        let data: Vec<u8> = (0..size).map(|i| (i & 0xFF) as u8).collect();

        println!("--- size = {} bytes ---", size);

        // sum_u8 — pure reduce, autovec friendly
        println!("  [sum_u8]");
        let c = run(CLASSIC_WAT, "sum_via_read", size, &data)?;
        let r = run(REDUCER_WAT, "sum_via_reducer", size, &data)?;
        pair_summary("    classic (read+loop)", "    reducer (sum_u8)", c, r, size);

        // count_byte — reduce with a predicate
        println!("  [count_byte]");
        let c = run3(COUNT_CLASSIC_WAT, "count_via_read", size, &data, 0x42)?;
        let r = run3(COUNT_REDUCER_WAT, "count_via_reducer", size, &data, 0x42)?;
        pair_summary("    classic (read+loop)", "    reducer (count_byte)", c, r, size);

        // popcount — bit-counting reduction
        println!("  [popcount]");
        let c = run(POPCOUNT_CLASSIC_WAT, "popcount_via_read", size, &data)?;
        let r = run(POPCOUNT_REDUCER_WAT, "popcount_via_reducer", size, &data)?;
        pair_summary("    classic (read+loop)", "    reducer (popcount)", c, r, size);

        println!();
    }

    Ok(())
}
