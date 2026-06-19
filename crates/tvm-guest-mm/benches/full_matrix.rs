//! Full-matrix bench: guest-mm bulk-copy vs all the other variants
//! we've measured. Same workload (sequential sum) at three sizes.
//!
//! Variants:
//!   - M32 native (default memory)
//!   - M64 native (64-bit memory)
//!   - host-side TVM raw fast path (host memcpy + guest sum)
//!   - host-side TVM-MM (imported memory, native loads)
//!   - guest-side TVM-MM bulk-copy (memory.copy + native sum)
//!
//! All use a 50-sample run with 5 warmups. Reports mean / p99 / GiB/s
//! and Mann-Whitney U + speedup vs M32 baseline.

use std::time::{Duration, Instant};
use tvm_guest_mm::{tvm_guest_mm_module_template, ModuleParams};
use tvm_test_harness::mann_whitney_u;
use tvm_wasmtime::{add_raw_imports, build_imported_setup, TvmHost};
use wasmtime::{Config, Engine, Linker, Memory, MemoryType, Module, Store};

const SAMPLES: usize = 50;
const WARMUP: usize = 5;
const SIZES: &[u32] = &[1024, 16 * 1024, 65536, 262_144];

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

struct Result {
    label: &'static str,
    mean_ns: f64,
    p99_ns: u128,
    gib_per_s: f64,
    raw: Vec<u128>,
}

fn summarize(label: &'static str, mut t: Vec<Duration>, size: u32) -> Result {
    t.sort();
    let raw: Vec<u128> = t.iter().map(|d| d.as_nanos()).collect();
    let mean_ns = raw.iter().map(|n| *n as f64).sum::<f64>() / raw.len() as f64;
    let gib = if mean_ns > 0.0 {
        (size as f64) / (mean_ns / 1e9) / (1u64 << 30) as f64
    } else {
        0.0
    };
    Result {
        label,
        mean_ns,
        p99_ns: pct(&t, 99.0),
        gib_per_s: gib,
        raw,
    }
}

// ---------- M32 ----------

const M32_WAT: &str = r#"
(module
  (memory (export "memory") 8)
  (func (export "buffer_ptr") (result i32) (i32.const 0))
  (func (export "sum") (param $ptr i32) (param $len i32) (result i64)
    (local $cur i32) (local $end i32) (local $acc i64)
    (local.set $cur (local.get $ptr))
    (local.set $end (i32.add (local.get $ptr) (local.get $len)))
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

fn run_m32(size: u32, data: &[u8]) -> anyhow::Result<Vec<Duration>> {
    let engine = Engine::default();
    let module = Module::new(&engine, M32_WAT)?;
    let linker: Linker<()> = Linker::new(&engine);
    let mut store = Store::new(&engine, ());
    let instance = linker.instantiate(&mut store, &module)?;
    let memory = instance.get_memory(&mut store, "memory").unwrap();
    memory.write(&mut store, 0, data)?;
    let sum = instance.get_typed_func::<(i32, i32), i64>(&mut store, "sum")?;
    time_loop(|| {
        let _ = sum.call(&mut store, (0, size as i32))?;
        Ok(())
    })
}

// ---------- M64 ----------

const M64_WAT: &str = r#"
(module
  (memory (export "memory") i64 8)
  (func (export "sum") (param $ptr i64) (param $len i64) (result i64)
    (local $cur i64) (local $end i64) (local $acc i64)
    (local.set $cur (local.get $ptr))
    (local.set $end (i64.add (local.get $ptr) (local.get $len)))
    (block $break
      (loop $continue
        (br_if $break (i64.eq (local.get $cur) (local.get $end)))
        (local.set $acc
          (i64.add (local.get $acc)
                   (i64.load8_u (local.get $cur))))
        (local.set $cur (i64.add (local.get $cur) (i64.const 1)))
        (br $continue)))
    (local.get $acc))
)
"#;

fn run_m64(size: u32, data: &[u8]) -> anyhow::Result<Vec<Duration>> {
    let mut config = Config::new();
    config.wasm_memory64(true);
    let engine = Engine::new(&config)?;
    let module = Module::new(&engine, M64_WAT)?;
    let linker: Linker<()> = Linker::new(&engine);
    let mut store = Store::new(&engine, ());
    let instance = linker.instantiate(&mut store, &module)?;
    let memory = instance.get_memory(&mut store, "memory").unwrap();
    memory.write(&mut store, 0, data)?;
    let sum = instance.get_typed_func::<(i64, i64), i64>(&mut store, "sum")?;
    time_loop(|| {
        let _ = sum.call(&mut store, (0, size as i64))?;
        Ok(())
    })
}

// ---------- Host TVM raw fast path ----------

const HOST_RAW_WAT: &str = r#"
(module
  (import "tvm" "alloc" (func $alloc (param i32 i32) (result i64)))
  (import "tvm" "read"  (func $read  (param i64 i32 i32) (result i32)))
  (memory (export "memory") 8)
  (func (export "alloc") (param $r i32) (param $s i32) (result i64)
    (call $alloc (local.get $r) (local.get $s)))
  (func (export "sum_via_read")
        (param $h i64) (param $len i32) (result i64)
    (local $cur i32) (local $end i32) (local $acc i64)
    ;; pull region bytes into default memory at offset 0 in one host call
    (drop (call $read (local.get $h) (i32.const 0) (local.get $len)))
    ;; sum natively
    (local.set $cur (i32.const 0))
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

fn run_host_raw(size: u32, data: &[u8]) -> anyhow::Result<Vec<Duration>> {
    use tvm_wasmtime::bindings::tvm::memory::manager::Host as ManagerHost;
    use tvm_wasmtime::bindings::tvm::memory::types::RegionKind as WitKind;
    let engine = Engine::default();
    let module = Module::new(&engine, HOST_RAW_WAT)?;
    let mut host = TvmHost::new();
    let region = ManagerHost::create_region(&mut host, WitKind::HotHeap, size + 4096)?;
    let mut linker: Linker<TvmHost> = Linker::new(&engine);
    add_raw_imports(&mut linker)?;
    let mut store = Store::new(&engine, host);
    let instance = linker.instantiate(&mut store, &module)?;
    let alloc = instance.get_typed_func::<(i32, i32), i64>(&mut store, "alloc")?;
    let sum = instance.get_typed_func::<(i64, i32), i64>(&mut store, "sum_via_read")?;
    let h = alloc.call(&mut store, (region as i32, size as i32))?;

    // Seed the region's bytes via host.write (the tvm_wasmtime BytesHost
    // path), reusing the host's directory.
    use tvm_core::Handle;
    let core_h = Handle::unpack(h as u64);
    store.data_mut().write_bytes(core_h, data)?;

    time_loop(|| {
        let _ = sum.call(&mut store, (h, size as i32))?;
        Ok(())
    })
}

// ---------- Host TVM-MM (imported memory) ----------

const HOST_MM_WAT: &str = r#"
(module
  (import "tvm" "r0" (memory $r 1))
  (func (export "sum_in_r0") (param $ptr i32) (param $len i32) (result i64)
    (local $cur i32) (local $end i32) (local $acc i64)
    (local.set $cur (local.get $ptr))
    (local.set $end (i32.add (local.get $ptr) (local.get $len)))
    (block $break
      (loop $continue
        (br_if $break (i32.eq (local.get $cur) (local.get $end)))
        (local.set $acc
          (i64.add (local.get $acc)
                   (i64.load8_u $r (local.get $cur))))
        (local.set $cur (i32.add (local.get $cur) (i32.const 1)))
        (br $continue)))
    (local.get $acc))
)
"#;

fn run_host_mm(size: u32, data: &[u8]) -> anyhow::Result<Vec<Duration>> {
    let pages = (size as u64).div_ceil(65536).max(1) as u32;
    let mut config = Config::new();
    config.wasm_multi_memory(true);
    let engine = Engine::new(&config)?;
    let module = Module::new(&engine, HOST_MM_WAT)?;
    let mut store: Store<()> = Store::new(&engine, ());
    let mem0 = Memory::new(&mut store, MemoryType::new(pages, None))?;
    let mut linker: Linker<()> = Linker::new(&engine);
    linker.define(&mut store, "tvm", "r0", mem0)?;
    let instance = linker.instantiate(&mut store, &module)?;
    mem0.write(&mut store, 0, data)?;
    let sum = instance.get_typed_func::<(i32, i32), i64>(&mut store, "sum_in_r0")?;
    time_loop(|| {
        let _ = sum.call(&mut store, (0, size as i32))?;
        Ok(())
    })
}

// ---------- Guest-mm bulk-copy (generic dispatch) ----------

fn run_guest_mm_bulk(size: u32, data: &[u8]) -> anyhow::Result<Vec<Duration>> {
    let user_body = r#"
        (func (export "sum_via_bulk")
              (param $src_pool i32) (param $src_off i32) (param $len i32) (result i64)
          (local $cur i32) (local $end i32) (local $acc i64)
          (call $tvm_copy_to_default (local.get $src_pool) (local.get $src_off) (i32.const 0) (local.get $len))
          (local.set $end (local.get $len))
          (block $break
            (loop $continue
              (br_if $break (i32.eq (local.get $cur) (local.get $end)))
              (local.set $acc
                (i64.add (local.get $acc) (i64.load8_u (local.get $cur))))
              (local.set $cur (i32.add (local.get $cur) (i32.const 1)))
              (br $continue)))
          (local.get $acc))
    "#;
    let p = ModuleParams {
        n_pools: 4,
        initial_pages_per_pool: (size as u64).div_ceil(65536).max(1) as u32,
        max_pages_per_pool: 16,
        user_body: user_body.to_string(),
    };
    let wat = tvm_guest_mm_module_template(&p);
    let mut config = Config::new();
    config.wasm_multi_memory(true);
    let engine = Engine::new(&config)?;
    let module = Module::new(&engine, &wat)?;
    let linker: Linker<()> = Linker::new(&engine);
    let mut store = Store::new(&engine, ());
    let instance = linker.instantiate(&mut store, &module)?;
    let mem1 = instance.get_memory(&mut store, "mem1").unwrap();
    mem1.write(&mut store, 0, data)?;
    let sum = instance.get_typed_func::<(i32, i32, i32), i64>(&mut store, "sum_via_bulk")?;
    time_loop(|| {
        let _ = sum.call(&mut store, (1, 0, size as i32))?;
        Ok(())
    })
}

// ---------- Guest-mm bulk-copy via specialized per-pool helper ----------
//
// Same as `run_guest_mm_bulk` but the user calls
// `tvm_copy_to_default_p1` directly — a single static `memory.copy 0 1`
// with no dispatch. Tests whether the dispatch overhead measurable.

fn run_guest_mm_bulk_specialized(size: u32, data: &[u8]) -> anyhow::Result<Vec<Duration>> {
    let user_body = r#"
        (func (export "sum_via_bulk_specialized")
              (param $src_off i32) (param $len i32) (result i64)
          (local $cur i32) (local $end i32) (local $acc i64)
          (call $tvm_copy_to_default_p1 (local.get $src_off) (i32.const 0) (local.get $len))
          (local.set $end (local.get $len))
          (block $break
            (loop $continue
              (br_if $break (i32.eq (local.get $cur) (local.get $end)))
              (local.set $acc
                (i64.add (local.get $acc) (i64.load8_u (local.get $cur))))
              (local.set $cur (i32.add (local.get $cur) (i32.const 1)))
              (br $continue)))
          (local.get $acc))
    "#;
    let p = ModuleParams {
        n_pools: 4,
        initial_pages_per_pool: (size as u64).div_ceil(65536).max(1) as u32,
        max_pages_per_pool: 16,
        user_body: user_body.to_string(),
    };
    let wat = tvm_guest_mm_module_template(&p);
    let mut config = Config::new();
    config.wasm_multi_memory(true);
    let engine = Engine::new(&config)?;
    let module = Module::new(&engine, &wat)?;
    let linker: Linker<()> = Linker::new(&engine);
    let mut store = Store::new(&engine, ());
    let instance = linker.instantiate(&mut store, &module)?;
    let mem1 = instance.get_memory(&mut store, "mem1").unwrap();
    mem1.write(&mut store, 0, data)?;
    let sum = instance.get_typed_func::<(i32, i32), i64>(&mut store, "sum_via_bulk_specialized")?;
    time_loop(|| {
        let _ = sum.call(&mut store, (0, size as i32))?;
        Ok(())
    })
}

// ---------- Guest-mm SIMD popcount (symmetric reducer family) ----------

fn run_guest_mm_popcount_simd(size: u32, data: &[u8]) -> anyhow::Result<Vec<Duration>> {
    let user_body = r#"
        (func (export "popcount_via_simd") (param $off i32) (param $len i32) (result i64)
          (call $tvm_simd_popcount_p1 (local.get $off) (local.get $len)))
    "#;
    let p = ModuleParams {
        n_pools: 4,
        initial_pages_per_pool: (size as u64).div_ceil(65536).max(1) as u32,
        max_pages_per_pool: 16,
        user_body: user_body.to_string(),
    };
    let wat = tvm_guest_mm_module_template(&p);
    let mut config = Config::new();
    config.wasm_multi_memory(true);
    let engine = Engine::new(&config)?;
    let module = Module::new(&engine, &wat)?;
    let linker: Linker<()> = Linker::new(&engine);
    let mut store = Store::new(&engine, ());
    let instance = linker.instantiate(&mut store, &module)?;
    let mem1 = instance.get_memory(&mut store, "mem1").unwrap();
    mem1.write(&mut store, 0, data)?;
    let f = instance.get_typed_func::<(i32, i32), i64>(&mut store, "popcount_via_simd")?;
    time_loop(|| {
        let _ = f.call(&mut store, (0, size as i32))?;
        Ok(())
    })
}

fn run_guest_mm_popcount_scalar(size: u32, data: &[u8]) -> anyhow::Result<Vec<Duration>> {
    let user_body = r#"
        (func (export "popcount_via_scalar") (param $off i32) (param $len i32) (result i64)
          (local $cur i32) (local $end i32) (local $acc i64)
          (local.set $cur (local.get $off))
          (local.set $end (i32.add (local.get $off) (local.get $len)))
          (block $break
            (loop $continue
              (br_if $break (i32.eq (local.get $cur) (local.get $end)))
              (local.set $acc
                (i64.add (local.get $acc)
                  (i64.extend_i32_u
                    (i32.popcnt (call $tvm_load_u8_p1 (local.get $cur))))))
              (local.set $cur (i32.add (local.get $cur) (i32.const 1)))
              (br $continue)))
          (local.get $acc))
    "#;
    let p = ModuleParams {
        n_pools: 4,
        initial_pages_per_pool: (size as u64).div_ceil(65536).max(1) as u32,
        max_pages_per_pool: 16,
        user_body: user_body.to_string(),
    };
    let wat = tvm_guest_mm_module_template(&p);
    let mut config = Config::new();
    config.wasm_multi_memory(true);
    let engine = Engine::new(&config)?;
    let module = Module::new(&engine, &wat)?;
    let linker: Linker<()> = Linker::new(&engine);
    let mut store = Store::new(&engine, ());
    let instance = linker.instantiate(&mut store, &module)?;
    let mem1 = instance.get_memory(&mut store, "mem1").unwrap();
    mem1.write(&mut store, 0, data)?;
    let f = instance.get_typed_func::<(i32, i32), i64>(&mut store, "popcount_via_scalar")?;
    time_loop(|| {
        let _ = f.call(&mut store, (0, size as i32))?;
        Ok(())
    })
}

// ---------- Guest-mm SIMD sum kernel ----------
//
// Calls the SIMD-specialized `tvm_simd_sum_u8_p1` directly — sums
// pool 1's bytes with no dispatch and no copy-then-loop. Pure
// SIMD-in-place.

fn run_guest_mm_simd(size: u32, data: &[u8]) -> anyhow::Result<Vec<Duration>> {
    let user_body = r#"
        (func (export "sum_via_simd") (param $off i32) (param $len i32) (result i64)
          (call $tvm_simd_sum_u8_p1 (local.get $off) (local.get $len)))
    "#;
    let p = ModuleParams {
        n_pools: 4,
        initial_pages_per_pool: (size as u64).div_ceil(65536).max(1) as u32,
        max_pages_per_pool: 16,
        user_body: user_body.to_string(),
    };
    let wat = tvm_guest_mm_module_template(&p);
    let mut config = Config::new();
    config.wasm_multi_memory(true);
    let engine = Engine::new(&config)?;
    let module = Module::new(&engine, &wat)?;
    let linker: Linker<()> = Linker::new(&engine);
    let mut store = Store::new(&engine, ());
    let instance = linker.instantiate(&mut store, &module)?;
    let mem1 = instance.get_memory(&mut store, "mem1").unwrap();
    mem1.write(&mut store, 0, data)?;
    let sum = instance.get_typed_func::<(i32, i32), i64>(&mut store, "sum_via_simd")?;
    time_loop(|| {
        let _ = sum.call(&mut store, (0, size as i32))?;
        Ok(())
    })
}

fn main() -> anyhow::Result<()> {
    println!("==> tvm-guest-mm full-matrix benchmark");
    println!("    {} samples + {} warmup", SAMPLES, WARMUP);
    println!();

    let _ = build_imported_setup; // keep import live

    for &size in SIZES {
        let data: Vec<u8> = (0..size).map(|i| (i & 0xFF) as u8).collect();

        let m32 = summarize("m32 native", run_m32(size, &data)?, size);
        let m64 = summarize("m64 native", run_m64(size, &data)?, size);
        let raw = summarize("host TVM raw", run_host_raw(size, &data)?, size);
        let mm = summarize("host TVM-MM", run_host_mm(size, &data)?, size);
        let gmm = summarize("guest-mm bulk", run_guest_mm_bulk(size, &data)?, size);
        let gms = summarize(
            "guest-mm spec",
            run_guest_mm_bulk_specialized(size, &data)?,
            size,
        );
        let sim = summarize("guest-mm simd", run_guest_mm_simd(size, &data)?, size);

        println!("--- size = {} bytes ---", size);
        for r in &[&m32, &m64, &raw, &mm, &gmm, &gms, &sim] {
            println!(
                "  {:<16} mean={:>10.0}ns  p99={:>10}ns  {:>5.2} GiB/s",
                r.label, r.mean_ns, r.p99_ns, r.gib_per_s
            );
        }

        // Compare each non-M32 variant to the M32 baseline.
        let baseline = m32.mean_ns;
        for r in &[&m64, &raw, &mm, &gmm, &gms, &sim] {
            let speedup = baseline / r.mean_ns;
            let u = mann_whitney_u(&m32.raw, &r.raw);
            println!("    {:<16} {:>5.2}x m32   U={:.3}", r.label, speedup, u);
        }
        println!();
    }

    // Popcount: scalar dispatch loop vs SIMD reducer kernel.
    println!("==> popcount: scalar dispatch vs SIMD reducer");
    for &size in SIZES {
        let data: Vec<u8> = (0..size).map(|i| (i & 0xFF) as u8).collect();
        let scalar = summarize(
            "scalar (per-byte)",
            run_guest_mm_popcount_scalar(size, &data)?,
            size,
        );
        let simd = summarize(
            "simd (popcount)",
            run_guest_mm_popcount_simd(size, &data)?,
            size,
        );
        println!("--- size = {} bytes ---", size);
        for r in &[&scalar, &simd] {
            println!(
                "  {:<20} mean={:>10.0}ns  p99={:>10}ns  {:>5.2} GiB/s",
                r.label, r.mean_ns, r.p99_ns, r.gib_per_s
            );
        }
        let speedup = scalar.mean_ns / simd.mean_ns;
        let u = mann_whitney_u(&scalar.raw, &simd.raw);
        println!("    speedup simd / scalar = {:.2}x   U={:.3}", speedup, u);
        println!();
    }

    Ok(())
}
