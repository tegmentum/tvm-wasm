//! Benchmark `tvm-guest-mm` against M32 baseline. Times a sequential-sum
//! workload over a 64KB region across both:
//!
//!   1. **Native M32**: standard `i32.load8_u (local.get $ptr+i)` against
//!      the default memory. Engine-emitted bounds-check.
//!   2. **TVM-MM Guest**: same workload but each byte goes through the
//!      generated `tvm_load_u8(pool=0, off)` dispatcher, which does an
//!      if/else chain on pool to select the memory immediate.
//!
//! Both variants live in self-contained wasm modules with no host
//! imports — the only difference is whether the access goes direct or
//! through the dispatcher.

use std::time::{Duration, Instant};
use tvm_guest_mm::{tvm_guest_mm_module_template, ModuleParams};
use tvm_test_harness::mann_whitney_u;
use wasmtime::{Config, Engine, Linker, Module, Store};

const SAMPLES: usize = 50;
const WARMUP: usize = 5;
const SIZES: &[u32] = &[1024, 16 * 1024, 65536];

// Re-implement percentile here since test_harness doesn't export it as a
// public function (it's used internally).
mod stats {
    use std::time::Duration;
    pub fn pct(sorted: &[Duration], p: f64) -> u128 {
        let n = sorted.len();
        if n == 0 {
            return 0;
        }
        let idx = ((p / 100.0) * (n as f64 - 1.0)).round() as usize;
        sorted[idx.min(n - 1)].as_nanos()
    }
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
    let mean_ns = t.iter().map(|d| d.as_nanos() as f64).sum::<f64>() / t.len() as f64;
    let p99 = stats::pct(t, 99.0);
    let bw = (size as f64) / (mean_ns / 1e9) / (1u64 << 30) as f64;
    println!(
        "  {:<20} size={:>6}  mean={:>8.0}ns  p99={:>8}ns  {:>5.2} GiB/s",
        label, size, mean_ns, p99, bw
    );
}

fn run_m32(size: u32, data: &[u8]) -> anyhow::Result<Vec<Duration>> {
    let m32_wat = r#"
        (module
          (memory (export "memory") 1)
          (func (export "buffer_ptr") (result i32) (i32.const 0))
          (func (export "sum_sequential") (param $ptr i32) (param $len i32) (result i64)
            (local $i i32) (local $acc i64)
            (block $break
              (loop $continue
                (br_if $break (i32.eq (local.get $i) (local.get $len)))
                (local.set $acc
                  (i64.add (local.get $acc)
                           (i64.load8_u (i32.add (local.get $ptr) (local.get $i)))))
                (local.set $i (i32.add (local.get $i) (i32.const 1)))
                (br $continue)))
            (local.get $acc)))
    "#;
    let engine = Engine::default();
    let module = Module::new(&engine, m32_wat)?;
    let linker: Linker<()> = Linker::new(&engine);
    let mut store = Store::new(&engine, ());
    let instance = linker.instantiate(&mut store, &module)?;
    let memory = instance.get_memory(&mut store, "memory").unwrap();
    memory.write(&mut store, 0, data)?;
    let sum = instance.get_typed_func::<(i32, i32), i64>(&mut store, "sum_sequential")?;
    time_loop(|| {
        let _ = sum.call(&mut store, (0, size as i32))?;
        Ok(())
    })
}

fn run_guest_mm_bulk(size: u32, data: &[u8], n_pools: u32) -> anyhow::Result<Vec<Duration>> {
    // Bulk-copy idiom: copy region into default memory in one dispatch
    // call, then sum natively. One pool dispatch per call regardless
    // of `len`. This is the workload pattern users should actually use.
    let user_body = r#"
        (func (export "sum_via_bulk") (param $src_pool i32) (param $src_off i32) (param $len i32) (result i64)
          (local $i i32) (local $acc i64)
          ;; One bulk dispatch: pool N → default memory at offset 0.
          (call $tvm_copy_to_default (local.get $src_pool) (local.get $src_off) (i32.const 0) (local.get $len))
          ;; Now sum natively from the default memory.
          (block $break
            (loop $continue
              (br_if $break (i32.eq (local.get $i) (local.get $len)))
              (local.set $acc
                (i64.add (local.get $acc)
                         (i64.load8_u (local.get $i))))
              (local.set $i (i32.add (local.get $i) (i32.const 1)))
              (br $continue)))
          (local.get $acc))
    "#;
    let p = ModuleParams {
        n_pools,
        initial_pages_per_pool: size.div_ceil(65536).max(1),
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

fn run_guest_mm(size: u32, data: &[u8], n_pools: u32) -> anyhow::Result<Vec<Duration>> {
    // The user body sums via dispatcher calls. Pool 1 is the data pool.
    let user_body = r#"
        (func (export "buffer_ptr") (result i32) (i32.const 0))
        (func (export "sum_via_dispatch") (param $ptr i32) (param $len i32) (result i64)
          (local $i i32) (local $acc i64)
          (block $break
            (loop $continue
              (br_if $break (i32.eq (local.get $i) (local.get $len)))
              (local.set $acc
                (i64.add (local.get $acc)
                         (i64.extend_i32_u
                           (call $tvm_load_u8
                             (i32.const 1)
                             (i32.add (local.get $ptr) (local.get $i))))))
              (local.set $i (i32.add (local.get $i) (i32.const 1)))
              (br $continue)))
          (local.get $acc))
    "#;
    let p = ModuleParams {
        n_pools,
        initial_pages_per_pool: size.div_ceil(65536).max(1),
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
    let sum = instance.get_typed_func::<(i32, i32), i64>(&mut store, "sum_via_dispatch")?;
    time_loop(|| {
        let _ = sum.call(&mut store, (0, size as i32))?;
        Ok(())
    })
}

fn main() -> anyhow::Result<()> {
    println!("==> tvm-guest-mm sequential-sum benchmark");
    println!("    {} samples + {} warmup", SAMPLES, WARMUP);
    println!();

    for &size in SIZES {
        let data: Vec<u8> = (0..size).map(|i| (i & 0xFF) as u8).collect();
        let mut m32_t = run_m32(size, &data)?;
        let mut bulk_t = run_guest_mm_bulk(size, &data, 16)?;
        let mut per_byte_t = run_guest_mm(size, &data, 16)?;

        report("m32 (native)", size, &mut m32_t);
        report("guest-mm bulk-copy", size, &mut bulk_t);
        report("guest-mm per-byte", size, &mut per_byte_t);

        let raw_m32: Vec<u128> = m32_t.iter().map(|d| d.as_nanos()).collect();
        let raw_bulk: Vec<u128> = bulk_t.iter().map(|d| d.as_nanos()).collect();
        let raw_per: Vec<u128> = per_byte_t.iter().map(|d| d.as_nanos()).collect();
        let u_bulk = mann_whitney_u(&raw_m32, &raw_bulk);
        let u_per = mann_whitney_u(&raw_m32, &raw_per);
        let mean = |v: &[u128]| v.iter().map(|n| *n as f64).sum::<f64>() / v.len() as f64;
        let speedup_bulk = mean(&raw_m32) / mean(&raw_bulk);
        let speedup_per = mean(&raw_m32) / mean(&raw_per);
        println!(
            "  -> bulk-copy idiom:    {:.2}x m32  U={:.3}  ({} pools)",
            speedup_bulk, u_bulk, 16
        );
        println!(
            "  -> per-byte dispatch:  {:.2}x m32  U={:.3}  ({} pools)",
            speedup_per, u_per, 16
        );
        println!();
    }

    // Dispatch-scaling: hold size constant, vary pool count. With the
    // BST emitter the per-byte path should grow ~log(N), not linearly.
    println!("==> per-byte dispatch scaling vs pool count (size = 16 KiB)");
    let size: u32 = 16 * 1024;
    let data: Vec<u8> = (0..size).map(|i| (i & 0xFF) as u8).collect();
    // Wasmtime enforces a 100-memory limit on modules by default, so we
    // cap the sweep at 64 pools. The scaling pattern is already clear by
    // that point — log(N) growth, not linear.
    for &n_pools in &[2u32, 4, 8, 16, 32, 64] {
        let mut t = run_guest_mm(size, &data, n_pools)?;
        report(&format!("per-byte n_pools={}", n_pools), size, &mut t);
    }

    // BST vs call_indirect dispatch — same per-byte sum loop, two
    // different dispatchers. Settles whether the indirect-table form
    // can beat ~log₂(N) compares on wasmtime.
    println!();
    println!("==> dispatch shape comparison (size = 16 KiB, n_pools = 64)");
    let mut bst = run_dispatch_named(size, &data, 64, "tvm_load_u8")?;
    let mut indirect = run_dispatch_named(size, &data, 64, "tvm_load_u8_indirect")?;
    report("BST (tvm_load_u8)  ", size, &mut bst);
    report("call_indirect      ", size, &mut indirect);
    let mean_d =
        |v: &[Duration]| v.iter().map(|d| d.as_nanos() as f64).sum::<f64>() / v.len() as f64;
    let speedup = mean_d(&bst) / mean_d(&indirect);
    let raw_b: Vec<u128> = bst.iter().map(|d| d.as_nanos()).collect();
    let raw_i: Vec<u128> = indirect.iter().map(|d| d.as_nanos()).collect();
    let u = mann_whitney_u(&raw_b, &raw_i);
    println!("    indirect / BST = {:.2}x   U={:.3}", speedup, u);

    Ok(())
}

/// Per-byte sum over a fixed buffer, calling the named dispatch
/// function (either the BST `tvm_load_u8` or the call_indirect
/// `tvm_load_u8_indirect`).
fn run_dispatch_named(
    size: u32,
    data: &[u8],
    n_pools: u32,
    dispatch_fn_name: &str,
) -> anyhow::Result<Vec<Duration>> {
    let user_body = format!(
        r#"
        (func (export "buffer_ptr") (result i32) (i32.const 0))
        (func (export "sum_via_dispatch") (param $ptr i32) (param $len i32) (result i64)
          (local $i i32) (local $acc i64)
          (block $break
            (loop $continue
              (br_if $break (i32.eq (local.get $i) (local.get $len)))
              (local.set $acc
                (i64.add (local.get $acc)
                         (i64.extend_i32_u
                           (call ${disp}
                             (i32.const 1)
                             (i32.add (local.get $ptr) (local.get $i))))))
              (local.set $i (i32.add (local.get $i) (i32.const 1)))
              (br $continue)))
          (local.get $acc))
    "#,
        disp = dispatch_fn_name
    );
    let p = ModuleParams {
        n_pools,
        initial_pages_per_pool: size.div_ceil(65536).max(1),
        max_pages_per_pool: 16,
        user_body,
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
    let sum = instance.get_typed_func::<(i32, i32), i64>(&mut store, "sum_via_dispatch")?;
    time_loop(|| {
        let _ = sum.call(&mut store, (0, size as i32))?;
        Ok(())
    })
}
