//! tvm-bench runner. Builds the M32 and TVM workload modules (auto-build),
//! exercises all benchmark classes, emits JSON results with Mann-Whitney U
//! significance tests against baseline.
//!
//! Wasmtime-only. Wasmer + V8 backends are tracked in BACKLOG.md.

use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

use anyhow::Context;
use serde::Serialize;
use tvm_wasmtime::bindings::tvm::memory::manager::Host as ManagerHost;
use tvm_wasmtime::bindings::tvm::memory::types::RegionKind;
use tvm_wasmtime::{add_raw_imports, TvmHost};
use wasmtime::{Config, Engine, Linker, Module, Store};

const SIZES: &[u32] = &[1024, 16 * 1024, 256 * 1024];
const WARMUP_ROUNDS: usize = 5;
const SAMPLES: usize = 50;
const SEED: u32 = 0xDEADBEEF;

#[derive(Serialize, Clone)]
struct Sample {
    variant: String,
    class: String,
    size_bytes: u32,
    runtime: &'static str,
    samples: usize,
    warmup: usize,
    seed: u32,
    mean_ns: f64,
    p50_ns: u128,
    p95_ns: u128,
    p99_ns: u128,
    throughput_gib_per_s: f64,
    notes: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    raw_ns: Vec<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mann_whitney_u_vs_baseline: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    speedup_vs_baseline: Option<f64>,
    /// Speedup vs M64 specifically — the "what we beat" metric. Computed
    /// for tvm and tvm-mm rows when an M64 sample exists for the same
    /// (class, size).
    #[serde(skip_serializing_if = "Option::is_none")]
    speedup_vs_m64: Option<f64>,
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent() // bench-framework
        .unwrap()
        .parent() // workspace root
        .unwrap()
        .to_path_buf()
}

fn fill_pattern(buf: &mut [u8], seed: u32) {
    let mut state = seed.wrapping_add(1);
    for byte in buf.iter_mut() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        *byte = state as u8;
    }
}

fn percentile(sorted: &[Duration], p: f64) -> u128 {
    let n = sorted.len();
    let idx = ((p / 100.0) * (n as f64 - 1.0)).round() as usize;
    sorted[idx.min(n - 1)].as_nanos()
}

// Mann-Whitney U is provided by `tvm-test-harness`; we re-export here so
// the rest of the runner doesn't have to know which crate it lives in.
use tvm_test_harness::mann_whitney_u;

fn summarize(variant: &str, class: &str, size: u32, timings: Vec<Duration>, notes: &str) -> Sample {
    let mut sorted = timings.clone();
    sorted.sort();
    let mean_ns = timings.iter().map(|d| d.as_nanos() as f64).sum::<f64>() / timings.len() as f64;
    let throughput_gib_per_s = if mean_ns > 0.0 {
        (size as f64) / (mean_ns / 1e9) / (1u64 << 30) as f64
    } else {
        0.0
    };
    Sample {
        variant: variant.to_string(),
        class: class.to_string(),
        size_bytes: size,
        runtime: "wasmtime",
        samples: timings.len(),
        warmup: WARMUP_ROUNDS,
        seed: SEED,
        mean_ns,
        p50_ns: percentile(&sorted, 50.0),
        p95_ns: percentile(&sorted, 95.0),
        p99_ns: percentile(&sorted, 99.0),
        throughput_gib_per_s,
        notes: notes.to_string(),
        raw_ns: timings.iter().map(|d| d.as_nanos()).collect(),
        mann_whitney_u_vs_baseline: None,
        speedup_vs_baseline: None,
        speedup_vs_m64: None,
    }
}

fn time_loop<F: FnMut() -> anyhow::Result<()>>(mut f: F) -> anyhow::Result<Vec<Duration>> {
    for _ in 0..WARMUP_ROUNDS {
        f()?;
    }
    let mut timings = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = Instant::now();
        f()?;
        timings.push(start.elapsed());
    }
    Ok(timings)
}

// -------------------- M32 setup --------------------

struct M32Ctx {
    store: Store<()>,
    instance: wasmtime::Instance,
    buffer_ptr: u32,
}

fn new_m32(wasm: &[u8]) -> anyhow::Result<M32Ctx> {
    let engine = Engine::default();
    let module = Module::new(&engine, wasm)?;
    let linker: Linker<()> = Linker::new(&engine);
    let mut store = Store::new(&engine, ());
    let instance = linker.instantiate(&mut store, &module)?;
    let buffer_ptr = instance
        .get_typed_func::<(), u32>(&mut store, "buffer_ptr")?
        .call(&mut store, ())?;
    Ok(M32Ctx {
        store,
        instance,
        buffer_ptr,
    })
}

// -------------------- M64 setup --------------------

struct M64Ctx {
    store: Store<()>,
    instance: wasmtime::Instance,
    buffer_ptr: u64,
}

fn m64_engine() -> anyhow::Result<Engine> {
    let mut config = Config::new();
    config.wasm_memory64(true);
    Engine::new(&config).map_err(Into::into)
}

fn new_m64(wasm: &[u8]) -> anyhow::Result<M64Ctx> {
    let engine = m64_engine()?;
    let module = Module::new(&engine, wasm)?;
    let linker: Linker<()> = Linker::new(&engine);
    let mut store = Store::new(&engine, ());
    let instance = linker.instantiate(&mut store, &module)?;
    let buffer_ptr = instance
        .get_typed_func::<(), u64>(&mut store, "buffer_ptr")?
        .call(&mut store, ())?;
    Ok(M64Ctx {
        store,
        instance,
        buffer_ptr,
    })
}

// -------------------- TVM setup --------------------

struct TvmCtx {
    store: Store<TvmHost>,
    instance: wasmtime::Instance,
    region: u16,
}

fn new_tvm(wasm: &[u8], region_capacity: u32) -> anyhow::Result<TvmCtx> {
    let engine = Engine::default();
    let module = Module::new(&engine, wasm)?;
    let mut host = TvmHost::new();
    let region = ManagerHost::create_region(&mut host, RegionKind::HotHeap, region_capacity)?;
    let mut linker: Linker<TvmHost> = Linker::new(&engine);
    add_raw_imports(&mut linker)?;
    let mut store = Store::new(&engine, host);
    let instance = linker.instantiate(&mut store, &module)?;
    Ok(TvmCtx {
        store,
        instance,
        region,
    })
}

// -------------------- TVM-Unified (imported memory + tvm.alloc) --------------------
//
// The full architectural play: the region's bytes live in a wasmtime
// memory exposed as an import. The guest accesses it natively (i32.load).
// alloc/dealloc/pin/spill go through TvmHost. This is what production
// usage should look like — it gives M32-level access cost AND TVM
// lifecycle.

const UNIFIED_WAT: &str = r#"
(module
  (import "tvm" "r0" (memory $r 1))
  (func (export "sum_in_r") (param $ptr i32) (param $len i32) (result i64)
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

fn run_tvm_unified_sequential(size: u32, data: &[u8]) -> anyhow::Result<Sample> {
    // One-call setup using the new helper.
    let (engine, mut store, linker, ids) =
        tvm_wasmtime::build_imported_setup(1, size + 4096, tvm_core::RegionKind::HotHeap)?;
    let region_id = ids[0];
    let handle = store.data_mut().imported_alloc(region_id, size)?;

    // Write payload into the region's wasm memory.
    let memory = store.data().imported_region(region_id).unwrap().memory();
    memory.write(&mut store, handle.offset as usize, data)?;

    let module = Module::new(&engine, UNIFIED_WAT)?;
    let instance = linker.instantiate(&mut store, &module)?;
    let sum = instance.get_typed_func::<(i32, i32), i64>(&mut store, "sum_in_r")?;
    let timings = time_loop(|| {
        let _ = sum.call(&mut store, (handle.offset as i32, size as i32))?;
        Ok(())
    })?;

    Ok(summarize(
        "tvm-unified",
        "sequential_sum",
        size,
        timings,
        "imported memory + tvm.alloc; native guest access",
    ))
}

// -------------------- TVM-MM (multi-memory imports) --------------------
//
// The TVM-MM variant: each region is exposed to the guest as an *imported*
// wasm memory. The guest reads/writes via native i32.load / i32.store
// instructions targeting that memory — no host call, no scratch copy.
//
// Working sets ≤ 4 GiB fit in a single imported memory and pay **zero**
// per-access cost relative to M32. The only cost when crossing a 4 GiB
// boundary is a wasm-instruction-level memory switch, not a host call.
//
// The workload below is hand-written WAT because Rust's wasm32 target
// doesn't support multi-memory imports without nightly + WAT shims.

const MM_WAT: &str = r#"
(module
  (import "tvm" "r0" (memory $r0 1))
  (import "tvm" "r1" (memory $r1 1))

  ;; Pointer-end loop instead of index-counter. Mirrors what rustc emits
  ;; for the M32 case — eliminates the per-iteration `i + ptr` add and
  ;; gives the engine a clearer monotonic-bounds pattern for elision.
  (func (export "sum_in_r0") (param $ptr i32) (param $len i32) (result i64)
    (local $cur i32) (local $end i32) (local $acc i64)
    (local.set $cur (local.get $ptr))
    (local.set $end (i32.add (local.get $ptr) (local.get $len)))
    (block $break
      (loop $continue
        (br_if $break (i32.eq (local.get $cur) (local.get $end)))
        (local.set $acc
          (i64.add (local.get $acc)
                   (i64.load8_u $r0 (local.get $cur))))
        (local.set $cur (i32.add (local.get $cur) (i32.const 1)))
        (br $continue)))
    (local.get $acc))

  (func (export "list_walk_r0") (param $ptr i32) (param $head i32) (result i64)
    (local $cur i32) (local $acc i64) (local $node i32) (local $next i32) (local $payload i32)
    (local.set $cur (local.get $head))
    (block $break
      (loop $continue
        (br_if $break (i32.eq (local.get $cur) (i32.const 0xFFFFFFFF)))
        (local.set $node (i32.add (local.get $ptr) (local.get $cur)))
        (local.set $next    (i32.load $r0 (local.get $node)))
        (local.set $payload (i32.load $r0 (i32.add (local.get $node) (i32.const 4))))
        (local.set $acc (i64.add (local.get $acc) (i64.extend_i32_u (local.get $payload))))
        (local.set $cur (local.get $next))
        (br $continue)))
    (local.get $acc))

  (func (export "columnar_filter_sum")
        (param $a_ptr i32) (param $b_ptr i32) (param $n i32) (param $threshold i32)
        (result i64)
    (local $i i32) (local $acc i64) (local $k i32) (local $v i32)
    (block $break
      (loop $continue
        (br_if $break (i32.eq (local.get $i) (local.get $n)))
        (local.set $k (i32.load $r0
          (i32.add (local.get $a_ptr) (i32.shl (local.get $i) (i32.const 2)))))
        (if (i32.lt_u (local.get $k) (local.get $threshold))
          (then
            (local.set $v (i32.load $r1
              (i32.add (local.get $b_ptr) (i32.shl (local.get $i) (i32.const 2)))))
            (local.set $acc (i64.add (local.get $acc) (i64.extend_i32_u (local.get $v))))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $continue)))
    (local.get $acc))
)
"#;

fn enable_multi_memory() -> anyhow::Result<Engine> {
    Engine::new(&tvm_wasmtime::imported_region_engine_config()).map_err(Into::into)
}

fn run_tvm_mm_sequential(size: u32, data: &[u8]) -> anyhow::Result<Sample> {
    use wasmtime::{Memory, MemoryType};
    let engine = enable_multi_memory()?;
    let module = Module::new(&engine, MM_WAT)?;
    // Pre-create two memories sized to comfortably hold our test data.
    let pages_needed = ((size as u64 + 65535) / 65536).max(1) as u32;
    let mut store = Store::new(&engine, ());
    let mem0 = Memory::new(
        &mut store,
        MemoryType::new(pages_needed, Some(pages_needed)),
    )?;
    let mem1 = Memory::new(&mut store, MemoryType::new(1, Some(1)))?;
    let mut linker: Linker<()> = Linker::new(&engine);
    linker.define(&mut store, "tvm", "r0", mem0)?;
    linker.define(&mut store, "tvm", "r1", mem1)?;
    let instance = linker.instantiate(&mut store, &module)?;
    mem0.write(&mut store, 0, data)?;
    let sum = instance.get_typed_func::<(i32, i32), i64>(&mut store, "sum_in_r0")?;
    let timings = time_loop(|| {
        let _ = sum.call(&mut store, (0, size as i32))?;
        Ok(())
    })?;
    Ok(summarize(
        "tvm-mm",
        "sequential_sum",
        size,
        timings,
        "imported memory; native i32 instructions, no host call",
    ))
}

fn run_tvm_mm_list(size: u32, bytes: &[u8], head_offset: u32) -> anyhow::Result<Sample> {
    use wasmtime::{Memory, MemoryType};
    let engine = enable_multi_memory()?;
    let module = Module::new(&engine, MM_WAT)?;
    let pages_needed = ((size as u64 + 65535) / 65536).max(1) as u32;
    let mut store = Store::new(&engine, ());
    let mem0 = Memory::new(
        &mut store,
        MemoryType::new(pages_needed, Some(pages_needed)),
    )?;
    let mem1 = Memory::new(&mut store, MemoryType::new(1, Some(1)))?;
    let mut linker: Linker<()> = Linker::new(&engine);
    linker.define(&mut store, "tvm", "r0", mem0)?;
    linker.define(&mut store, "tvm", "r1", mem1)?;
    let instance = linker.instantiate(&mut store, &module)?;
    mem0.write(&mut store, 0, bytes)?;
    let walk = instance.get_typed_func::<(i32, i32), i64>(&mut store, "list_walk_r0")?;
    let timings = time_loop(|| {
        let _ = walk.call(&mut store, (0, head_offset as i32))?;
        Ok(())
    })?;
    Ok(summarize(
        "tvm-mm",
        "list_walk",
        size,
        timings,
        "imported memory; native node loads",
    ))
}

fn run_tvm_mm_columnar(
    size: u32,
    col_a: &[u8],
    col_b: &[u8],
    n: u32,
    threshold: u32,
) -> anyhow::Result<Sample> {
    use wasmtime::{Memory, MemoryType};
    let engine = enable_multi_memory()?;
    let module = Module::new(&engine, MM_WAT)?;
    let pages_needed = ((size as u64 + 65535) / 65536).max(1) as u32;
    let mut store = Store::new(&engine, ());
    // Each column gets its own imported memory.
    let mem_a = Memory::new(
        &mut store,
        MemoryType::new(pages_needed, Some(pages_needed)),
    )?;
    let mem_b = Memory::new(
        &mut store,
        MemoryType::new(pages_needed, Some(pages_needed)),
    )?;
    let mut linker: Linker<()> = Linker::new(&engine);
    linker.define(&mut store, "tvm", "r0", mem_a)?;
    linker.define(&mut store, "tvm", "r1", mem_b)?;
    let instance = linker.instantiate(&mut store, &module)?;
    mem_a.write(&mut store, 0, col_a)?;
    mem_b.write(&mut store, 0, col_b)?;
    let q =
        instance.get_typed_func::<(i32, i32, i32, i32), i64>(&mut store, "columnar_filter_sum")?;
    let timings = time_loop(|| {
        let _ = q.call(&mut store, (0, 0, n as i32, threshold as i32))?;
        Ok(())
    })?;
    Ok(summarize(
        "tvm-mm",
        "columnar_filter_sum",
        size,
        timings,
        "two imported memories; one per column",
    ))
}

// -------------------- 4.1 sequential --------------------

fn bench_sequential(
    m32: &[u8],
    tvm: &[u8],
    m64: Option<&[u8]>,
    size: u32,
) -> anyhow::Result<Vec<Sample>> {
    let mut data = vec![0u8; size as usize];
    fill_pattern(&mut data, SEED);

    // M32
    let mut a = new_m32(m32)?;
    let memory = a.instance.get_memory(&mut a.store, "memory").unwrap();
    memory.write(&mut a.store, a.buffer_ptr as usize, &data)?;
    let sum = a
        .instance
        .get_typed_func::<(u32, u32), u64>(&mut a.store, "sum_sequential")?;
    let buf_ptr = a.buffer_ptr;
    let m32_t = time_loop(|| {
        let _ = sum.call(&mut a.store, (buf_ptr, size))?;
        Ok(())
    })?;
    let s_m32 = summarize(
        "m32",
        "sequential_sum",
        size,
        m32_t,
        "engine-emitted load loop",
    );

    // M64
    let s_m64 = if let Some(m64_bytes) = m64 {
        let mut c = new_m64(m64_bytes)?;
        let memory = c.instance.get_memory(&mut c.store, "memory").unwrap();
        memory.write(&mut c.store, c.buffer_ptr as usize, &data)?;
        let sum = c
            .instance
            .get_typed_func::<(u64, u64), u64>(&mut c.store, "sum_sequential")?;
        let buf_ptr = c.buffer_ptr;
        let m64_t = time_loop(|| {
            let _ = sum.call(&mut c.store, (buf_ptr, size as u64))?;
            Ok(())
        })?;
        Some(summarize(
            "m64",
            "sequential_sum",
            size,
            m64_t,
            "wasm64 engine-emitted load loop",
        ))
    } else {
        None
    };

    // TVM
    let mut b = new_tvm(tvm, size + 4096)?;
    let alloc = b
        .instance
        .get_typed_func::<u32, i64>(&mut b.store, "tvm_alloc_in_region0")?;
    let write = b
        .instance
        .get_typed_func::<(i64, u32, u32), i32>(&mut b.store, "tvm_write_bytes")?;
    let sum = b
        .instance
        .get_typed_func::<(i64, u32), u64>(&mut b.store, "tvm_sum_sequential")?;
    let memory = b.instance.get_memory(&mut b.store, "memory").unwrap();
    let handle = alloc.call(&mut b.store, size)?;
    memory.write(&mut b.store, 0, &data)?;
    let _ = write.call(&mut b.store, (handle, 0, size))?;
    let tvm_t = time_loop(|| {
        let _ = sum.call(&mut b.store, (handle, size))?;
        Ok(())
    })?;
    let _ = b.region;
    let s_tvm = summarize(
        "tvm",
        "sequential_sum",
        size,
        tvm_t,
        "raw fast path; bulk reads",
    );

    let s_mm = run_tvm_mm_sequential(size, &data)
        .map_err(|e| eprintln!("tvm-mm sequential failed: {e}"))
        .ok();
    let s_unified = run_tvm_unified_sequential(size, &data)
        .map_err(|e| eprintln!("tvm-unified sequential failed: {e}"))
        .ok();
    let mut v = samples_with_mm(s_m32, s_m64, s_mm, s_tvm);
    if let Some(u) = s_unified {
        v.push(u);
    }
    Ok(v)
}

fn samples_with_optional(m32: Sample, m64: Option<Sample>, tvm: Sample) -> Vec<Sample> {
    let mut v = vec![m32];
    if let Some(m) = m64 {
        v.push(m);
    }
    v.push(tvm);
    v
}

fn samples_with_mm(
    m32: Sample,
    m64: Option<Sample>,
    mm: Option<Sample>,
    tvm: Sample,
) -> Vec<Sample> {
    let mut v = vec![m32];
    if let Some(m) = m64 {
        v.push(m);
    }
    if let Some(m) = mm {
        v.push(m);
    }
    v.push(tvm);
    v
}

// -------------------- 4.2 random access --------------------

fn bench_random(
    m32: &[u8],
    tvm: &[u8],
    m64: Option<&[u8]>,
    size: u32,
) -> anyhow::Result<Vec<Sample>> {
    let cells = size / 4;
    let steps = 4 * cells; // 4 loops over the ring
                           // Build a deterministic ring: cell[i] = next index, where each index
                           // appears once.
    let mut ring: Vec<u32> = (0..cells).collect();
    // Knuth shuffle with fixed seed for determinism.
    let mut state = SEED.wrapping_add(7);
    for i in (1..cells as usize).rev() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let j = (state as usize) % (i + 1);
        ring.swap(i, j);
    }
    // Convert into "next pointer" form: pos `ring[i]` should hold `ring[i+1]`.
    let mut bytes = vec![0u8; (cells * 4) as usize];
    for i in 0..cells as usize {
        let idx_here = ring[i];
        let idx_next = ring[(i + 1) % cells as usize];
        let off = (idx_here * 4) as usize;
        bytes[off..off + 4].copy_from_slice(&idx_next.to_le_bytes());
    }

    let mut a = new_m32(m32)?;
    let memory = a.instance.get_memory(&mut a.store, "memory").unwrap();
    memory.write(&mut a.store, a.buffer_ptr as usize, &bytes)?;
    let chase = a
        .instance
        .get_typed_func::<(u32, u32, u32), u64>(&mut a.store, "random_chase")?;
    let buf_ptr = a.buffer_ptr;
    let m32_t = time_loop(|| {
        let _ = chase.call(&mut a.store, (buf_ptr, cells, steps))?;
        Ok(())
    })?;
    let s_m32 = summarize(
        "m32",
        "random_chase",
        size,
        m32_t,
        "engine-emitted u32 load per step",
    );

    // M64
    let s_m64 = if let Some(m64_bytes) = m64 {
        let mut c = new_m64(m64_bytes)?;
        let memory = c.instance.get_memory(&mut c.store, "memory").unwrap();
        memory.write(&mut c.store, c.buffer_ptr as usize, &bytes)?;
        let chase = c
            .instance
            .get_typed_func::<(u64, u64, u64), u64>(&mut c.store, "random_chase")?;
        let buf_ptr = c.buffer_ptr;
        let m64_t = time_loop(|| {
            let _ = chase.call(&mut c.store, (buf_ptr, cells as u64, steps as u64))?;
            Ok(())
        })?;
        Some(summarize(
            "m64",
            "random_chase",
            size,
            m64_t,
            "wasm64 engine-emitted u32 load per step",
        ))
    } else {
        None
    };

    let mut b = new_tvm(tvm, size + 4096)?;
    let alloc = b
        .instance
        .get_typed_func::<u32, i64>(&mut b.store, "tvm_alloc_in_region0")?;
    let write = b
        .instance
        .get_typed_func::<(i64, u32, u32), i32>(&mut b.store, "tvm_write_bytes")?;
    let chase = b
        .instance
        .get_typed_func::<(i64, u32, u32), u64>(&mut b.store, "tvm_random_chase")?;
    let memory = b.instance.get_memory(&mut b.store, "memory").unwrap();
    let handle = alloc.call(&mut b.store, cells * 4)?;
    memory.write(&mut b.store, 0, &bytes)?;
    let _ = write.call(&mut b.store, (handle, 0, cells * 4))?;
    let tvm_t = time_loop(|| {
        let _ = chase.call(&mut b.store, (handle, cells, steps))?;
        Ok(())
    })?;
    let s_tvm = summarize(
        "tvm",
        "random_chase",
        size,
        tvm_t,
        "raw bulk read; chase in guest",
    );
    Ok(samples_with_optional(s_m32, s_m64, s_tvm))
}

// -------------------- 4.3 pointer-heavy (linked list) --------------------

fn bench_list(
    m32: &[u8],
    tvm: &[u8],
    m64: Option<&[u8]>,
    size: u32,
) -> anyhow::Result<Vec<Sample>> {
    let nodes = size / 8; // each node is 8 bytes
    let bytes_len = (nodes * 8) as usize;
    // Build a list whose order is shuffled so next-pointers don't trivially
    // line up with next-in-memory (would defeat the test).
    let mut order: Vec<u32> = (0..nodes).collect();
    let mut state = SEED.wrapping_add(11);
    for i in (1..nodes as usize).rev() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let j = (state as usize) % (i + 1);
        order.swap(i, j);
    }
    let head_offset = order[0] * 8;
    let sentinel: u32 = 0xFFFF_FFFF;
    let mut bytes = vec![0u8; bytes_len];
    for i in 0..nodes as usize {
        let here = order[i] as usize;
        let next_off = if i + 1 < nodes as usize {
            order[i + 1] * 8
        } else {
            sentinel
        };
        bytes[here * 8..here * 8 + 4].copy_from_slice(&next_off.to_le_bytes());
        bytes[here * 8 + 4..here * 8 + 8].copy_from_slice(&(i as u32).to_le_bytes());
    }

    let mut a = new_m32(m32)?;
    let memory = a.instance.get_memory(&mut a.store, "memory").unwrap();
    memory.write(&mut a.store, a.buffer_ptr as usize, &bytes)?;
    let walk = a
        .instance
        .get_typed_func::<(u32, u32), u64>(&mut a.store, "list_walk")?;
    let buf_ptr = a.buffer_ptr;
    let m32_t = time_loop(|| {
        let _ = walk.call(&mut a.store, (buf_ptr, head_offset))?;
        Ok(())
    })?;
    let s_m32 = summarize("m32", "list_walk", size, m32_t, "engine-emitted node load");

    let s_m64 = if let Some(m64_bytes) = m64 {
        let mut c = new_m64(m64_bytes)?;
        let memory = c.instance.get_memory(&mut c.store, "memory").unwrap();
        memory.write(&mut c.store, c.buffer_ptr as usize, &bytes)?;
        let walk = c
            .instance
            .get_typed_func::<(u64, u64), u64>(&mut c.store, "list_walk")?;
        let buf_ptr = c.buffer_ptr;
        let m64_t = time_loop(|| {
            let _ = walk.call(&mut c.store, (buf_ptr, head_offset as u64))?;
            Ok(())
        })?;
        Some(summarize(
            "m64",
            "list_walk",
            size,
            m64_t,
            "wasm64 node load",
        ))
    } else {
        None
    };

    let mut b = new_tvm(tvm, size + 4096)?;
    let alloc = b
        .instance
        .get_typed_func::<u32, i64>(&mut b.store, "tvm_alloc_in_region0")?;
    let write = b
        .instance
        .get_typed_func::<(i64, u32, u32), i32>(&mut b.store, "tvm_write_bytes")?;
    let walk = b
        .instance
        .get_typed_func::<(i64, u32, u32), u64>(&mut b.store, "tvm_list_walk")?;
    let memory = b.instance.get_memory(&mut b.store, "memory").unwrap();
    let handle = alloc.call(&mut b.store, nodes * 8)?;
    memory.write(&mut b.store, 0, &bytes)?;
    let _ = write.call(&mut b.store, (handle, 0, nodes * 8))?;
    let total_bytes = nodes * 8;
    let tvm_t = time_loop(|| {
        let _ = walk.call(&mut b.store, (handle, head_offset, total_bytes))?;
        Ok(())
    })?;
    let s_tvm = summarize(
        "tvm",
        "list_walk",
        size,
        tvm_t,
        "raw bulk read; walk in guest",
    );
    let s_mm = run_tvm_mm_list(size, &bytes, head_offset).ok();
    Ok(samples_with_mm(s_m32, s_m64, s_mm, s_tvm))
}

// -------------------- 4.4 growth --------------------

fn bench_growth(
    m32: &[u8],
    tvm: &[u8],
    m64: Option<&[u8]>,
    size: u32,
) -> anyhow::Result<Vec<Sample>> {
    let block = 32u32;
    let count = size / block;

    let mut a = new_m32(m32)?;
    let bump = a
        .instance
        .get_typed_func::<(u32, u32, u32), u64>(&mut a.store, "bump_alloc_touch")?;
    let buf_ptr = a.buffer_ptr;
    let m32_t = time_loop(|| {
        let _ = bump.call(&mut a.store, (buf_ptr, count, block))?;
        Ok(())
    })?;
    let s_m32 = summarize("m32", "growth_bump", size, m32_t, "static buffer + index");

    let s_m64 = if let Some(m64_bytes) = m64 {
        let mut c = new_m64(m64_bytes)?;
        let bump = c
            .instance
            .get_typed_func::<(u64, u64, u64), u64>(&mut c.store, "bump_alloc_touch")?;
        let buf_ptr = c.buffer_ptr;
        let m64_t = time_loop(|| {
            let _ = bump.call(&mut c.store, (buf_ptr, count as u64, block as u64))?;
            Ok(())
        })?;
        Some(summarize(
            "m64",
            "growth_bump",
            size,
            m64_t,
            "wasm64 static + index",
        ))
    } else {
        None
    };

    let mut tvm_t = Vec::with_capacity(SAMPLES);
    for _ in 0..(SAMPLES + WARMUP_ROUNDS) {
        let mut b = new_tvm(tvm, size + 4096)?;
        let bump = b
            .instance
            .get_typed_func::<(u32, u32, u32), u64>(&mut b.store, "tvm_bump_alloc_touch")?;
        let region_id = b.region as u32;
        let start = Instant::now();
        let _ = bump.call(&mut b.store, (region_id, count, block))?;
        let elapsed = start.elapsed();
        if tvm_t.len() < SAMPLES {
            tvm_t.push(elapsed);
        }
    }
    let s_tvm = summarize("tvm", "growth_bump", size, tvm_t, "fresh region per sample");
    Ok(samples_with_optional(s_m32, s_m64, s_tvm))
}

// -------------------- 4.5 multi-region --------------------

fn bench_multi_region(
    m32: &[u8],
    tvm: &[u8],
    m64: Option<&[u8]>,
    size: u32,
) -> anyhow::Result<Vec<Sample>> {
    // Partition: 60% hot, 30% warm, 10% cold.
    let hot_size = size * 6 / 10;
    let warm_size = size * 3 / 10;
    let cold_size = size - hot_size - warm_size;
    let iters = 10_000u32;

    // Seed each partition with deterministic bytes.
    let mut hot = vec![0u8; hot_size as usize];
    let mut warm = vec![0u8; warm_size as usize];
    let mut cold = vec![0u8; cold_size as usize];
    fill_pattern(&mut hot, SEED);
    fill_pattern(&mut warm, SEED.wrapping_add(1));
    fill_pattern(&mut cold, SEED.wrapping_add(2));

    let mut a = new_m32(m32)?;
    let memory = a.instance.get_memory(&mut a.store, "memory").unwrap();
    memory.write(&mut a.store, a.buffer_ptr as usize, &hot)?;
    memory.write(
        &mut a.store,
        a.buffer_ptr as usize + hot_size as usize,
        &warm,
    )?;
    memory.write(
        &mut a.store,
        a.buffer_ptr as usize + (hot_size + warm_size) as usize,
        &cold,
    )?;
    let mix = a
        .instance
        .get_typed_func::<(u32, u32, u32, u32, u32, u32), u64>(&mut a.store, "multi_region_mix")?;
    let buf_ptr = a.buffer_ptr;
    let m32_t = time_loop(|| {
        let _ = mix.call(
            &mut a.store,
            (buf_ptr, hot_size, warm_size, cold_size, iters, SEED),
        )?;
        Ok(())
    })?;
    let s_m32 = summarize(
        "m32",
        "multi_region_90_9_1",
        size,
        m32_t,
        "single linear memory; hot/warm/cold contiguous",
    );

    // M64
    let s_m64 = if let Some(m64_bytes) = m64 {
        let mut c = new_m64(m64_bytes)?;
        let memory = c.instance.get_memory(&mut c.store, "memory").unwrap();
        memory.write(&mut c.store, c.buffer_ptr as usize, &hot)?;
        memory.write(
            &mut c.store,
            c.buffer_ptr as usize + hot_size as usize,
            &warm,
        )?;
        memory.write(
            &mut c.store,
            c.buffer_ptr as usize + (hot_size + warm_size) as usize,
            &cold,
        )?;
        let mix = c
            .instance
            .get_typed_func::<(u64, u64, u64, u64, u64, u32), u64>(
                &mut c.store,
                "multi_region_mix",
            )?;
        let buf_ptr = c.buffer_ptr;
        let m64_t = time_loop(|| {
            let _ = mix.call(
                &mut c.store,
                (
                    buf_ptr,
                    hot_size as u64,
                    warm_size as u64,
                    cold_size as u64,
                    iters as u64,
                    SEED,
                ),
            )?;
            Ok(())
        })?;
        Some(summarize(
            "m64",
            "multi_region_90_9_1",
            size,
            m64_t,
            "wasm64 single memory; hot/warm/cold contiguous",
        ))
    } else {
        None
    };

    // TVM: allocate three separate regions.
    let engine = Engine::default();
    let module = Module::new(&engine, tvm)?;
    let mut host = TvmHost::new();
    let r_hot = ManagerHost::create_region(&mut host, RegionKind::HotHeap, hot_size + 64).unwrap();
    let r_warm =
        ManagerHost::create_region(&mut host, RegionKind::ObjectArena, warm_size + 64).unwrap();
    let r_cold =
        ManagerHost::create_region(&mut host, RegionKind::ObjectArena, cold_size + 64).unwrap();
    assert_eq!(r_hot, 0);
    let mut linker: Linker<TvmHost> = Linker::new(&engine);
    add_raw_imports(&mut linker)?;
    let mut store = Store::new(&engine, host);
    let instance = linker.instantiate(&mut store, &module)?;
    let alloc = instance.get_typed_func::<u32, i64>(&mut store, "tvm_alloc_in_region0")?;
    let write = instance.get_typed_func::<(i64, u32, u32), i32>(&mut store, "tvm_write_bytes")?;
    let memory = instance.get_memory(&mut store, "memory").unwrap();

    let h_hot = alloc.call(&mut store, hot_size)?;
    memory.write(&mut store, 0, &hot)?;
    let _ = write.call(&mut store, (h_hot, 0, hot_size))?;

    // Allocate inside r_warm and r_cold by constructing handles directly via
    // the manager. Easier: reuse alloc but via a host-side alloc since we
    // have access to the directory.
    let h_warm_off = store.data_mut().directory.alloc(r_warm, warm_size)?;
    let h_cold_off = store.data_mut().directory.alloc(r_cold, cold_size)?;
    // Pack into i64 for the workload.
    let h_warm = (h_warm_off.region_id as i64) << 48
        | (h_warm_off.generation as i64) << 32
        | (h_warm_off.offset as i64);
    let h_cold = (h_cold_off.region_id as i64) << 48
        | (h_cold_off.generation as i64) << 32
        | (h_cold_off.offset as i64);
    memory.write(&mut store, 0, &warm)?;
    let _ = write.call(&mut store, (h_warm, 0, warm_size))?;
    memory.write(&mut store, 0, &cold)?;
    let _ = write.call(&mut store, (h_cold, 0, cold_size))?;

    let mix = instance.get_typed_func::<(i64, u32, i64, u32, i64, u32, u32, u32), u64>(
        &mut store,
        "tvm_multi_region_mix",
    )?;
    let tvm_t = time_loop(|| {
        let _ = mix.call(
            &mut store,
            (
                h_hot, hot_size, h_warm, warm_size, h_cold, cold_size, iters, SEED,
            ),
        )?;
        Ok(())
    })?;
    let s_tvm = summarize(
        "tvm",
        "multi_region_90_9_1",
        size,
        tvm_t,
        "three regions; per-tier residency hint",
    );
    Ok(samples_with_optional(s_m32, s_m64, s_tvm))
}

// -------------------- 4.6 columnar --------------------

fn bench_columnar(
    m32: &[u8],
    tvm: &[u8],
    m64: Option<&[u8]>,
    size: u32,
) -> anyhow::Result<Vec<Sample>> {
    let n = size / 8; // u32 col_a + u32 col_b
                      // Fill col_a with deterministic 0..n and col_b with a payload.
    let mut col_a = vec![0u8; (n * 4) as usize];
    let mut col_b = vec![0u8; (n * 4) as usize];
    for i in 0..n {
        let off = (i * 4) as usize;
        col_a[off..off + 4].copy_from_slice(&i.to_le_bytes());
        col_b[off..off + 4].copy_from_slice(&(i.wrapping_mul(13)).to_le_bytes());
    }
    let threshold = n / 2;

    let mut a = new_m32(m32)?;
    let memory = a.instance.get_memory(&mut a.store, "memory").unwrap();
    memory.write(&mut a.store, a.buffer_ptr as usize, &col_a)?;
    memory.write(&mut a.store, a.buffer_ptr as usize + col_a.len(), &col_b)?;
    let q = a
        .instance
        .get_typed_func::<(u32, u32, u32), u64>(&mut a.store, "columnar_filter_sum")?;
    let buf_ptr = a.buffer_ptr;
    let m32_t = time_loop(|| {
        let _ = q.call(&mut a.store, (buf_ptr, n, threshold))?;
        Ok(())
    })?;
    let s_m32 = summarize(
        "m32",
        "columnar_filter_sum",
        size,
        m32_t,
        "two cols contiguous",
    );

    // M64
    let s_m64 = if let Some(m64_bytes) = m64 {
        let mut c = new_m64(m64_bytes)?;
        let memory = c.instance.get_memory(&mut c.store, "memory").unwrap();
        memory.write(&mut c.store, c.buffer_ptr as usize, &col_a)?;
        memory.write(&mut c.store, c.buffer_ptr as usize + col_a.len(), &col_b)?;
        let q = c
            .instance
            .get_typed_func::<(u64, u64, u32), u64>(&mut c.store, "columnar_filter_sum")?;
        let buf_ptr = c.buffer_ptr;
        let m64_t = time_loop(|| {
            let _ = q.call(&mut c.store, (buf_ptr, n as u64, threshold))?;
            Ok(())
        })?;
        Some(summarize(
            "m64",
            "columnar_filter_sum",
            size,
            m64_t,
            "wasm64 two cols contiguous",
        ))
    } else {
        None
    };

    // TVM: separate region per column.
    let engine = Engine::default();
    let module = Module::new(&engine, tvm)?;
    let mut host = TvmHost::new();
    let r_a = ManagerHost::create_region(&mut host, RegionKind::HotHeap, n * 4 + 64).unwrap();
    let r_b = ManagerHost::create_region(&mut host, RegionKind::HotHeap, n * 4 + 64).unwrap();
    let mut linker: Linker<TvmHost> = Linker::new(&engine);
    add_raw_imports(&mut linker)?;
    let mut store = Store::new(&engine, host);
    let instance = linker.instantiate(&mut store, &module)?;
    let alloc = instance.get_typed_func::<u32, i64>(&mut store, "tvm_alloc_in_region0")?;
    let write = instance.get_typed_func::<(i64, u32, u32), i32>(&mut store, "tvm_write_bytes")?;
    let memory = instance.get_memory(&mut store, "memory").unwrap();
    // Region 0 was r_a.
    let h_a = alloc.call(&mut store, n * 4)?;
    memory.write(&mut store, 0, &col_a)?;
    let _ = write.call(&mut store, (h_a, 0, n * 4))?;
    let h_b_handle = store.data_mut().directory.alloc(r_b, n * 4)?;
    let h_b = (h_b_handle.region_id as i64) << 48
        | (h_b_handle.generation as i64) << 32
        | (h_b_handle.offset as i64);
    memory.write(&mut store, 0, &col_b)?;
    let _ = write.call(&mut store, (h_b, 0, n * 4))?;
    let q = instance
        .get_typed_func::<(i64, i64, u32, u32), u64>(&mut store, "tvm_columnar_filter_sum")?;
    let tvm_t = time_loop(|| {
        let _ = q.call(&mut store, (h_a, h_b, n, threshold))?;
        Ok(())
    })?;
    let s_tvm = summarize(
        "tvm",
        "columnar_filter_sum",
        size,
        tvm_t,
        "one region per column; bulk reads",
    );
    let _ = r_a;
    let s_mm = run_tvm_mm_columnar(size, &col_a, &col_b, n, threshold).ok();
    Ok(samples_with_mm(s_m32, s_m64, s_mm, s_tvm))
}

// -------------------- 4.9 spill-driven --------------------
//
// Working set is 4× the per-region budget. TVM cycles regions through
// Cold tier; M32/M64 cannot represent this scenario without a separately-
// implemented swap mechanism, so we report a ZERO-cost reference for them
// labeled "infeasible: working set exceeds linear memory budget" and the
// real TVM number. This is the unique-capability bench.

fn bench_spill_driven(
    _m32: &[u8],
    tvm: &[u8],
    _m64: Option<&[u8]>,
    size: u32,
) -> anyhow::Result<Vec<Sample>> {
    use tempfile::tempdir;

    let resident_budget = size; // resident cap
    let n_tiers = 4u32; // total working set = resident_budget * n_tiers

    let m32_s = Sample {
        variant: "m32".into(),
        class: "spill_driven".into(),
        size_bytes: size,
        runtime: "wasmtime",
        samples: 0,
        warmup: 0,
        seed: SEED,
        mean_ns: 0.0,
        p50_ns: 0,
        p95_ns: 0,
        p99_ns: 0,
        throughput_gib_per_s: 0.0,
        notes: format!(
            "infeasible: working set ({} B) exceeds resident budget ({} B); \
             requires user-implemented swap",
            resident_budget * n_tiers,
            resident_budget
        ),
        raw_ns: vec![],
        mann_whitney_u_vs_baseline: None,
        speedup_vs_baseline: None,
        speedup_vs_m64: None,
    };
    let m64_s = Sample {
        variant: "m64".into(),
        ..m32_s.clone()
    };

    // TVM: build a host with a backing store, create n_tiers regions of
    // resident_budget bytes each, do a workload that touches each tier in
    // round-robin, and let the directory spill the inactive ones.
    let tmp = tempdir()?;
    let backing_path = tmp.path().to_path_buf();

    let engine = Engine::default();
    let module = Module::new(&engine, tvm)?;
    let host = TvmHost::with_backing(&backing_path)?;
    let mut linker: Linker<TvmHost> = Linker::new(&engine);
    add_raw_imports(&mut linker)?;
    let mut store = Store::new(&engine, host);
    let instance = linker.instantiate(&mut store, &module)?;

    // Create n_tiers regions; allocate inside each.
    let mut handles: Vec<i64> = Vec::with_capacity(n_tiers as usize);
    for _ in 0..n_tiers {
        let r = ManagerHost::create_region(
            store.data_mut(),
            RegionKind::ObjectArena,
            resident_budget + 64,
        )?;
        let h = store.data_mut().directory.alloc(r, resident_budget)?;
        let packed = (h.region_id as i64) << 48 | (h.generation as i64) << 32 | (h.offset as i64);
        // Write a few bytes so the spill has something real to persist.
        store.data_mut().directory.write(h, &vec![0xAA; 64])?;
        handles.push(packed);
    }

    // Touch each handle in round-robin, demoting the just-touched-and-now-
    // inactive ones to Cold via spill. Time the cycle.
    let read_fn = instance.get_typed_func::<(i64, u32), u64>(&mut store, "tvm_sum_sequential")?;

    let cycles = 8u32;
    let touched_per_cycle = n_tiers;
    let tvm_t = time_loop(|| {
        for cycle in 0..cycles {
            let active = (cycle % n_tiers) as usize;
            // Demote everything except the active region.
            for (i, _) in handles.iter().enumerate() {
                if i != active {
                    let region_id = ((handles[i] >> 48) & 0xFFFF) as u16;
                    let _ = ManagerHost::spill_region(store.data_mut(), region_id);
                }
            }
            let _ = read_fn.call(&mut store, (handles[active], 64))?;
        }
        let _ = touched_per_cycle;
        Ok(())
    })?;
    let s_tvm = summarize(
        "tvm",
        "spill_driven",
        size,
        tvm_t,
        "n_tiers=4 round-robin with spill; only TVM can represent",
    );

    Ok(vec![m32_s, m64_s, s_tvm])
}

// -------------------- 4.8 large-working-set probe --------------------

fn bench_large_ws(
    m32: &[u8],
    tvm: &[u8],
    m64: Option<&[u8]>,
    size: u32,
) -> anyhow::Result<Vec<Sample>> {
    // 16 blocks × (size/16) bytes per block; 10K iters of random access.
    let n_blocks = 16u32;
    let block_size = size / n_blocks;
    let iters = 10_000u32;

    // Generate the data once; same content across variants.
    let mut data = vec![0u8; (n_blocks * block_size) as usize];
    fill_pattern(&mut data, SEED);

    let mut a = new_m32(m32)?;
    let memory = a.instance.get_memory(&mut a.store, "memory").unwrap();
    memory.write(&mut a.store, a.buffer_ptr as usize, &data)?;
    let probe = a
        .instance
        .get_typed_func::<(u32, u32, u32, u32, u32), u64>(&mut a.store, "large_ws_probe")?;
    let buf_ptr = a.buffer_ptr;
    let m32_t = time_loop(|| {
        let _ = probe.call(&mut a.store, (buf_ptr, block_size, n_blocks, iters, SEED))?;
        Ok(())
    })?;
    let s_m32 = summarize(
        "m32",
        "large_ws_probe",
        size,
        m32_t,
        "single 32-bit linear memory; n_blocks contiguous",
    );

    let s_m64 = if let Some(m64_bytes) = m64 {
        let mut c = new_m64(m64_bytes)?;
        let memory = c.instance.get_memory(&mut c.store, "memory").unwrap();
        memory.write(&mut c.store, c.buffer_ptr as usize, &data)?;
        let probe = c
            .instance
            .get_typed_func::<(u64, u64, u64, u64, u32), u64>(&mut c.store, "large_ws_probe")?;
        let buf_ptr = c.buffer_ptr;
        let m64_t = time_loop(|| {
            let _ = probe.call(
                &mut c.store,
                (
                    buf_ptr,
                    block_size as u64,
                    n_blocks as u64,
                    iters as u64,
                    SEED,
                ),
            )?;
            Ok(())
        })?;
        Some(summarize(
            "m64",
            "large_ws_probe",
            size,
            m64_t,
            "wasm64 single memory; n_blocks contiguous",
        ))
    } else {
        None
    };

    // TVM: one region per block.
    let engine = Engine::default();
    let module = Module::new(&engine, tvm)?;
    let mut host = TvmHost::new();
    for _ in 0..n_blocks {
        ManagerHost::create_region(&mut host, RegionKind::HotHeap, block_size + 64).unwrap();
    }
    let mut linker: Linker<TvmHost> = Linker::new(&engine);
    add_raw_imports(&mut linker)?;
    let mut store = Store::new(&engine, host);
    let instance = linker.instantiate(&mut store, &module)?;
    let alloc = instance.get_typed_func::<u32, i64>(&mut store, "tvm_alloc_in_region0")?;
    let write = instance.get_typed_func::<(i64, u32, u32), i32>(&mut store, "tvm_write_bytes")?;
    let memory = instance.get_memory(&mut store, "memory").unwrap();

    // Seed each block: alloc handle in its own region, write data.
    let mut packed_handles: Vec<i64> = Vec::with_capacity(n_blocks as usize);
    // Region 0 is the only one alloc-via-export reaches; for the rest we
    // directly call the directory.
    let h0 = alloc.call(&mut store, block_size)?;
    memory.write(&mut store, 0, &data[..block_size as usize])?;
    let _ = write.call(&mut store, (h0, 0, block_size))?;
    packed_handles.push(h0);
    for i in 1..n_blocks {
        let off = (i * block_size) as usize;
        let chunk = &data[off..off + block_size as usize];
        let h = store.data_mut().directory.alloc(i as u16, block_size)?;
        let packed = (h.region_id as i64) << 48 | (h.generation as i64) << 32 | (h.offset as i64);
        memory.write(&mut store, 0, chunk)?;
        let _ = write.call(&mut store, (packed, 0, block_size))?;
        packed_handles.push(packed);
    }
    // Pack handle array at a known location in guest memory (we'll use offset 1MB so it doesn't collide with our static buffers).
    let handles_offset = 4 * 1024 * 1024u32;
    let mut handles_bytes = Vec::with_capacity((n_blocks as usize) * 8);
    for &p in &packed_handles {
        handles_bytes.extend_from_slice(&p.to_le_bytes());
    }
    // Grow memory if needed.
    let needed_pages = ((handles_offset as usize + handles_bytes.len()) + 65535) / 65536;
    let current_size = memory.data_size(&store);
    if current_size < needed_pages * 65536 {
        let extra = (needed_pages * 65536 - current_size + 65535) / 65536;
        memory.grow(&mut store, extra as u64)?;
    }
    memory.write(&mut store, handles_offset as usize, &handles_bytes)?;

    let probe = instance
        .get_typed_func::<(u32, u32, u32, u32, u32), u64>(&mut store, "tvm_large_ws_probe")?;
    let tvm_t = time_loop(|| {
        let _ = probe.call(
            &mut store,
            (handles_offset, n_blocks, block_size, iters, SEED),
        )?;
        Ok(())
    })?;
    let s_tvm = summarize(
        "tvm",
        "large_ws_probe",
        size,
        tvm_t,
        "one region per block; bulk read on entry",
    );

    Ok(samples_with_optional(s_m32, s_m64, s_tvm))
}

// -------------------- 4.7 JVM heap --------------------

fn bench_jvm(m32: &[u8], tvm: &[u8], m64: Option<&[u8]>, size: u32) -> anyhow::Result<Vec<Sample>> {
    let n = size / 32;
    let mut a = new_m32(m32)?;
    let scan = a
        .instance
        .get_typed_func::<(u32, u32), u64>(&mut a.store, "gen_alloc_scan")?;
    let buf_ptr = a.buffer_ptr;
    let m32_t = time_loop(|| {
        let _ = scan.call(&mut a.store, (buf_ptr, n))?;
        Ok(())
    })?;
    let s_m32 = summarize(
        "m32",
        "jvm_gen_alloc_scan",
        size,
        m32_t,
        "static buffer + index",
    );

    let s_m64 = if let Some(m64_bytes) = m64 {
        let mut c = new_m64(m64_bytes)?;
        let scan = c
            .instance
            .get_typed_func::<(u64, u64), u64>(&mut c.store, "gen_alloc_scan")?;
        let buf_ptr = c.buffer_ptr;
        let m64_t = time_loop(|| {
            let _ = scan.call(&mut c.store, (buf_ptr, n as u64))?;
            Ok(())
        })?;
        Some(summarize(
            "m64",
            "jvm_gen_alloc_scan",
            size,
            m64_t,
            "wasm64 static + index",
        ))
    } else {
        None
    };

    let mut tvm_t = Vec::with_capacity(SAMPLES);
    for _ in 0..(SAMPLES + WARMUP_ROUNDS) {
        let mut b = new_tvm(tvm, size + 4096)?;
        let scan = b
            .instance
            .get_typed_func::<(u32, u32), u64>(&mut b.store, "tvm_gen_alloc_scan")?;
        let region_id = b.region as u32;
        let start = Instant::now();
        let _ = scan.call(&mut b.store, (region_id, n))?;
        let elapsed = start.elapsed();
        if tvm_t.len() < SAMPLES {
            tvm_t.push(elapsed);
        }
    }
    let s_tvm = summarize(
        "tvm",
        "jvm_gen_alloc_scan",
        size,
        tvm_t,
        "fresh region per sample; alloc-then-scan",
    );
    Ok(samples_with_optional(s_m32, s_m64, s_tvm))
}

// -------------------- harness --------------------

fn auto_build_workloads() -> anyhow::Result<()> {
    let root = workspace_root();
    for dir in &[
        "bench-framework/workloads/m32",
        "bench-framework/workloads/tvm",
    ] {
        let status = Command::new("cargo")
            .current_dir(root.join(dir))
            .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
            .status()?;
        if !status.success() {
            anyhow::bail!("workload build failed in {dir}");
        }
    }
    // M64 is optional — requires nightly + rust-src. Best-effort build.
    let m64_dir = root.join("bench-framework/workloads/m64");
    let _ = Command::new("cargo")
        .current_dir(&m64_dir)
        .args([
            "+nightly",
            "build",
            "--release",
            "-Zbuild-std=panic_abort,std",
            "--target",
            "wasm64-unknown-unknown",
        ])
        .status();
    Ok(())
}

fn try_read_m64_wasm() -> Option<Vec<u8>> {
    let path = workspace_root().join(
        "bench-framework/workloads/m64/target/wasm64-unknown-unknown/release/tvm_bench_workload_m64.wasm",
    );
    std::fs::read(&path).ok()
}

fn read_wasm(rel: &str) -> anyhow::Result<Vec<u8>> {
    let path = workspace_root().join(rel);
    std::fs::read(&path).with_context(|| format!("missing wasm: {}", path.display()))
}

fn annotate_against_baseline(samples: &mut [Sample]) {
    if samples.is_empty() {
        return;
    }
    let baseline_raw = samples[0].raw_ns.clone();
    let baseline_mean = samples[0].mean_ns;
    samples[0].mann_whitney_u_vs_baseline = Some(0.5);
    samples[0].speedup_vs_baseline = Some(1.0);
    let m64_mean = samples
        .iter()
        .find(|s| s.variant == "m64" && s.mean_ns > 0.0)
        .map(|s| s.mean_ns);
    for s in samples.iter_mut().skip(1) {
        let u = mann_whitney_u(&baseline_raw, &s.raw_ns);
        let speedup = if s.mean_ns > 0.0 {
            baseline_mean / s.mean_ns
        } else {
            0.0
        };
        s.mann_whitney_u_vs_baseline = Some(u);
        s.speedup_vs_baseline = Some(speedup);
        if let Some(m) = m64_mean {
            if (s.variant == "tvm" || s.variant == "tvm-mm") && s.mean_ns > 0.0 {
                s.speedup_vs_m64 = Some(m / s.mean_ns);
            }
        }
    }
}

fn main() -> anyhow::Result<()> {
    println!("==> auto-building workloads...");
    auto_build_workloads()?;

    let m32_wasm = read_wasm(
        "bench-framework/workloads/m32/target/wasm32-unknown-unknown/release/tvm_bench_workload_m32.wasm",
    )?;
    let tvm_wasm = read_wasm(
        "bench-framework/workloads/tvm/target/wasm32-unknown-unknown/release/tvm_bench_workload_tvm.wasm",
    )?;
    let m64_wasm = try_read_m64_wasm();
    let m64_status = if m64_wasm.is_some() {
        "available"
    } else {
        "absent — skipping"
    };
    println!("==> M64 backend: {}", m64_status);

    let mut results: Vec<Sample> = Vec::new();

    println!(
        "==> running benchmarks ({} samples × warmup {}, seed {:#x})",
        SAMPLES, WARMUP_ROUNDS, SEED
    );

    // Function-pointer dispatch. A `Bench` trait would be more idiomatic
    // but the bench fns differ enough in setup (some need three regions,
    // some need a single buffer with sub-partitions) that the current
    // shape doesn't generalize cleanly. The bench-class signature has
    // stayed `(m32, tvm, m64, size) -> Vec<Sample>` since the framework
    // started; converting to a trait is tracked as future work.
    type BenchFn = fn(&[u8], &[u8], Option<&[u8]>, u32) -> anyhow::Result<Vec<Sample>>;
    let bench_fns: &[(&str, BenchFn)] = &[
        ("sequential", bench_sequential),
        ("random", bench_random),
        ("list", bench_list),
        ("growth", bench_growth),
        ("multi_region", bench_multi_region),
        ("columnar", bench_columnar),
        ("jvm", bench_jvm),
        ("large_ws", bench_large_ws),
        ("spill_driven", bench_spill_driven),
    ];

    let m64_ref = m64_wasm.as_deref();

    println!(
        "\n{:<14} {:>10} {:>12} {:>12} {:>12} {:>12} {:>12} {:>10}",
        "class", "size", "m32", "m64", "tvm-mm", "tvm-uni", "tvm", "TVM/M64"
    );
    println!("{}", "-".repeat(110));
    for &size in SIZES {
        for (name, f) in bench_fns {
            let mut samples = f(&m32_wasm, &tvm_wasm, m64_ref, size)?;
            annotate_against_baseline(&mut samples);
            let by_variant = |v: &str| {
                samples
                    .iter()
                    .find(|s| s.variant == v)
                    .map(|s| s.mean_ns)
                    .unwrap_or(f64::NAN)
            };
            let tvm_vs_m64 = samples
                .iter()
                .find(|s| s.variant == "tvm")
                .and_then(|s| s.speedup_vs_m64)
                .unwrap_or(f64::NAN);
            println!(
                "{:<14} {:>10} {:>12.0} {:>12.0} {:>12.0} {:>12.0} {:>12.0} {:>9.2}x",
                name,
                size,
                by_variant("m32"),
                by_variant("m64"),
                by_variant("tvm-mm"),
                by_variant("tvm-unified"),
                by_variant("tvm"),
                tvm_vs_m64,
            );
            results.extend(samples);
        }
    }

    // Summary: count where each variant wins / ties / loses vs M64.
    println!("\n==> headline: TVM vs M64 (the realistic baseline)");
    let mut tvm_wins = 0;
    let mut tvm_losses = 0;
    let mut tvm_ties = 0;
    let mut total_with_m64 = 0;
    let mut max_speedup = 0.0f64;
    let mut min_speedup = f64::INFINITY;
    for s in &results {
        if s.variant != "tvm" || s.speedup_vs_m64.is_none() {
            continue;
        }
        let r = s.speedup_vs_m64.unwrap();
        if r == 0.0 {
            continue;
        }
        total_with_m64 += 1;
        if r > 1.05 {
            tvm_wins += 1;
        } else if r < 0.95 {
            tvm_losses += 1;
        } else {
            tvm_ties += 1;
        }
        max_speedup = max_speedup.max(r);
        if r > 0.0 {
            min_speedup = min_speedup.min(r);
        }
    }
    println!(
        "    tvm wins:   {}/{}    tvm ties:  {}/{}    tvm loses: {}/{}",
        tvm_wins, total_with_m64, tvm_ties, total_with_m64, tvm_losses, total_with_m64
    );
    if total_with_m64 > 0 {
        println!(
            "    best TVM/M64 speedup: {:.2}x   worst: {:.2}x",
            max_speedup, min_speedup
        );
    }

    let out_dir = workspace_root().join("bench-framework/results");
    std::fs::create_dir_all(&out_dir)?;
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let path = out_dir.join(format!("all-{timestamp}.json"));
    std::fs::write(&path, serde_json::to_vec_pretty(&results)?)?;

    println!("\nwrote {} samples to {}", results.len(), path.display());
    println!(
        "\nU statistic: 0.5 no diff; <0.5 baseline (m32) faster; >0.5 variant faster.\n\
         speedup_vs_baseline: m32 mean / variant mean; >1.0 = variant beats m32."
    );
    Ok(())
}
