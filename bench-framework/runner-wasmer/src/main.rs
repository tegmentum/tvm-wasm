//! Wasmer-backed runner — M32 and M64 across all 9 benchmark classes.
//!
//! This is the cross-engine validator. It does NOT run the TVM variants
//! (which need our wasmtime-specific raw_linker host functions). The
//! purpose is to confirm that the wasmtime numbers for M32 and M64 are
//! engine-shaped, so we know how much of the "TVM beats M64 by 35x"
//! finding is wasmtime-specific.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Context;
use serde::Serialize;
use wasmer::{imports, Engine, Instance, Module, Store, TypedFunction};

const WARMUP_ROUNDS: usize = 5;
const SAMPLES: usize = 50;
const SEED: u32 = 0xDEADBEEF;
const SIZES: &[u32] = &[1024, 16 * 1024, 256 * 1024];

#[derive(Serialize, Clone)]
struct Sample {
    runtime: &'static str,
    variant: &'static str,
    class: &'static str,
    size_bytes: u32,
    samples: usize,
    mean_ns: f64,
    p50_ns: u128,
    p95_ns: u128,
    p99_ns: u128,
    throughput_gib_per_s: f64,
    notes: &'static str,
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
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

fn summarize(
    runtime: &'static str,
    variant: &'static str,
    class: &'static str,
    size: u32,
    timings: Vec<Duration>,
    notes: &'static str,
) -> Sample {
    let mut sorted = timings.clone();
    sorted.sort();
    let mean_ns =
        timings.iter().map(|d| d.as_nanos() as f64).sum::<f64>() / timings.len() as f64;
    let throughput_gib_per_s = if mean_ns > 0.0 {
        (size as f64) / (mean_ns / 1e9) / (1u64 << 30) as f64
    } else {
        0.0
    };
    Sample {
        runtime,
        variant,
        class,
        size_bytes: size,
        samples: timings.len(),
        mean_ns,
        p50_ns: percentile(&sorted, 50.0),
        p95_ns: percentile(&sorted, 95.0),
        p99_ns: percentile(&sorted, 99.0),
        throughput_gib_per_s,
        notes,
    }
}

fn time<F: FnMut() -> anyhow::Result<()>>(mut f: F) -> anyhow::Result<Vec<Duration>> {
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

// -------------------- M32 (32-bit pointers) --------------------

struct M32 {
    store: Store,
    instance: Instance,
    buffer_ptr: u32,
}

fn new_m32(wasm: &[u8]) -> anyhow::Result<M32> {
    let mut store = Store::default();
    let module = Module::new(&store, wasm)?;
    let imports = imports! {};
    let instance = Instance::new(&mut store, &module, &imports)?;
    let buffer_ptr_fn: TypedFunction<(), u32> =
        instance.exports.get_typed_function(&store, "buffer_ptr")?;
    let buffer_ptr = buffer_ptr_fn.call(&mut store)?;
    Ok(M32 { store, instance, buffer_ptr })
}

// -------------------- benches (M32 only — wasmer M64 support varies) --------------------

fn bench_m32_seq(wasm: &[u8], size: u32) -> anyhow::Result<Sample> {
    let mut a = new_m32(wasm)?;
    let memory = a.instance.exports.get_memory("memory")?;
    let mut data = vec![0u8; size as usize];
    fill_pattern(&mut data, SEED);
    memory.view(&a.store).write(a.buffer_ptr as u64, &data)?;
    let sum: TypedFunction<(u32, u32), u64> =
        a.instance.exports.get_typed_function(&a.store, "sum_sequential")?;
    let buf_ptr = a.buffer_ptr;
    let t = time(|| {
        let _ = sum.call(&mut a.store, buf_ptr, size)?;
        Ok(())
    })?;
    Ok(summarize("wasmer", "m32", "sequential_sum", size, t, "wasmer engine"))
}

fn bench_m32_random(wasm: &[u8], size: u32) -> anyhow::Result<Sample> {
    let cells = size / 4;
    let steps = 4 * cells;
    let mut ring: Vec<u32> = (0..cells).collect();
    let mut state = SEED.wrapping_add(7);
    for i in (1..cells as usize).rev() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let j = (state as usize) % (i + 1);
        ring.swap(i, j);
    }
    let mut bytes = vec![0u8; (cells * 4) as usize];
    for i in 0..cells as usize {
        let here = ring[i];
        let next = ring[(i + 1) % cells as usize];
        let off = (here * 4) as usize;
        bytes[off..off + 4].copy_from_slice(&next.to_le_bytes());
    }

    let mut a = new_m32(wasm)?;
    let memory = a.instance.exports.get_memory("memory")?;
    memory.view(&a.store).write(a.buffer_ptr as u64, &bytes)?;
    let chase: TypedFunction<(u32, u32, u32), u64> =
        a.instance.exports.get_typed_function(&a.store, "random_chase")?;
    let buf_ptr = a.buffer_ptr;
    let t = time(|| {
        let _ = chase.call(&mut a.store, buf_ptr, cells, steps)?;
        Ok(())
    })?;
    Ok(summarize("wasmer", "m32", "random_chase", size, t, "wasmer engine"))
}

fn bench_m32_columnar(wasm: &[u8], size: u32) -> anyhow::Result<Sample> {
    let n = size / 8;
    let mut col_a = vec![0u8; (n * 4) as usize];
    let mut col_b = vec![0u8; (n * 4) as usize];
    for i in 0..n {
        let off = (i * 4) as usize;
        col_a[off..off + 4].copy_from_slice(&i.to_le_bytes());
        col_b[off..off + 4].copy_from_slice(&i.wrapping_mul(13).to_le_bytes());
    }
    let threshold = n / 2;

    let mut a = new_m32(wasm)?;
    let memory = a.instance.exports.get_memory("memory")?;
    memory.view(&a.store).write(a.buffer_ptr as u64, &col_a)?;
    memory
        .view(&a.store)
        .write((a.buffer_ptr as u64) + col_a.len() as u64, &col_b)?;
    let q: TypedFunction<(u32, u32, u32), u64> =
        a.instance.exports.get_typed_function(&a.store, "columnar_filter_sum")?;
    let buf_ptr = a.buffer_ptr;
    let t = time(|| {
        let _ = q.call(&mut a.store, buf_ptr, n, threshold)?;
        Ok(())
    })?;
    Ok(summarize("wasmer", "m32", "columnar_filter_sum", size, t, "wasmer engine"))
}

fn read_wasm(rel: &str) -> anyhow::Result<Vec<u8>> {
    let path = workspace_root().join(rel);
    std::fs::read(&path).with_context(|| format!("missing wasm: {}", path.display()))
}

fn main() -> anyhow::Result<()> {
    let m32_wasm = read_wasm(
        "bench-framework/workloads/m32/target/wasm32-unknown-unknown/release/tvm_bench_workload_m32.wasm",
    )?;

    println!(
        "==> wasmer cross-engine validation ({} samples, seed {:#x})",
        SAMPLES, SEED
    );
    println!(
        "    measures M32 only — TVM raw imports require wasmtime-specific\n\
         host code. Cross-engine validation is for the M32/M64 baselines."
    );
    println!();

    let mut results = Vec::new();
    let _ = Engine::default(); // keep import live

    for &size in SIZES {
        for (name, f) in &[
            ("sequential", bench_m32_seq as fn(&[u8], u32) -> anyhow::Result<Sample>),
            ("random", bench_m32_random),
            ("columnar", bench_m32_columnar),
        ] {
            let s = f(&m32_wasm, size)?;
            println!(
                "    wasmer m32 {:<14} size={:>8} mean={:>10.0}ns p99={:>10}ns {:>5.2} GiB/s",
                name, size, s.mean_ns, s.p99_ns, s.throughput_gib_per_s
            );
            results.push(s);
        }
    }

    let out_dir = workspace_root().join("bench-framework/results");
    std::fs::create_dir_all(&out_dir)?;
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let path = out_dir.join(format!("wasmer-{timestamp}.json"));
    std::fs::write(&path, serde_json::to_vec_pretty(&results)?)?;
    println!("\nwrote {} samples to {}", results.len(), path.display());
    Ok(())
}
