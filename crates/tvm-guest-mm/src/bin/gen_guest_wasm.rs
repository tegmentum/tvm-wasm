//! Generate a deployable multi-memory guest `.wasm` (or `.wat`).
//!
//! The browser Cold-tier embedding (`web/tvm-web`) ships a concrete
//! `.wasm`, not runtime-generated WAT. This binary emits the default
//! 64-pool TVM-MM module — N exported memories (`mem0..memN`) plus the
//! exported dispatch helpers (`tvm_load_u32`, `tvm_store_u32`,
//! `tvm_copy_to_default`, …) — and writes it out.
//!
//! Usage:
//!   gen_guest_wasm [--pools N] [--max-pages-per-pool N] [--wat] <out_path>
//!
//! With `--wat` the raw WAT text is written instead of compiled bytes
//! (handy for inspection / `wasm-tools print` diffing).
//!
//! `--max-pages-per-pool` sets the declared wasm memory maximum per pool
//! (each page = 64 KiB). The default of 65536 (4 GiB) is fine on host
//! wasmtime and Node's V8; browser V8 reserves virtual address space up
//! to that maximum per exported memory for its trap-handler fast path,
//! so `pools × max_pages × 64 KiB` must fit the per-tab budget — pass
//! a smaller value (e.g. 256 → 16 MiB per pool) for browser builds.

use std::path::Path;
use std::process::ExitCode;

use tvm_guest_mm::{tvm_guest_mm_module_template, ModuleParams, DEFAULT_POOL_COUNT};

fn main() -> ExitCode {
    let defaults = ModuleParams::default();
    let mut pools = DEFAULT_POOL_COUNT;
    let mut max_pages = defaults.max_pages_per_pool;
    let mut emit_wat = false;
    let mut out: Option<String> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--pools" => {
                let Some(v) = args.next() else {
                    return fail("--pools needs a value");
                };
                match v.parse() {
                    Ok(n) => pools = n,
                    Err(_) => return fail("--pools value must be an integer"),
                }
            }
            "--max-pages-per-pool" => {
                let Some(v) = args.next() else {
                    return fail("--max-pages-per-pool needs a value");
                };
                match v.parse::<u32>() {
                    Ok(0) => return fail("--max-pages-per-pool must be >= 1"),
                    Ok(n) if n > 65536 => {
                        return fail("--max-pages-per-pool exceeds wasm32 cap (65536)")
                    }
                    Ok(n) => max_pages = n,
                    Err(_) => return fail("--max-pages-per-pool value must be an integer"),
                }
            }
            "--wat" => emit_wat = true,
            "-h" | "--help" => {
                println!(
                    "usage: gen_guest_wasm [--pools N] [--max-pages-per-pool N] [--wat] <out_path>"
                );
                return ExitCode::SUCCESS;
            }
            other if !other.starts_with('-') => out = Some(other.to_string()),
            other => return fail(&format!("unknown flag: {other}")),
        }
    }

    let Some(out) = out else {
        return fail("missing <out_path>");
    };

    if max_pages < defaults.initial_pages_per_pool {
        return fail(&format!(
            "--max-pages-per-pool ({max_pages}) < initial pages ({})",
            defaults.initial_pages_per_pool
        ));
    }

    let params = ModuleParams {
        n_pools: pools,
        max_pages_per_pool: max_pages,
        ..defaults
    };
    let wat = tvm_guest_mm_module_template(&params);

    if let Some(parent) = Path::new(&out).parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                return fail(&format!("create {}: {e}", parent.display()));
            }
        }
    }

    let result = if emit_wat {
        std::fs::write(&out, wat.as_bytes())
    } else {
        match wat::parse_str(&wat) {
            Ok(bytes) => std::fs::write(&out, bytes),
            Err(e) => return fail(&format!("WAT did not compile: {e}")),
        }
    };

    match result {
        Ok(()) => {
            eprintln!(
                "wrote {out} ({pools} pools, max {max_pages} pages/pool, {} bytes WAT)",
                wat.len()
            );
            ExitCode::SUCCESS
        }
        Err(e) => fail(&format!("write {out}: {e}")),
    }
}

fn fail(msg: &str) -> ExitCode {
    eprintln!("gen_guest_wasm: {msg}");
    ExitCode::FAILURE
}
